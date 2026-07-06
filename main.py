"""nanoagent v2 CLI 入口 — channel 薄壳。

装配：LLMClient + ToolRegistry + MainLoop + SessionStore + SkillLoader + UserFacts
      + Harness（channel-agnostic orchestrator）

两种运行模式由顶部常量 `TASK` / `TASK_FILE` 切换：
- 两者非空 → one-shot 模式（stateless，不过 Harness / Session / UserFacts）
- 都为空 → REPL 模式（CLI channel 只管 stdin/stdout，编排走 harness.handle）

改参数就改顶部常量。不走 argparse，不走 env（env 仅 API key / OPENAI_MODEL_ID）。
"""

import os
import signal
import sys
from pathlib import Path
from typing import Optional

from nanoagent.core.llm_client import LLMClient
from nanoagent.core.prompt_assets import load_prompt
from nanoagent.runtime.subagent import SubAgentRunner, build_subagent_llm
from nanoagent.memory.global_memory import GlobalMemory
from nanoagent.memory.global_memory_distiller import GlobalMemoryDistiller
from nanoagent.evolution.runtime_memory.lesson_ingestor import LessonIngestor
from nanoagent.evolution.runtime_memory.lesson_retriever import LessonRetriever
from nanoagent.evolution.runtime_memory.online_reflector import OnlineReflector
from nanoagent.evolution.runtime_memory.outcome_tracker import OutcomeTracker
from nanoagent.evolution.runtime_memory.sqlite_backend import (
    DEFAULT_DB_PATH as _DEFAULT_LESSONS_DB_PATH,
    SqliteMemoryBackend,
)
from nanoagent.memory.user_facts import UserFacts
from nanoagent.core.paths import data_dir
from nanoagent.runtime.context_budget import ContextBudgetConfig
from nanoagent.runtime.harness import Harness
from nanoagent.runtime.interrupt import RunInterrupt
from nanoagent.runtime.main_loop import MainLoop
from nanoagent.runtime.session import SessionStore
from nanoagent.runtime.token_counter import TokenCounter
from nanoagent.runtime.tool_output_store import ToolOutputStore
from nanoagent.skills.loader import SkillLoader
from nanoagent.tool.arxiv import ArxivTool
from nanoagent.tool.describe_script import DescribeScriptTool
from nanoagent.tool.fetch import FetchTool
from nanoagent.tool.glob import GlobTool
from nanoagent.tool.grep import GrepTool
from nanoagent.tool.load_skill import LoadSkillTool
from nanoagent.tool.registry import ToolRegistry
from nanoagent.tool.search import SearchTool
from nanoagent.tool.skill_exec import SkillExecTool
from nanoagent.tool.subagent_tool import SubAgentTool

# ============================================================
# 运行参数（改这里就改行为，不需要命令行）
# ============================================================

# 任务模式：两者都为空进 REPL；非空进 one-shot
TASK: str = ""
TASK_FILE: str = ""  # 路径字符串，按 utf-8 读取

# LLM 参数
MODEL: str = os.getenv("OPENAI_MODEL_ID", "gpt-5.4-mini")
PROVIDER: str = ""  # 留空 = openai；可选 dashscope / dashscope_us / zhipu / deepseek / ollama / vllm
# provider 透传 body（如 qwen 的 enable_thinking=False 关思考——agent 工具循环不需要
# 长推理，关掉快 ~8×、省 ~10× 输出 token）。None = 不透传。eval driver 可 patch 注入。
MODEL_EXTRA_BODY: Optional[dict] = None
MAX_ITERATIONS: int = 20

# P0.3 副 LLM Evaluator（日报质量评审）。DASHSCOPE_API_KEY 未设时跳过。
EVALUATOR_MODEL: str = "qwen-plus"
EVALUATOR_PROVIDER: str = "dashscope"
EVALUATOR_TIMEOUT: int = 15  # 秒；副 LLM 超过即 fail-open 走 finalize
EVALUATOR_MAX_RETRIES: int = 1  # advisor 原话"非常小但非常硬"，一次 retry 足够

# 在线微反思（docs/81 铁律修正案）：第 2 次同 op 失败且飞轮无可召回的
# suggested_action 时，副 LLM 基于报错产一个 [online-reflector] 修复假设。
# 默认关：未经小测验证 + eval 可比性；DASHSCOPE_API_KEY 缺失也自然降级。
ONLINE_REFLECTOR_MODEL: str = "qwen3.6-flash"
ONLINE_REFLECTOR_PROVIDER: str = "dashscope"
ONLINE_REFLECTOR_TIMEOUT: int = 15
ONLINE_REFLECTOR_EXTRA_BODY: Optional[dict] = {"enable_thinking": False}

# 持久化路径
CLI_SESSION_KEY: str = "cli:default"
SESSION_DIR: Path = Path("data/runtime/sessions")
SKILLS_DIR: Path = Path("skills")
USER_FACTS_PATH: Path = Path("data/memory/user_facts.json")
AUTO_MEMORY_PATH: Path = Path("data/memory/auto_memory.md")  # 全局自动记忆（Dream-lite）

# SubAgent 研究子 agent（联网 search + fetch，结论隔离回主对话）
SUBAGENT_RESEARCH_MAX_ITER: int = 6
SUBAGENT_RESEARCH_PROMPT: str = """你是一个研究子 agent。用 search / fetch 工具就给定任务查证，返回简洁、有依据的结论。
- 先 search 找来源，再 fetch 读关键页面。
- 只回结论 + 关键依据 / 链接，不要过程独白。
- 查不到就如实说，不要编造。"""
ARXIV_STORAGE_DIR: Path = Path("data/arxiv_papers")  # arxiv-mcp-server 本地存储
LESSONS_DB_PATH: Path = _DEFAULT_LESSONS_DB_PATH  # SSOT 在 sqlite_backend.DEFAULT_DB_PATH
def _env_bool(name: str, default: bool) -> bool:
    """env override helper：未设 / 空字符串 → default；'0'/'false'/'no' → False；其他 truthy → True。

    eval harness 用 NANOAGENT_ENABLE_LESSON_RECALL=0 跑 baseline、=1 跑 evolved，
    同一份代码两种模式不必改源文件。"""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


ENABLE_LESSON_RECALL: bool = _env_bool("NANOAGENT_ENABLE_LESSON_RECALL", True)  # Phase C：第 1 次失败时也查 backend 中的 promoted lesson
ENABLE_ONLINE_REFLECTOR: bool = _env_bool("NANOAGENT_ONLINE_REFLECTOR", False)  # 在线微反思，默认关（待小测验证）
# 滑窗失败率总闸（Track A-b.4）。默认开（生产兜底）。Track E eval 关掉：实测它在 R2
# 上抢在熔断器降级完成前掐断、抹平熔断收益（详见 main_loop 注释）。
ENABLE_FAILURE_RATE_GATE: bool = _env_bool("NANOAGENT_ENABLE_FAILURE_RATE_GATE", True)

# 刀4：PromotionGate/状态机已删——lesson 治理改为 lesson_score 从账本派生的
# 单分数注入闸（score ≥ T）。晋升/降级/退休全塌成"过没过线",无独立 gate。

# P4-lite D-context：Context Hygiene Foundation 默认开启
ENABLE_CONTEXT_HYGIENE: bool = True
CONTEXT_MAX_TOKENS: int = 80_000
# 全局自动记忆（Dream-lite）触发参数：撑爆 90% / 切换 50%（占 CONTEXT_MAX_TOKENS 比例）。
# 每点一个 per-session 布尔、各最多触发一次（防同一 session 反复进出反复蒸）。
AUTO_DISTILL_OVERFLOW_RATIO: float = 0.9
AUTO_DISTILL_SWITCH_RATIO: float = 0.5
CONTEXT_WARN_RATIO: float = 0.8
CONTEXT_HARD_RATIO: float = 0.95
TOOL_OUTPUT_MAX_SINGLE_TOKENS: int = 2_000
TOOL_OUTPUT_MAX_TOTAL_TOKENS: int = 12_000
TOOL_OUTPUTS_DIR_NAME: str = "tool_outputs"

# Session 级上下文体量软告警阈值（token）。历史滚到此值以上，每个 session 首次
# 越过时在回复末尾附一句"建议 /new"。只量+提醒、不自动截断历史。0 = 关闭。
SESSION_HISTORY_WARN_TOKENS: int = 60_000

BASE_IDENTITY: str = """# Role
你是一个具备高度自主权与逻辑分析能力的智能 Agent 核心。通过精准的逻辑推理和工具调用，高效、闭环地解决用户问题。

# 决策方式
每轮"调工具 vs 直接答"之前，简要判断：
- 任务是否需要外部信息？已有上下文够不够？
- 若调工具，哪个最直接命中？（不凭直觉选熟悉的工具）
- 多条信息可并行？单次响应并发调用降低延迟
- 上下文充足时直接结构化回答，不做无效工具调用
- 根据工具返回的实际反馈随时调整策略——换关键词、切换工具、或总结收尾

# 错误处理（Critical）
主动监控工具运行状态，严禁静默忽略失败：

1. **识别失败**：工具返回含 `[... 错误]` / `[error]` / "退出码" 非 0 / Traceback / Exception 等失败标志，视为执行异常
2. **先尝试修复**：换参数、换替代工具、或基于已获取的部分信息降级完成
3. **修复成功** → 在最终输出末尾简短标注受影响范围（如"论文部分因 HF API 异常，改用 arxiv 替代"），不必打断主流程
4. **修复失败** → 明确告知用户：失败的工具名 / 报错关键点 / 为什么无法继续 / 是否需要进一步指令

# Response Standards
- **结构化**：优先使用 Markdown 标题、列表、表格
- **高信息增益**：拒绝废话，直接呈现核心数据、结论、代码
- **溯源**：引用工具数据时保留原始来源或链接"""

# md 资产覆盖（prompts/base_identity.md）；文件缺失/正文空 → 回退上面默认。
# run_bot 经 cli_main.BASE_IDENTITY 读同一份。
BASE_IDENTITY = load_prompt("base_identity", BASE_IDENTITY)


# ============================================================
# 装配
# ============================================================

def read_utf8_file(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"任务文件不存在: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_registry(skill_loader: SkillLoader) -> ToolRegistry:
    """注册所有可被 LLM 调的 BaseTool。

    skill_loader 作为依赖注入：LoadSkillTool 在 schema 里动态列出当前 skills，
    热增删 skill 后调 loader.reload() 即反映到下一轮 LLM 调用。
    """
    registry = ToolRegistry()
    registry.register(FetchTool())
    registry.register(SkillExecTool(skills_dir=SKILLS_DIR))
    registry.register(DescribeScriptTool(skills_dir=SKILLS_DIR))
    registry.register(LoadSkillTool(skill_loader=skill_loader))
    registry.register(ArxivTool(storage_path=ARXIV_STORAGE_DIR))
    # grep / glob 接替原 CalcTool 当 strict baseline：schema 完全静态 → strict_mode
    # 默认开着，与 SkillExecTool 的 free-form args（strict 表达不了、保持关）形成对照，
    # 是"按能力而非厂商决定 strict"的活样本。比 calc 那个纯 probe 多了真实用途。
    registry.register(GrepTool())
    registry.register(GlobTool())
    if os.getenv("TAVILY_API_KEY") or os.getenv("EXA_API_KEY"):
        registry.register(SearchTool())
    # SubAgent 委派：一个能联网 search + fetch 的研究子 agent，暴露给主 LLM。
    # 隔离价值——啰嗦的搜索 / 抓取过程留在子 agent context，只回结论，不污染主对话。
    # runner 共享本 registry：裁剪时按 allowlist 留 search/fetch，剔除 is_subagent（防套娃）。
    research_runner = SubAgentRunner(llm_builder=build_subagent_llm, base_tool_registry=registry)
    registry.register(SubAgentTool(
        name="research",
        description="委派一个能联网搜索 + 抓取网页的子 agent 完成研究 / 查证子任务，只回结论（隔离上下文，不污染主对话）",
        system_prompt=SUBAGENT_RESEARCH_PROMPT,
        runner=research_runner,
        allowed_tools=("web_search", "fetch"),
        max_iterations=SUBAGENT_RESEARCH_MAX_ITER,
    ))
    return registry


def build_runtime_memory(
    *,
    sqlite_check_same_thread: bool = True,
) -> tuple[
    Optional[LessonRetriever],
    Optional[OutcomeTracker],
    Optional[LessonIngestor],
]:
    """Phase C/D/B：装配 SqliteMemoryBackend → 共享给 retriever / outcome / ingestor。

    sqlite_check_same_thread:
        CLI 默认 True（main thread 装配 + 使用，sqlite3 自带保护）。
        Telegram bot（run_bot.py）传 False——main thread 装配 + 单 worker
        thread 使用，配合 channel 内置 single-worker ThreadPoolExecutor 串行
        消费。

    `ENABLE_LESSON_RECALL=False` 时四 None → MainLoop / Harness 退化到 P0.2 行为。
    `ENABLE_PROMOTION_GATE=False` 时 retriever / outcome / ingestor 仍正常，仅 gate 为 None。
    backend 异常（库文件损坏 / 权限）时全部降级 None，不阻断 agent 启动。

    生命周期：atexit 注册 backend.close()，进程退出时 commit + 释放 SQLite 句柄。

    所有组件共享同一 backend：retriever 读 active lesson（score≥T）注入 hint；
    outcome_tracker 写已用 lesson 的 helped/hurt 账本；ingestor 把 trace 失败
    转 candidate lesson 入库——三者同 SQLite 文件，事务一致。晋降退休由 lesson_score
    从账本派生,无独立 gate。
    """
    if not ENABLE_LESSON_RECALL:
        return None, None, None
    try:
        import atexit
        backend = SqliteMemoryBackend(
            db_path=LESSONS_DB_PATH,
            check_same_thread=sqlite_check_same_thread,
        )
        atexit.register(backend.close)
        return (
            LessonRetriever(backend),
            OutcomeTracker(backend),
            LessonIngestor(backend),
        )
    except Exception as e:
        print(f"[main] runtime_memory 装配失败（已降级，agent 仍可正常启动）: {e}")
        return None, None, None


def build_context_hygiene() -> tuple[
    Optional[TokenCounter], Optional[ToolOutputStore], Optional[ContextBudgetConfig]
]:
    """P4-lite D-context：装配 TokenCounter / ToolOutputStore / ContextBudgetConfig。

    `ENABLE_CONTEXT_HYGIENE=False` 时三件全 None → MainLoop 退化到 Phase 3a/B+ 行为
    （chars/4 估算 + max_tool_output_chars 字符截断 + 单点 warning），向后兼容。

    ToolOutputStore 无常驻句柄，每次 store_if_needed 一次性 open-write-close，
    不需要 atexit close。
    """
    if not ENABLE_CONTEXT_HYGIENE:
        return None, None, None
    counter = TokenCounter()
    tool_outputs_dir = data_dir(TOOL_OUTPUTS_DIR_NAME)
    store = ToolOutputStore(
        base_dir=tool_outputs_dir,
        token_counter=counter,
        max_single_tokens=TOOL_OUTPUT_MAX_SINGLE_TOKENS,
    )
    budget_cfg = ContextBudgetConfig(
        max_context_tokens=CONTEXT_MAX_TOKENS,
        warn_ratio=CONTEXT_WARN_RATIO,
        hard_ratio=CONTEXT_HARD_RATIO,
        max_single_tool_tokens=TOOL_OUTPUT_MAX_SINGLE_TOKENS,
        max_total_tool_tokens_per_run=TOOL_OUTPUT_MAX_TOTAL_TOKENS,
    )
    return counter, store, budget_cfg


def build_loop(
    skill_loader: SkillLoader,
    *,
    lesson_retriever: Optional[LessonRetriever] = None,
    interrupt: Optional[RunInterrupt] = None,
) -> MainLoop:
    """构造 MainLoop。lesson_retriever 由调用方装配——避免本函数重复 build
    SqliteBackend（Phase D 引入 OutcomeTracker 后需共享同一 backend）。

    PR 2 + advisor §3.B：装配期扫所有 skill 的 coverage manifest（skill.yaml）→
    union 合并 specs 传给 MainLoop。skill 未声明 coverage 段时返空列表 →
    MainLoop 跳过 coverage 检查（framework 不内建任何 skill 知识）。
    """
    llm = LLMClient(
        model=MODEL,
        provider=PROVIDER or None,
        instance_name="main_loop",
        extra_body=MODEL_EXTRA_BODY,
    )
    counter, store, budget_cfg = build_context_hygiene()
    online_reflector = None
    if ENABLE_ONLINE_REFLECTOR and os.getenv("DASHSCOPE_API_KEY"):
        reflector_llm = LLMClient(
            model=ONLINE_REFLECTOR_MODEL,
            provider=ONLINE_REFLECTOR_PROVIDER,
            instance_name="online_reflector",
            timeout=ONLINE_REFLECTOR_TIMEOUT,
            extra_body=ONLINE_REFLECTOR_EXTRA_BODY,
        )
        online_reflector = OnlineReflector(llm=reflector_llm)
        print(f"[OnlineReflector] enabled: {ONLINE_REFLECTOR_PROVIDER}/"
              f"{ONLINE_REFLECTOR_MODEL} (timeout={ONLINE_REFLECTOR_TIMEOUT}s)")
    return MainLoop(
        name="assistant",
        llm=llm,
        tool_registry=build_registry(skill_loader),
        max_iterations=MAX_ITERATIONS,
        enable_failure_rate_gate=ENABLE_FAILURE_RATE_GATE,
        lesson_retriever=lesson_retriever,
        online_reflector=online_reflector,
        token_counter=counter,
        tool_output_store=store,
        context_budget_config=budget_cfg,
        interrupt=interrupt,
    )


def resolve_task() -> str:
    if TASK:
        return TASK.strip()
    if TASK_FILE:
        return read_utf8_file(TASK_FILE)
    return ""


def print_llm_instances() -> None:
    """打印当前已创建的 LLM 实例 -> 模型映射。"""
    llm_rows = LLMClient.describe_instances()
    if not llm_rows:
        return
    print("[LLM Instances]")
    for idx, row in enumerate(llm_rows, 1):
        print(f"  {idx}. {row['instance_name']} -> {row['model']}")


# ============================================================
# 运行模式
# ============================================================

def run_once(loop: MainLoop, task: str) -> int:
    """One-shot，stateless，不接 Session / UserFacts / Skill。"""
    if not task:
        print("任务不能为空。")
        return 1
    print(f"Task: {task}\n")
    answer = loop.run([{"role": "user", "content": task}])
    print("=" * 60)
    print(answer)
    print(f"\n({loop.llm.usage})")
    return 0


def _run_turn_interruptible(harness: Harness, text: str, interrupt: RunInterrupt):
    """跑一个 turn，期间把 Ctrl-C 转成协作式打断（步边界停）。

    turn 外（idle 的 input()）仍是默认 KeyboardInterrupt 退出语义——本函数只在 turn
    期间临时接管 SIGINT，结束后原样还回。非主线程 / 平台不支持装 handler 时退回普通
    handle（无协作式打断，行为不变）。
    """
    interrupt.reset()
    requested_once = {"v": False}

    def _on_sigint(signum, frame):
        if not requested_once["v"]:
            requested_once["v"] = True
            print("\n[打断] 已请求停止，当前步跑完即停…", flush=True)
        interrupt.request()

    try:
        prev = signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, OSError):
        return harness.handle(text)
    try:
        return harness.handle(text)
    finally:
        signal.signal(signal.SIGINT, prev)


def repl(harness: Harness, interrupt: RunInterrupt) -> int:
    """CLI channel：只管 stdin/stdout，编排走 harness.handle。"""
    header = "nanoagent v2 REPL。"
    n_sessions = len(harness.list_sessions())
    if n_sessions > 0:
        header += f"已存 {n_sessions} 个历史会话（首句消息默认开新会话；用 /resume <key> 续上）。"
    header += (
        "\n命令：/skills | /skill <name> | /profile | "
        "/sessions [all|<n>] | /new (=/clear) | /resume <key> | /rewind [n] | exit"
        "\n（跑任务时按 Ctrl-C 协作式打断：当前步跑完即停，不丢 trace）"
    )
    print(header)

    while True:
        try:
            tag = f"[{harness.current_skill}] " if harness.current_skill else ""
            text = input(f"\n{tag}>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0

        if text.lower() in {"exit", "quit"}:
            print("已退出。")
            return 0
        if not text:
            continue

        resp = _run_turn_interruptible(harness, text, interrupt)
        if resp.kind == "system":
            print(resp.content)
        else:
            print("\n" + "=" * 60)
            print(resp.content)
            if resp.usage:
                print(f"\n({resp.usage})")


# ============================================================
# 入口
# ============================================================

def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    # stdin 也强制 UTF-8——非交互场景（codex / pipe）下父进程可能写 UTF-8 字节
    # 但 Windows Python 默认按 cp936 解，触发 errors='replace' 把中文变 '?'。
    # errors='replace' 这里是兜底（如果父进程真写了非 UTF-8 字节，至少不崩）
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    # 装配顺序：loader → runtime_memory（共享 backend） → loop
    # LoadSkillTool 依赖 loader 的动态 description，必须先建好 loader
    # Phase D：lesson_retriever + outcome_tracker 共享同一 SqliteBackend
    loader = SkillLoader(SKILLS_DIR)
    lesson_retriever, outcome_tracker, lesson_ingestor = build_runtime_memory()
    # 协作式打断标志：REPL 的 Ctrl-C 处理器置位，MainLoop 步边界查（one-shot 也传，
    # 走 run() 的 KeyboardInterrupt 兜底路径）。
    interrupt = RunInterrupt()
    loop = build_loop(loader, lesson_retriever=lesson_retriever, interrupt=interrupt)

    task = resolve_task()
    if task:
        print_llm_instances()
        return run_once(loop, task)

    store = SessionStore(persist_dir=SESSION_DIR)
    user_facts = UserFacts(USER_FACTS_PATH)

    # 副 LLM critic（发布流程质量评审，非阻断）：守 DASHSCOPE_API_KEY，缺失 → None。
    # EVALUATOR_* 常量沿用为 critic 子 LLM 的配置（model/provider/timeout/封顶轮数）。
    has_dashscope = bool(os.getenv("DASHSCOPE_API_KEY"))

    critic_llm = None
    if has_dashscope:
        critic_llm = LLMClient(
            model=EVALUATOR_MODEL,
            provider=EVALUATOR_PROVIDER,
            instance_name="critic",
            timeout=EVALUATOR_TIMEOUT,
        )
        print(f"[Critic] enabled: {EVALUATOR_PROVIDER}/{EVALUATOR_MODEL} "
              f"(timeout={EVALUATOR_TIMEOUT}s, max_revise={EVALUATOR_MAX_RETRIES})")

    print_llm_instances()

    # 全局自动记忆（Dream-lite）：GlobalMemory 总建（只读也要能注入 prompt）；distiller
    # 守 DASHSCOPE_API_KEY，缺失 → None（记忆只读、不更新）。
    global_memory = GlobalMemory(AUTO_MEMORY_PATH)
    global_memory_distiller = (
        GlobalMemoryDistiller(global_memory, SubAgentRunner(llm_builder=build_subagent_llm))
        if has_dashscope else None
    )
    if global_memory_distiller is not None:
        print("[GlobalMemory] auto-distill enabled")

    harness = Harness(
        loop=loop,
        store=store,
        loader=loader,
        user_facts=user_facts,
        session_key=CLI_SESSION_KEY,
        base_identity=BASE_IDENTITY,
        critic_llm=critic_llm,
        critic_max_revise=EVALUATOR_MAX_RETRIES,
        outcome_tracker=outcome_tracker,
        lesson_ingestor=lesson_ingestor,
        session_history_warn_tokens=SESSION_HISTORY_WARN_TOKENS,
        global_memory=global_memory,
        global_memory_distiller=global_memory_distiller,
        context_window_tokens=CONTEXT_MAX_TOKENS,
        auto_distill_overflow_ratio=AUTO_DISTILL_OVERFLOW_RATIO,
        auto_distill_switch_ratio=AUTO_DISTILL_SWITCH_RATIO,
    )
    return repl(harness, interrupt)


if __name__ == "__main__":
    raise SystemExit(main())
