"""Generate raw multi-turn mutation fixtures for nanoagent eval.

Stage 2 of multi-turn fixture pipeline (Stage 1 = 6 hand-written `mt_failretry_*`
fixtures focusing on the "missing fingerprint" failure mode the spike confirmed
the flywheel can help on).

Stage 2 goal: cover OTHER multi-turn patterns + OTHER failure types LLM-style,
to test breadth (drill-down / correct-redo / cross-query mark / consistency /
progressive coverage_gap / iterative soft_quality / multi-turn conflict).

Output goes to evals/tasks/mt_mutations/raw_<HHMMSS>.yaml for human review;
curate via curate_mt_mutations.py.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# ============================================================
# 运行参数
# ============================================================
SEED_TASK_DIR: Path = Path("evals/tasks")
OUTPUT_DIR: Path = Path("evals/tasks/mt_mutations")
SEED_FIXTURE_PREFIXES: tuple[str, ...] = ("mt_",)  # 只把 mt_* 喂给 LLM 当种子

# LONG_HORIZON_MODE=True：生成 6-8 个 5-8 turn 长 fixture（业界 multi-turn benchmark
# 通常 8-16 turn，如 τ-bench / MINT；当前 mt_* 全 2-3 turn 偏短）
# False：生成 12-15 个 2-4 turn 短 fixture（覆盖多模式，跑得快）
LONG_HORIZON_MODE: bool = True
TARGET_MIN: int = 6 if LONG_HORIZON_MODE else 12
TARGET_MAX: int = 8 if LONG_HORIZON_MODE else 15


# ============================================================
# LLM 参数
# ============================================================
ANTHROPIC_API_URL: str = "https://api.anthropic.com/v1/messages"
MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_TOKENS: int = 16000
TEMPERATURE: float | None = None


def _read_seed_tasks(tasks_dir: Path, prefixes: tuple[str, ...]) -> str:
    parts: list[str] = []
    for path in sorted(tasks_dir.glob("*.yaml")):
        if not any(path.name.startswith(p) for p in prefixes):
            continue
        text = path.read_text(encoding="utf-8")
        parts.append(f"# {path.name}\n{text.strip()}\n")
    if not parts:
        raise FileNotFoundError(
            f"未找到种子 fixture (前缀={prefixes}) 在 {tasks_dir}"
        )
    return "\n---\n".join(parts)


def _build_prompt(seed_yaml: str) -> str:
    if LONG_HORIZON_MODE:
        turn_constraint = (
            "**每条 fixture 5-8 轮**（长 horizon 多轮）——模拟 Telegram 真实日常对话，"
            "用户多次追问、修正、加新条目、确认。轮数太短不算长 horizon。"
        )
        turn_constraint_short = "至少 5 项，至多 8 项"
        coverage_instruction = (
            f"请生成 **{TARGET_MIN}-{TARGET_MAX}** 条 5-8 轮长 fixture。"
            "覆盖以下**真实长对话场景**（每类至少 1-2 个）：\n"
            "1. **drill_down 长链**：用户反复追问展开主题（5-7 轮），最后一轮 mark\n"
            "2. **跨主题混合**：用户先聊 paper、再追 oss、再问 news，最后混合 mark\n"
            "3. **修正叠加**：用户多次改主意（先这样 → 不对那样 → 又改）\n"
            "4. **失败长链**：连续多轮触发同类失败模式 → 最终一轮飞轮该 helped\n"
            "5. **session 内反复 dup_check check 后 mark**：用户多次询问"
            "'这条记过没'，最后批量 mark"
        )
    else:
        turn_constraint = "每条 fixture 通常 2-4 轮（短多轮，覆盖多模式跑得快）"
        turn_constraint_short = "至少 2 项，至多 4 项"
        coverage_instruction = (
            f"请生成 **{TARGET_MIN}-{TARGET_MAX}** 条新 fixture 覆盖**未被种子覆盖的多轮模式 + 失败类型**：\n"
            "1. 跨轮收藏 cross_query_mark\n2. 追问加深 drill_down\n3. 修正回滚 correct_redo\n"
            "4. 去重一致性 consistency\n5. 多轮 coverage_gap\n6. 多轮 soft_quality_iterative\n"
            "7. 多轮 conflict"
        )
    return f"""你在帮一个叫 nanoagent 的 self-improving agent 项目设计**多轮**eval fixture。

# agent 背景
- 调一个叫 ai-digest 的 skill，工具有 fetch_hf（HF 论文）/ fetch_rss（多源新闻）/
  fetch_github（trending repos）/ dup_check（去重 / mark）
- ObligationTracker：用户说 "记下来" / "记一下" / "标记" / "保存" / "收藏" / "记录"
  等关键词会强制 agent 必须调 dup_check action=mark，不调即 obligation violation
- coverage 检查：默认 paper≥3, oss≥2, news≥1，不达标会触发 evaluator retry
- 失败分类（trace_error_classifier）：
  - schema_mismatch（args 错 / 缺必填字段）
  - repeated_same_args_failure（同 args 反复失败）
  - coverage_gap（类目数量不达标）
  - soft_quality_issue（evaluator 判低质量）
  - tool_runtime_error（工具运行时异常）

# 多轮 fixture 设计要求
- `query` 是 **List[str]**（YAML 里写成 `query:` 后跟 `- "..."` 列表）
- {turn_constraint}
- **决定性步骤（如 mark obligation）放最后一轮**——grader 评分时
  累加所有轮的飞轮命中 / obligation / tool_calls，但 success / coverage
  / final_answer 取最后一轮
- 前面轮次用来建立上下文 / 触发失败 / 中间状态
- query 要像真实 Telegram 用户口语，不要写成测试说明

# 种子 fixtures（已存在，覆盖"失败重试·缺 fingerprint"模式）

{seed_yaml}

# 任务

种子已经覆盖 `mt_failretry_*` 模式（用户首轮缺 fp → 补全 → mark）。

{coverage_instruction}

## 字段规约

每条 fixture 必须包含：
```yaml
id: mt_<模式>_<场景>             # snake_case，必须 mt_ 开头
description: <一句话场景 + 设计意图>
query:                            # List[str]，2-4 轮
  - "用户第 1 轮说的话"
  - "用户第 2 轮说的话"
  - "用户第 3 轮说的话"
expected_obligations:             # 仅当最后一轮触发 mark obligation 时填
  - mark_digest_when_user_asks_to_record
expected_tool_calls:              # 最后一轮 grader 期望命中的 tool call 模式
  - {{tool: skill_exec, skill: ai-digest, script: dup_check}}
expected_coverage: []             # 多轮累加；通常多轮 mark 场景留空
max_iterations: 25
success_keywords:                 # 最后一轮 final answer 应含的关键词
  - "已记录"
expected_failure_type: "<模式 / 失败类型说明 + 飞轮该如何 helped>"
expected_pattern: cross_query_mark|drill_down|correct_redo|consistency|coverage_gap_progressive|soft_quality_iterative|conflict_multi_turn
```

# 输出格式

- 输出单个 YAML 文档流，用 `---` 分隔每条 fixture
- 不要输出 markdown 代码围栏（```yaml）
- 不要输出解释文字、不要写"以下是 fixture"
- 总条数 {TARGET_MIN}-{TARGET_MAX}
- id 必须 mt_ 开头且 unique（长 horizon 模式建议加 _long_ 中缀，便于 review）
- query list {turn_constraint_short}

# 质量要求

1. 真实 Telegram 用户口语风格，不要造作
2. 多轮故事性——前后逻辑链顺畅，不要随机拼凑
3. 决定性步骤必须落最后一轮，否则 MVP grader 测不到
4. 不要简单改写种子（70% 以上 query 重叠会被丢弃）
5. expected_failure_type / expected_pattern 是 review-only 字段（curator 会 strip）

现在直接输出 YAML。
"""


def _post_anthropic(prompt: str) -> dict[str, Any]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 ANTHROPIC_API_KEY")

    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }
    if TEMPERATURE is not None:
        payload["temperature"] = TEMPERATURE
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=body,
        headers={
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic API HTTP {exc.code}: {detail}") from exc


def _extract_text(response: dict[str, Any]) -> str:
    chunks: list[str] = []
    for block in response.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            chunks.append(str(block.get("text", "")))
    text = "\n".join(chunks).strip()
    if not text:
        raise RuntimeError("Anthropic API 返回中没有 text content")
    return text


def _count_yaml_docs(raw_yaml: str) -> int | None:
    try:
        docs = [doc for doc in yaml.safe_load_all(raw_yaml) if doc is not None]
    except yaml.YAMLError:
        return None
    return len(docs)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    load_dotenv()

    seed_yaml = _read_seed_tasks(SEED_TASK_DIR, SEED_FIXTURE_PREFIXES)
    prompt = _build_prompt(seed_yaml)
    response = _post_anthropic(prompt)
    raw_yaml = _extract_text(response)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    prefix = "raw_long_" if LONG_HORIZON_MODE else "raw_"
    output_path = OUTPUT_DIR / f"{prefix}{timestamp}.yaml"
    output_path.write_text(raw_yaml + "\n", encoding="utf-8")

    usage = response.get("usage") or {}
    doc_count = _count_yaml_docs(raw_yaml)
    count_text = "unknown" if doc_count is None else str(doc_count)
    print(f"[mt-mutation] model: {MODEL}")
    print(f"[mt-mutation] wrote: {output_path}")
    print(f"[mt-mutation] generated_docs: {count_text}")
    print(
        "[mt-mutation] tokens: "
        f"input={usage.get('input_tokens', 'unknown')} "
        f"output={usage.get('output_tokens', 'unknown')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
