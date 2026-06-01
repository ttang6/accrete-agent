# nanoagent

> 超轻量、带**反馈学习**能力的 agent 框架 —— LLM 全权决策的 tool-use 主循环 + 从失败 trace 自动学习的 lesson 飞轮，并附一个完整的 AI 日报 Telegram Bot 作为 reference implementation。

## 差异化

- **反馈学习**：agent 跑失败 → trace 自动抽成 lesson → 状态机筛选 → 下次同类失败时召回注入。整条飞轮无 LLM consolidator、不依赖 DSPy/GEPA，纯规则驱动。
- **完整产品形态**：不止有 loop，还有常驻 Telegram Bot + 主动日报 + HITL 反馈的 reference impl（`skills/ai-digest`）。
- **效果经过实测**：50 fixture × 144 trial 跨 model ablation：

  | 指标 | baseline | evolved |
  |---|---|---|
  | lesson 召回使用率 | 0 | **0.64** |
  | helped 次数 | 0 | **58** |
  | 弱-中档 model 上岗率 | — | **单调 ~20×** |

## 快速开始

需要 [uv](https://github.com/astral-sh/uv) 和 Python ≥ 3.11。

```bash
uv venv && uv pip install -e .          # 装框架
cp .env.example .env                     # 填 OPENAI_API_KEY 等（见下）
uv run python main.py                     # 启动 CLI REPL
```

`.env` 至少需要一个 OpenAI 兼容 provider 的 key。主 LLM 走 `OPENAI_API_KEY` + `OPENAI_BASE_URL`（默认模型见 `main.py` 顶部常量）；副 LLM（日报评审 / 偏好蒸馏）可选 `DASHSCOPE_API_KEY`，缺失则自动降级跳过。

跑 ai-digest skill 或 Telegram bot 时按需装 extra：

```bash
uv pip install -e ".[digest]"            # ai-digest skill 依赖
uv pip install -e ".[telegram]"          # Telegram channel 依赖
uv run python run_bot.py                  # 启动 Telegram bot（需 TELEGRAM_BOT_TOKEN）
```

## 项目结构

```
src/nanoagent/
├ core/        LLM client / message / provider / logger / paths / trace schema
├ tool/        BaseTool + registry；fetch / search / skill_exec / arxiv / calc ...
├ runtime/     主循环（native function-calling，非 ReAct）/ harness / session
│              + 协议级自主性（coverage / failure-memory / evaluator）
│              + 隔离子 agent（SubAgent：受限工具 + context 隔离）+ telegram channel
├ skills/      渐进披露的 skill loader（启动只扫 metadata，body 按需 lazy load）
├ memory/      UserFacts 用户画像
├ evolution/   反馈学习飞轮（trace → lesson → 状态机 → 召回）+ 滚动偏好蒸馏 pipeline
└ lesson/      lesson 运维 CLI（手动 promote / retire / list）

main.py         CLI 薄壳（装配 + REPL / one-shot）
run_bot.py      Telegram channel 薄壳
skills/ai-digest/   reference skill：AI 日报抓取 / 去重 / 评审
```

## 反馈学习飞轮

核心思路：把每次失败留下的 trace 当作可复用的经验来源，自动沉淀成下次能召回的 lesson。

```
跑任务 → 失败写进 trace
      → EpisodeExtractor 抽出结构化 episode
      → LessonGenerator 生成 candidate lesson（含结构化修复示例）
      → PromotionGate 状态机：candidate → probation → promoted（或 retired）
      → 下次同类失败时 LessonRetriever 召回，把修复 hint 注入上下文
      → OutcomeTracker 回写 helped / hurt / ineffective，闭环
```

lesson 全程持久化在 SQLite，进程重启不丢；阈值可由 `NANOAGENT_PROMOTION_*` 环境变量覆盖。

## 测试

```bash
uv run python -m pytest -q
```
