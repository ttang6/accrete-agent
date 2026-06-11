# nanoagent

> 超轻量 agent 框架 —— LLM 全权决策的 tool-use 主循环 + 两层反馈学习（失败 → lesson、对话 → 偏好），并附一个完整的 AI 日报 Telegram Bot 作为 reference implementation。

## 特点

- **两层反馈学习**：系统层从失败 trace 沉淀 lesson、同类失败时召回；用户层从对话蒸馏 skill 偏好、作软指导注入。规则驱动，无额外 LLM consolidator。
- **层次化失败处理**：工具失败先由确定性机制处理（熔断、退避、失败计数防循环）；修复经验事后从运行记录里提炼，下次同类失败时作为提示召回。LLM 反思只在很窄的条件下参与，且它的建议要经实际运行验证才算数。
- **闭环应用体验**：Telegram Bot + 主动日报 + HITL 反馈，日报能力封装为SKill： `skills/ai-digest`。
- **简易Evaluation**：`evals/` 用跨 model ablation 量反馈学习前后的差异。

## 快速开始

需要 [uv](https://github.com/astral-sh/uv) 和 Python ≥ 3.11。

```bash
uv venv && uv pip install -e .          # 装框架
cp .env.example .env                     # 填 OPENAI_API_KEY 等（见下）
uv run python main.py                     # 启动 CLI REPL
```

`.env` 至少需要一个 OpenAI 兼容 provider 的 key。主 LLM 走 `OPENAI_API_KEY` + `OPENAI_BASE_URL`（默认模型见 `main.py` 顶部常量）；副 LLM（日报评审 / 偏好蒸馏 / 在线微反思）可选 `DASHSCOPE_API_KEY`，缺失则自动降级跳过。provider 支持 openai / dashscope / dashscope_us / anthropic / zhipu / deepseek / ollama / vllm 等多端点（偏好蒸馏可走中立第三方）。

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
├ tool/        BaseTool + registry
├ runtime/     主循环 + telegram channel
├ skills/      渐进披露的 skill loader
├ memory/      UserFacts 用户画像
├ evolution/   反馈学习飞轮（trace → lesson → 状态机 → 召回）+ 滚动偏好蒸馏 pipeline
│              + 层次化处理（online_reflector 在线反思 / offline_mint + reflector 离线教训）
└ lesson/      lesson 运维 CLI（手动 promote / retire / list）

main.py             CLI 薄壳（装配 + REPL / one-shot）
run_bot.py          Telegram channel 薄壳
run_offline_mint.py 离线 mint 批跑入口（从已观测恢复产教训）
skills/ai-digest/   reference skill：AI 日报抓取 / 去重 / 评审
```

## 反馈学习

两个轴：从**失败**学（系统层）+ 从**对话**学用户偏好（用户层）。

**系统层** —— 把每次失败留下的 trace 当作可复用经验，自动沉淀成下次能召回的 lesson：

```
跑任务 → 失败写进 trace
      → EpisodeExtractor 抽出结构化 episode（失败后的成功恢复自动配对为修复证据）
      → LessonGenerator 生成 candidate lesson（按失败原因聚合，含结构化修复示例）
      → PromotionGate 状态机：candidate → probation → promoted（或 retired）
      → 下次同类失败时 LessonRetriever 召回，把修复 hint 注入上下文
      → OutcomeTracker 回写 helped / hurt / ineffective，闭环
```

lesson 全程持久化在 SQLite，进程重启不丢；阈值可由 `NANOAGENT_PROMOTION_*` 环境变量覆盖。

**用户层** —— 一个隔离的副 agent 把近期对话蒸成 skill 级 NL 偏好 summary，经语义闸门 + 存储硬校验后落盘，下次进该 skill 时作为软指导注入；在 turn / skill / session 边界按节奏触发，保守写入。

## 本地评估

评估基于确定性故障注入，主模型采用 Qwen3.6-Plus，Reflector 和 Distiller 采用 Qwen3.6-Flash，用户层 LLM judge 采用 Claude Sonnet 4.6。共150 次试验：50 个任务 × 关学习 / 开学习 / 人工写好教训三部分。数字仅供参考：

| 指标          | 关     | 开           |
| ----------- | ----- | ----------- |
| 工具调用成功率     | 71%   | 81%         |
| 单任务平均步数     | 30.5  | 24.0（−21%）  |
| 单任务平均 token | 30.5k | 22.4k（−27%） |
| 单任务平均耗时     | 21.2s | 13.9s（−34%） |

用户层偏好做过消融对照：偏好摘要 + 应用契约逐层加上后，LLM 盲评的贴合偏好评分由 1/5 升至 4/5，且随注入深度单调。
