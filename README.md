# Accrete Agent

`accrete` 是一个基于 native function calling 的 agent 系统，关注任务的运行可靠性：长任务或自主态下，多数失败不再以报错的形式出现，也难以靠重试等常规手段覆盖。框架为此在 harness 层内建确定性防护与经验学习机制，把失败运行沉淀为后续可复用的经验，并配套结构化 trace 与确定性评测，验证这些机制的实际效果。

## 特性

`accrete` 当前聚焦单 agent 主循环和运行可靠性，不是多 agent 编排框架。核心能力包括：

- **Native function-calling agent loop**：基于模型原生 tool call 的循环，由模型持续决策下一步动作，runtime 负责执行、观察、上下文管理和收束。
- **统一工具抽象**：tool、skill、MCP、SubAgent 都作为可声明、可调用、可观测的 action 接入主循环。
- **运行期可靠性机制**：按失败能否被确定性修复分层——可修复的归 harness（失败分类、参数修复、同参熔断），修不了的沉淀为经验记忆（lesson），相似场景召回、按后续运行效果校准；skill 侧从任务轨迹提炼可复用方法论，回写进既有 skill 文档。
- **用户记忆**：长期对话中的身份、偏好和目标压缩成可持久化记忆，跨 session 作为软指导注入。
- **轻量本地持久化**：session、用户记忆和 lesson 主要落在本地文件 / SQLite，默认不引入向量数据库。
- **消息网关解耦**：CLI、one-shot、Telegram 等 channel 只负责接入和路由，主循环、状态和记忆留在 Harness / runtime 内部。
- **可观测性**：建立 log、metrics、trace / trajectory 组件，用于复盘运行过程、把失败归因到具体步骤，并支撑后续 evaluation。

## Agent Loop

`accrete` 的主循环围绕“决策—执行—观察—留痕”运转：模型基于当前 session、可用工具和运行期反馈持续选择下一步动作；runtime 负责把动作落到 tool、skill、MCP 或 SubAgent 上，并把过程记录成 trace。

```text
user task
  -> MainLoop
  -> tool / skill / MCP / SubAgent
  -> observation
  -> failure memory / critic / context budget
  -> answer
  -> trace
```

trace 是复盘、评测和经验学习的共同输入。失败相关轨迹会被压缩成 lesson，在相似场景中召回，并通过后续运行结果继续校准。

```text
failed trace
  -> episode
  -> candidate lesson
  -> retrieval
  -> outcome tracking
```

lesson 是一条结构化的经验记忆：触发条件锚定在可观测的失败特征上，有效性由后续运行结果回写；长期帮不上忙的记忆会被降权、退出注入。

## 参考实现：AI 日报推送

`skills/ai-digest` 是基于 `accrete` 构建的参考实现：它会持续抓取 AI 领域候选信息，做去重、筛选、评审，并通过 Telegram channel 发布。

这个场景足够简单，也足够接近真实长期任务：有稳定输入、重复内容、来源污染、用户偏好和实际发布动作，并且可以日常运行。它会持续产生可观察的 trace，用来检验框架的可靠性、学习和记忆机制是否能在真实循环中成立。

## 工程化验证

### 消融实验

**问题**：lesson 飞轮能不能把一次失败转化为后续任务里的行为改善？

5 个业务域各埋一条“事先不可知的隐藏规则”——不踩一次坑就做不对，重试和查文档都救不了；学习任务与测试任务使用不同对象、不同错误表述，考察的是规则迁移而不是复现样本。三种条件对照：**OFF**（不召回经验）、**ON**（从学习任务自动沉淀 lesson、测试时召回）、**ORACLE**（人工写的 gold lesson，作为经验质量上限）。失败全部由确定性注入产生，零真实网络；每个测试任务重复运行 3 次，报单次成功率和 pass^3 稳定性。

| 模型 | OFF | ON (pass^1) | ON (pass^3) | ORACLE |
|---|---:|---:|---:|---:|
| gpt-5.4-mini | 0.00 | 0.80 | 0.67 | 0.98 |
| qwen3.6-plus | 0.00 | **1.00** | **1.00** | 1.00 |
| claude-sonnet-4-6 | 0.00 | 0.80 | 0.80 | 0.85 |

| 模型 | OFF 平均步数 | ON 平均步数 | OFF 平均 token | ON 平均 token |
|---|---:|---:|---:|---:|
| gpt-5.4-mini | 17.3 | 17.0 | 11,015 | 10,238 |
| qwen3.6-plus | 25.2 | 16.8 | 19,363 | 11,167 |
| claude-sonnet-4-6 | 15.1 | 15.5 | 16,200 | 15,850 |

> 观察到 OFF 和 ON 在一些模型上的平均步数 / token 差距不大；但关键差别在于 OFF 为失败空转成本，ON 能把任务完成。

> 受控合成实验：证明 lesson 飞轮能学习并迁移“隐藏参数 / 本地规则”这一类失败，不等于能解决所有真实世界失败。

### 多轮状态评测

**问题**：多轮用户交互中，agent 能不能把任务结束后的真实系统状态改对，而不是只生成看起来合理的回复？

基于 AI 日报场景的 6 类多轮任务（发布前修订、来源核验、历史去重、重复发布幂等、故障恢复、中断后恢复），每类 2 个实例、各重复运行 3 次。user simulator 现场驱动多轮对话；判分不读回复文本，只用确定性规则检查真实副作用（历史账本增量、归档文件、被剔除条目），不用模型当裁判。内容源来自预采集真实数据的本地回放，每次运行使用独立临时环境，零真实网络、零生产数据污染。

| 模型 | 任务成功率 | pass^3 通过 | 结束状态正确率 | 平均轮数 | 平均 token |
|---|---:|---:|---:|---:|---:|
| qwen3.6-plus | 36/36 = 100% | 12/12 | 36/36 = 100% | 4.33 | 101,981 |
| gpt-5.4-mini | 31/36 = 86.1% | 8/12 | 32/36 = 88.9% | 4.00 | 84,486 |
| claude-sonnet-4-6 | 35/36 = 97.2% | 11/12 | 35/36 = 97.2% | 4.19 | 128,581 |

> 本地模拟评测：借鉴 τ-bench 的多轮 user simulator 与最终状态判分思路，但不和 τ-bench 官方结果做横向比较，也不代表开放世界长期可靠性。

## 快速开始

需要 [uv](https://github.com/astral-sh/uv) 和 Python >= 3.11。

```bash
uv venv
uv pip install -e .
cp .env.example .env
uv run python main.py
```

`.env` 至少需要一个 OpenAI 兼容 provider 的 key。主 LLM 使用 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`；默认模型在 `main.py` 顶部常量里配置。副 LLM 用于日报评审和全局记忆蒸馏，可选 `DASHSCOPE_API_KEY`，缺失时自动跳过对应能力。

运行 AI 日报 skill 或 Telegram bot：

```bash
uv pip install -e ".[digest]"
uv pip install -e ".[telegram]"
uv run python run_bot.py
```

Telegram bot 需要 `TELEGRAM_BOT_TOKEN`；可选 `TELEGRAM_CHAT_ID` 做白名单，`TELEGRAM_DIGEST_AT=HH:MM` 开启进程内每日定时推送。

如需接入通用 MCP server auto-expand：

```bash
uv pip install -e ".[mcp]"
```

## 项目结构

```text
src/accrete/
├ core/        LLM client / provider / message / logger / trace schema
├ tool/        BaseTool + registry；fetch / search / grep / glob / skill_exec / mcp_tool
├ runtime/     MainLoop / Harness / session / context budget / critic / telegram channel
├ skills/      skill loader / metadata lazy loading
├ memory/      UserFacts / global memory distillation
├ evolution/   trace -> episode -> lesson -> retrieval -> outcome tracking
└ lesson/      lesson 运维 CLI

main.py             CLI REPL / one-shot 入口
run_bot.py          Telegram bot 入口
skills/ai-digest/   AI 日报 reference skill
```
