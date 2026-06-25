# nanoagent

`nanoagent` 是一个基于 native function calling 的 agent 系统。在工具调用之外，它在运行期持续沉淀可复用状态：从失败轨迹中提炼修复经验（lesson），从长期对话中蒸馏用户记忆，并预留 skill methodology refinement，用于基于运行反馈优化已有 skill 的使用方式。

## 特性

`nanoagent` 当前聚焦单 agent 主循环和运行期自我改进，不是多 agent 编排框架。核心能力包括：

- **双层运行期学习**：系统层把失败运行记录转成可检索 lesson，在相似错误复现时召回，用运行结果继续校准其有效性；skill 侧预留 methodology refinement，用于基于运行反馈自主调整已有 skill 的使用方式。用户层把长期对话中的身份、偏好和目标压缩成可持久化用户记忆，在跨 session 对话中作为软指导注入。
- **Native function-calling agent loop**：基于模型原生 tool call 的循环，由模型持续决策下一步动作，runtime 负责执行、观察、上下文管理和收束。
- **统一工具抽象**：tool、skill、MCP、SubAgent 都作为可声明、可调用、可观测的 action 接入主循环。
- **轻量本地持久化**：session、用户记忆和 lesson 主要落在本地文件 / SQLite，默认不引入向量数据库。
- **消息网关解耦**：CLI、one-shot、Telegram 等 channel 只负责接入和路由，主循环、状态和记忆留在 Harness / runtime 内部。
- **可观测性**：建立 log、metrics、trace / trajectory 组件，用于复盘运行过程、定位失败原因，并支撑后续 evaluation。

## Agent Loop

`nanoagent` 的主循环围绕“决策—执行—观察—留痕”运转：模型基于当前 session、可用工具和运行期反馈持续选择下一步动作；runtime 负责把动作落到 tool、skill、MCP 或 SubAgent 上，并把过程记录成 trace。

```text
user task
  -> MainLoop
  -> tool / skill / MCP / SubAgent
  -> observation
  -> failure memory / critic / context budget
  -> answer
  -> trace
```

trace 不只是日志，也是后续学习和评测的输入。失败相关轨迹会被压缩成 lesson，在相似场景中召回，并通过后续运行结果继续校准。

```text
failed trace
  -> episode
  -> candidate lesson
  -> promote / retire
  -> retrieval
  -> outcome tracking
```

lesson 不是聊天记录里的“反思文本”，而是带触发条件、状态和效果回写的运行期对象。

## 参考实现：AI 日报推送

`skills/ai-digest` 是基于 `nanoagent` 构建的参考实现：它会持续抓取 AI 领域候选信息，做去重、筛选、评审，并通过 Telegram channel 发布。

这个场景足够简单，也足够接近真实长期任务：有稳定输入、重复内容、来源污染、用户偏好和实际发布动作，并且可以日常运行。它会持续产生可观察的 trace，用来检验框架的学习、记忆和可靠性机制是否能在真实循环中成立。

## 工程化验证

### 消融实验

**目的**：验证 lesson 飞轮是否能把一次失败转化为后续任务里的行为改善。

**数据准备**：准备 5 个业务域，每个域各有 3 个学习任务和 3 个测试任务，共 15 个学习场景、15 个测试场景。学习和测试使用不同对象、不同错误表述，但隐藏规则属于同一类问题，避免测试阶段只是复现训练样本。

**实验方法**：每个测试任务重复运行 3 次，同时记录单次成功率和 pass^3 稳定性；对比三种条件：

- **OFF**：不召回经验。
- **ON**：从学习任务自动沉淀的 lesson，再在测试任务召回。
- **ORACLE**：人工写的 gold lesson，作为经验质量上限。

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

**目的**：验证 agent 在多轮用户交互中，能不能稳定地调用工具、遵守流程，并把任务结束后的系统状态改对，而不是只生成看起来合理的回复。

**数据准备**：基于 AI 日报推送场景合成 6 类多轮任务，覆盖发布前修订、来源核验、历史去重、重复发布幂等、故障恢复和中断后恢复。每类任务设计 2 个不同实例；内容源来自预采集真实数据的本地回放。每次测试运行都使用独立临时环境，避免污染真实数据，也便于检查最终状态。

**实验方法**：用 user simulator 驱动被测 agent 多轮对话；每个任务重复运行 3 次，检查最终状态和真实副作用。

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
src/nanoagent/
├ core/        LLM client / provider / message / logger / trace schema
├ tool/        BaseTool + registry；fetch / search / grep / glob / skill_exec / mcp_tool
├ runtime/     MainLoop / Harness / session / context budget / critic / telegram channel
├ skills/      skill loader / metadata lazy loading / coverage contract
├ memory/      UserFacts / global memory distillation
├ evolution/   trace -> episode -> lesson -> promotion -> retrieval -> outcome tracking
└ lesson/      lesson 运维 CLI

main.py             CLI REPL / one-shot 入口
run_bot.py          Telegram bot 入口
skills/ai-digest/   AI 日报 reference skill
evals/              failure-learning 与状态化多轮评测
docs/               设计记录、实验报告和重构笔记
```

## 测试

```bash
uv run python -m pytest -q
```

Codex 环境里如果默认临时目录权限异常，可以显式指定仓库内临时目录：

```bash
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.\.pytest_tmp_codex
```
