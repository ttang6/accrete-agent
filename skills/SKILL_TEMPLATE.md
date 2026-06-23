# SKILL_TEMPLATE.md — 写新 Skill 前必读

> 参考 Anthropic 官方 Skills 文档 `https://code.claude.com/docs/zh-CN/skills` 写就，防止把 skill 滑坡成 agent 定义。
> 新建 skill 时复制这个模板到 `skills/<your-skill-name>/SKILL.md` 再改。

---

## 一、什么是 Skill

**一句话**：Skill 是**按需加载**的能力扩展，假设 agent 已存在，只告诉它"怎么做这一件事"。

### Skill ≠ 完整 prompt

❌ **错（写成 agent 定义）**：
```yaml
---
name: code-reviewer
description: ...
---

# Role
你是一位严谨的高级代码审查员...

# Goal
发现代码质量问题...

# 输出格式
...
```

✅ **对（扩展 agent 能力）**：
```yaml
---
name: code-review
description: 审查代码质量的清单和流程
---

审查代码时，逐项检查以下五类问题：
1. 类型安全
2. 错误处理
3. 并发陷阱
4. 性能瓶颈
5. 可读性

每项问题标注严重程度（critical / major / minor）。
```

区别：**后者假设 agent 已有基础能力**，只补充"审查该做什么"。

### Skill 的两类内容（抄官方说法）

1. **参考内容**（reference）：约定 / 模式 / 风格指南 / 领域知识。内联使用。
2. **任务内容**（task）：特定操作的分步说明（deploy / commit）。通常加 `disable-model-invocation: true` 由用户 `/skill-name` 显式触发。

---

## 二、Frontmatter 字段参考

**基于官方文档**（`https://code.claude.com/docs/zh-CN/skills`）的字段。只列 nanoagent 会用到的。

```yaml
---
name: my-skill
description: 一句话描述这个 skill 做什么、什么时候用
disable-model-invocation: false   # true = 只允许用户 / 代码显式触发（默认 false = 两者都可）
allowed-tools:                    # 激活此 skill 时能用的工具白名单（空 = 不限）
  - tool_name_a
  - tool_name_b
---
```

### 字段使用建议

| 字段 | 何时用 |
|---|---|
| `name` | 必填。小写 + 连字符，和目录名一致 |
| `description` | 强烈建议。LLM 用它决定是否加载，前 250 字符被截断，**关键用例前置** |
| `disable-model-invocation: true` | 有副作用的任务（deploy / digest / onboarding）或希望用户显式触发的 |
| `allowed-tools` | 明确指定本 skill 用到的工具。起到"工具隔离硬约束"作用（对比只靠 prompt 说"请用 X"）|

---

## 三、Body 写作反模式对照表

| 反模式（❌） | 正模式（✅） | 理由 |
|---|---|---|
| `# Role` / `你是一个 X agent` | 直接写流程 / 规则 | 身份由默认 system prompt 承担 |
| `# Goal` / `你的任务是...` | 直接描述步骤 | skill 是扩展，不是 agent 定义 |
| 重复默认 prompt 已有的规范（"用中文回答"） | 只写 skill 特有的 | 避免 token 浪费 + 不一致风险 |
| 长篇大论（>500 行） | 关键规则精炼，详细资料放支持文件 | 官方建议 SKILL.md < 500 行 |
| 工具列表用散文写在 body 里 | 用 `allowed-tools` 字段声明 | 结构化字段更易维护 |
| 身份型命名（`researcher` / `editor`） | 任务型命名（`code-review` / `ai-digest`） | 按做什么不按是谁 |

---

## 四、两个参考示例

### 示例 A：任务型 skill（带流程 + 输出规范）

```yaml
---
name: deploy-staging
description: 把当前代码部署到 staging 环境，跑烟雾测试
disable-model-invocation: true
allowed-tools:
  - Bash
---

# 流程

1. 确认当前分支是 `main` 且 working tree 干净
2. 跑 `pnpm test` 确认测试全绿
3. 跑 `pnpm build` 产出 bundle
4. 用 `scripts/deploy-staging.sh` 推到 staging
5. 等 30 秒后 `curl https://staging.example.com/health` 验证

# 异常处理

- 任一步失败立即停止，不要继续后续步骤
- 输出简洁的失败原因给用户，不要自己尝试修
```

### 示例 B：参考型 skill（风格约定）

```yaml
---
name: commit-message-style
description: 本仓库的 commit message 规范（Conventional Commits 变体）
---

# 格式

`<type>: <subject>`

subject 用祈使句，小写开头，不加句号，≤60 字符。

# type 列表

| type | 含义 |
|---|---|
| feat | 新功能 |
| fix | bug 修复 |
| refactor | 不改行为的结构调整 |
| docs | 仅文档 |
| test | 仅测试 |
| chore | 构建 / 工具 / 配置 |

# 示例

- ✅ `feat: add user profile onboarding skill`
- ✅ `fix: dedup parser drops trailing period on URL`
- ❌ `Added onboarding.` （没 type / 过去式 / 句号）
```

---

## 五、提交前 checklist

在把新 skill 加进版本控制前，逐项过一遍：

- [ ] `name` 字段是任务型（动词/操作），不是角色型（研究员/编辑）
- [ ] `description` 第一句 250 字内能让 LLM 判断"什么时候用"
- [ ] body 里**没有** `# Role` / `# Goal` / `你是一个 X` 这类身份定义
- [ ] body 里**没有**重复默认 system prompt 已有的通用规范
- [ ] 如果用到工具，写进 `allowed-tools` 而非散文描述
- [ ] 如果是副作用操作（部署 / 数据写入），加 `disable-model-invocation: true`
- [ ] body 总长 < 500 行；超了用支持文件（`skills/<name>/reference.md` 等）
- [ ] 本地 smoke test：`SkillsLoader.load_all()` 能成功解析

---

## 六、本项目已有 skills

| skill | 类型 | 触发 | 说明 |
|---|---|---|---|
| `ai-digest` | 任务 | `/ai` 用户命令 | 生成今日 AI 日报 |
| `onboarding` | 任务 | 系统自动路由（`USER_PROFILE` 为空时） | 首次用户冷启动访谈（Inversion 模式）|

### 未来扩展候选（**暂不做**，列在这是防滑坡提醒）

- ❌ `/query` skill —— 普通问答走默认 prompt 就够了
- ❌ `researcher` / `writer` / `editor` 类角色型 skill
- ❌ "100+ skill 大全"

新增 skill 前问自己：**"不独立就会污染现有 prompt 吗？"** 是 → 独立；否则 → 塞进相关 skill 的 body。
