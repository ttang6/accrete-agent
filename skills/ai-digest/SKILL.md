---
name: ai-digest
description: 生成今日 AI 领域日报，覆盖论文 / 开源 / 行业动态三个维度。用户请求"今日日报 / 生成日报 / AI 日报"时使用。
scope: ai-digest
allowed-tools:
  - skill_exec
  - describe_script
  - arxiv
  - fetch
---

# 可用资源

## 并行拉候选（三维度）

本 skill 自带 python 脚本，通过 `skill_exec` 工具调用：

- `fetch_hf` — 拉取 Hugging Face 今日精选论文（由 HF 编辑人工筛选，含 upvotes 热度）
- `fetch_rss` — 从 `sources.yaml` 订阅的 RSS 源拉近 48h 文章（覆盖 Anthropic / DeepMind / OpenAI / HF Blog / 多个 AI 研究博客）
- `fetch_github` — 拉取 GitHub Trending 仓库（HTML 抓取 + 按天缓存），可按语言过滤

调用格式：

```
skill_exec(skill="ai-digest", script="fetch_hf", args={"max_results": 10, "sort": "hot"})
skill_exec(skill="ai-digest", script="fetch_rss", args={"category": "AI Research", "max_results": 15})
skill_exec(skill="ai-digest", script="fetch_github", args={"language": "python", "since": "daily", "max_results": 10})
```

RSS 的 `category` 可选值：`AI Research` / `AI Engineering` / `Open Source` / `Industry News`。留空 = 全部。
HF 的 `sort` 可选值：`hot` / `rising` / `new`。留空使用接口默认。
GitHub 的 `since` 可选值：`daily` / `weekly` / `monthly`；`language` 留空 = 所有语言。

## 历史去重（只用 check）

`dup_check` — 选题前查跨批次重复，避免同一篇论文 / 博客 / 仓库在多次日报中重复出现。
历史记录存在 `data/memory/digest_reported.jsonl`，跨 channel 共享。

**dup_check 只有 `check` 一个 action 供你调用**：传 fingerprints 列表查"哪些已报过"。
**不要自己写历史（不存在 `mark` 工具调用）**——发布流程会在你输出日报后，按末尾机器块
自动登记本期采纳条目，你只管选题去重，不管落库。

**⚠️ dup_check 与 fetch_* 不同，args 必填**（无合理默认）。**首次调用前先
`describe_script(skill="ai-digest", script="dup_check")` 拿完整 schema**，按返回的结构化示例填 args。

Fingerprint 规则（三类天然全局唯一，无需前缀）：
- 论文 → `arxiv_id`（如 `2510.12345`）
- RSS 文章 → 完整 `link` URL
- GitHub 仓库 → `owner/repo` 全名

**失败降级（关键）**：dup_check 调用失败时**跳过历史去重继续主流程**，不阻塞日报输出。日报末尾简短标注"本次历史去重未生效（dup_check 调用失败）"即可，不重试超过 1 次。

## 按需深度查询

`arxiv` 工具（跨 skill 通用的 BaseTool，通过 MCP 协议桥接到 arxiv-mcp-server）用于**对已拿到 arxiv_id 的论文做深度补强**。默认日报流程**不用它拉新论文候选**（arxiv 按 date 排序是 preprint 裸流，没质量信号，会稀释 HF Daily 的热门筛选）。

典型触发场景（均需要已知 `paper_id`）：

- **摘要太短 / 元数据不全** → `arxiv(action="get_abstract", paper_id="2510.XXXXX")` 拿完整 abstract + 作者列表 + 分类
- **用户要求看全文** → `arxiv(action="download_paper", paper_id=X)` 下载后 `arxiv(action="read_paper", paper_id=X)` 读全文 markdown
- **想看相关工作** → `arxiv(action="citation_graph", paper_id=X)` 拿引用 / 被引用图

`fetch` 工具（跨 skill 通用）：用于抓取非 arxiv 的 URL 全文（博客、GitHub README 等）。

# 流程

1. **并行拉候选（三维度）**：
   - `fetch_hf(max_results=10)` 拿论文候选
   - `fetch_rss(category="", max_results=20)` 拿 RSS 文章候选
   - `fetch_github(since="daily", max_results=10)` 拿 GitHub Trending 候选
2. **历史去重（check）**：
   - **首次本会话调用 dup_check 前，先 `describe_script(skill="ai-digest", script="dup_check")` 拿 schema**（同一会话内已调过可跳过）
   - 从上一步 raw candidates 抽出 fingerprints（arxiv_id / URL / owner/repo）
   - 调 `skill_exec(skill="ai-digest", script="dup_check", args={"action": "check", "fingerprints": [...]})` 拿到"已报过 / 新鲜"分组
   - 已报过的条目**默认跳过**，除非有重大新进展值得重报（此时在后续输出里显式说明"延续 X 话题新进展"）
   - **失败降级**：调用失败时跳过本步骤直接进入挑选，日报末尾简短标注即可
3. **挑选**（只从新鲜条目里挑）：
   - **覆盖目标（软方法论，不是硬闸）**：尽量三维度都有料——论文 ≥ 3 篇、开源 ≥ 2 条、行业动态 ≥ 1 条。某维度本轮确实拉不到新料（如 RSS 近 48h 无更新）就如实省段，不要为凑数塞旧条目或硬补无关内容。
   - 论文：按 upvotes 降序挑前 3-5 篇（低热度论文跳过）
   - RSS：按分类挑——`AI Research` / `AI Engineering` / `Open Source` / `Industry News` 各 1-2 条，总计 6-8 条
   - GitHub：挑 AI / LLM / agent 相关 2-3 个，去掉明显无关的（如纯前端框架）
   - 忽略明显招聘 / 活动 / 融资性质的条目（rss 层已过滤一次黑名单，有漏网再判）
4. **补详情（可选）**：
   - 论文摘要太短或元数据缺失 → `arxiv(action="get_abstract", paper_id=X)` 补完整 abstract / 作者 / 分类
   - 非 arxiv 条目摘要太短（< 100 字）或关键信息缺失 → `fetch` 工具打开 URL 读全文
   - 用户明确要求查某篇论文的相关工作时 → `arxiv(action="citation_graph", paper_id=X)`
5. **按输出规范写**（散文日报 + 末尾机器块，见下）
6. **不要自己登记历史**：本期采纳条目由发布流程按你输出的机器块自动写入去重历史，
   你不调任何 mark 工具、也不口头说"已记录"。

# 输出规范

```
AI 日报 — YYYY-MM-DD

（可选：主线句，概括今日三大亮点，不超 50 字）

## 论文速览

**[论文原文英文标题]**
arxiv YYYY-MM-DD · upvotes N · [abs](https://arxiv.org/abs/{arxiv_id})
[2-3 句中文事实描述]

**[第二篇...]**
...

---

## 开源与工程

**[项目/文章原名]**
YYYY-MM-DD · [source_name] · [link](URL)
[2-3 句中文事实]

---

## 行业动态

**[原文标题]**
YYYY-MM-DD · [source_name] · [link](URL)
[2-3 句中文事实]
```

某段本轮无新内容时整段省略（不写"无内容"占位）。

## 末尾机器块（必附，发布流程唯一真值）

散文日报给人读；**在日报正文之后，再附一个机器可读的 JSON 块**，列出本期采纳的全部条目。
发布流程**只读这个机器块**来登记去重历史与做出处核对——散文不参与抽取，也不校验两者一致，**以机器块为准**。漏附机器块 = 本期一条都不会被登记。

格式（放在整个回答的最末尾，用 ```json 围栏）：

```json
{"adopted_items": [
  {"fingerprint": "2510.12345", "source": "fetch_hf", "title": "论文英文标题"},
  {"fingerprint": "https://blog.example/post", "source": "fetch_rss", "title": "文章标题"},
  {"fingerprint": "owner/repo", "source": "fetch_github", "title": "仓库名"}
]}
```

机器块规则：
- 一条 = 一个采纳条目；`fingerprint` 必填，用 `## 历史去重` 里的三类格式（arxiv_id / 完整 URL / owner/repo）。
- `source` 填来源工具名（`fetch_hf` / `fetch_rss` / `fetch_github`），`title` 填条目标题。
- **只列散文里真正写进日报的条目**——fingerprint 必须来自本轮 fetch_* 的真实返回，发布流程会做出处核对，凭空编造的条目会被剔除、不登记。
- 本轮一条都没采纳（全部已报过 / 三维度皆空）→ 附 `{"adopted_items": []}`。

# 翻译策略

- **标题保留原文英文**（论文 + 英文博客）
- **事实描述用中文**
- **新方法 / 模型 / benchmark 名保留英文**（如 `Chain-of-Agents` / `PagedAttention` / `Claude Opus 4`）
- **通用技术词用中文**（推理 / 训练 / 数据集 / 对齐）

# Constraints

1. **严禁编造 / 不塞旧**：arxiv_id / URL / 日期必须来自 `fetch_hf` / `fetch_rss` / `fetch_github` / `arxiv` 工具的真实返回；本轮某段无新内容就省段，不得塞历史 entries 或拼凑
2. **零元叙述**：不写"根据我搜索到的信息 / 根据工具返回"之类废话，直接输出日报正文
3. **链接完整**：论文条目用 `https://arxiv.org/abs/{arxiv_id}` 标准格式
4. **不写"为什么重要"判断层**：泛用重要性判断对不认识用户容易沦为模板，留给用户显式请求深入分析时再触发
5. **上限约束**：论文 ≤ 5 篇，开源 + 工程 ≤ 5 条，行业动态 ≤ 3 条。总条目控制在 10-13 条，太多冲淡重点
