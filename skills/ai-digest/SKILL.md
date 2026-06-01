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

## 历史去重

`dup_check` — 跨批次日报条目去重，避免同一篇论文 / 博客 / 仓库在多次日报中重复出现。
历史记录存在 `data/memory/digest_reported.jsonl`，跨 channel 共享。

**⚠️ dup_check 与 fetch_* 不同，args 必填**（有两个 action 分派，无合理默认）。**调用前必须先 `describe_script(skill="ai-digest", script="dup_check")` 拿完整 schema**，按返回的结构化示例填 args。

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
   - 论文：按 upvotes 降序挑前 3-5 篇（低热度论文跳过）
   - RSS：按分类挑——`AI Research` / `AI Engineering` / `Open Source` / `Industry News` 各 1-2 条，总计 6-8 条
   - GitHub：挑 AI / LLM / agent 相关 2-3 个，去掉明显无关的（如纯前端框架）
   - 忽略明显招聘 / 活动 / 融资性质的条目（rss 层已过滤一次黑名单，有漏网再判）
4. **补详情（可选）**：
   - 论文摘要太短或元数据缺失 → `arxiv(action="get_abstract", paper_id=X)` 补完整 abstract / 作者 / 分类
   - 非 arxiv 条目摘要太短（< 100 字）或关键信息缺失 → `fetch` 工具打开 URL 读全文
   - 用户明确要求查某篇论文的相关工作时 → `arxiv(action="citation_graph", paper_id=X)`
5. **按输出规范写**
6. **历史去重（mark）**：以下任一信号都触发：
   - **正常路径**：输出日报后立即调，不等用户确认
   - **用户显式确认**：用户说"记下来 / 标记 / 保存 / 记一下今天报过的"等表达
   - 调用：`skill_exec(skill="ai-digest", script="dup_check", args={"action": "mark", "items": [{"fingerprint": ..., "source": ..., "title": ...}, ...]})` 把本次采纳的所有条目（论文 + RSS + GitHub）写入历史
   - **不要**仅口头说"已记录"而不调工具——harness 已声明 mark obligation，未真调用会被 RequiredActionGate 拦下来
   - **失败降级**：mark 失败不影响已输出的日报，简短标注即可

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
