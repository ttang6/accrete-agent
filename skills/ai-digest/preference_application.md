## ai-digest 偏好应用规则

如果上面的偏好提到 **消融实验 / ablation / 模块拆解 / 组件级对比 / ablation table** 类信号：

- **search/filter 阶段**：`fetch_hf` 的 `query` 必须包含至少一个相关关键词（如 "ablation"、"module"、"component"、"evaluation"、"benchmark"）；rerank paper 候选时优先含此类信号的论文。
- **ranking 阶段**：在多篇论文质量相近时，**含 ablation / component analysis 的优先级更高**。
- **summary 阶段**：在描述每篇 paper 时，如果 paper 含 ablation 信息，**必须显式说明**（例如 "该 paper 提供完整 ablation 表 / 模块拆解 / 组件级对比"）。

如果偏好提到 **benchmark / leaderboard / eval script / eval harness** 类信号：

- `fetch_hf` query 优先包含 benchmark / eval 关键词；`fetch_github` 优先保留含 evaluation harness / docker / release 的仓库。
- 描述 paper 时显式提及是否报告 benchmark 数字 / 是否附 eval code。

如果偏好提到 **statistical significance / sample size / std / CI / multi-seed** 类信号：

- 描述 paper 时优先标注样本量 / 置信区间 / 是否多次随机种子。
- 缺失这些信息的 paper 在 ranking 时降级（除非别处压倒性优秀）。

如果偏好提到 **可复现 / reproducibility / docker / GitHub release** 类信号：

- `fetch_github` 优先保留含 release / docker image / eval script 的仓库。
- final summary 中提及对应工具链证据。

如果偏好提到 **某个具体 topic**（如 RAG、agent eval、robotics、multimodal 等）：

- `fetch_hf` query 必须包含该 topic 关键词。
- final summary 中相关条目的 ranking 优先级提升。

## 兜底规则

- 偏好与当前用户请求冲突时，**以用户请求为准**，偏好让位。
- 当前批次没有匹配偏好的候选时，正常输出（不得编造或硬塞 fake evidence）。
- ai-digest 的硬约束（论文 ≤ 5 / 开源 ≤ 5 / 行业 ≤ 3 / 不写"为什么重要"判断层）不得被偏好覆盖；偏好只在这些上限内做选择和排序。
