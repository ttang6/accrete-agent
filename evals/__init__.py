"""nanoagent eval harness（项目级 dev tool，不属于 runtime）。

目的：跑固定任务集，量化"飞轮真的有效（或不有效）"。

baseline vs evolved 对照（advisor 二轮裁定 + 用户 pivot 决定）：
- baseline: NANOAGENT_ENABLE_LESSON_RECALL=0 + NANOAGENT_ENABLE_PROMOTION_GATE=0
            （ingestor 仍跑，让 evolved 有 lesson 可用）
- evolved:  NANOAGENT_ENABLE_LESSON_RECALL=1 + NANOAGENT_ENABLE_PROMOTION_GATE=1
            （用 baseline 跑完留下的 sqlite 起点，模拟"飞轮起作用"）

关键 metric（grader 从 trace JSONL 抽）：
- 任务成功率 / 步数 / token / tool 失败率 / obligation 完成率
- lesson 命中率 / lesson_helped / lesson_hurt / lesson_ineffective（P0.2 ACTION_OUTCOME_UPDATE 信号）
- coverage missing / final answer 长度

不跑真实 LLM 时（mock 模式）只验证 grader / aggregator / diff_report 通路；
真实 baseline vs evolved 对照需要 OPENAI_API_KEY + DASHSCOPE_API_KEY。
"""
