"""协作式打断信号：一次 run 里"用户请求停下"的标志位。

用途：用户在 agent 跑长任务时按 Ctrl-C，希望它停下——但不是硬 kill（会丢 trace、留半
截状态），而是**协作式**：channel 侧（如 CLI 的 Ctrl-C 处理器）把标志置位，主循环在每个
步边界查一次，置位就干净收尾——发一条带步序号的打断事件、快照当前对话、返回半截结果。
"协作式" = 不打断正在进行的当前步，等它跑完、在边界停。

这也让"打断"成为一条**带步级位置的隐式标注**：用户在第几步停下 = 他觉得"问题不晚于此"，
是事后失败归因的免费野生真值（对齐总路线图交互态 regime 的真值来源）。

设计成独立小对象（而非直接用 threading.Event）是为了命名贴合语义、且方便单测——测试
直接构造它、置位、断言主循环行为，不碰信号与线程。
"""
from __future__ import annotations


class RunInterrupt:
    """一次 run 的协作式取消标志。channel 置位、主循环在步边界查。"""

    def __init__(self) -> None:
        self._requested = False

    def request(self) -> None:
        """请求打断（幂等，多次调用等同一次）。"""
        self._requested = True

    @property
    def requested(self) -> bool:
        """当前是否已被请求打断。"""
        return self._requested

    def reset(self) -> None:
        """清除标志。每次 run 开始时调，避免上一轮的残留请求误停这一轮。"""
        self._requested = False
