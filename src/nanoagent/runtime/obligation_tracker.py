"""ObligationTracker —— RequiredActionGate 的 per-turn 状态机。

A 问题（LLM 没调 mark）的根因不是 SKILL.md 描述不够强，而是缺一层
middleware：把"必须调用某工具"从 prompt 软指令下沉到协议级硬控制。

与 RepairGate 互补：
  RepairGate            : 你调了工具，调用必须合法（pre-tool-call）
  RequiredActionGate    : 你该调工具，必须调过（pre-finalization）

每 user-turn 实例化一份（同 FailureMemory 模式）。生命周期：
  1. Harness 收到 user_text → 让 SkillContract 匹配 lexical_hints
     → 命中的 ActionContract 注册为 open obligation
  2. MainLoop 每次 tool_call_success → tracker.notice_tool_call 检查 satisfied
  3. MainLoop 在 "LLM 直接答" 分支调 tracker.has_unsatisfied()：
     - 全 satisfied → 允许 finish
     - 有 unsatisfied + 未 repair 过 → inject repair message，强制再一轮
     - 有 unsatisfied + 已 repair 过 → emit violation event，允许 finish 不死循环

不在范围（暂未实现）：
- 跨 turn obligation（用户上轮说"记下来"，下轮才调 mark）—— 复杂语义留待后续
- LLM-driven user_intent 分类 —— 当前只 lexical hints
- on_missing.mode != repair_once 的策略（block / hard_fail）—— 同上
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from nanoagent.skills.contract import ActionContract

_logger = logging.getLogger("nanoagent.obligation_tracker")


@dataclass
class ObligationState:
    """单条 obligation 的运行期状态。"""
    contract: ActionContract
    status: str = "open"  # open / satisfied
    repair_emitted: bool = False  # repair_once 模式下记是否已 inject 过

    @property
    def id(self) -> str:
        return self.contract.id

    def mark_satisfied(self) -> None:
        self.status = "satisfied"


@dataclass
class ObligationTracker:
    """per-turn obligation 跟踪器。

    一个 turn 内多次 tool call → 多次 notice_tool_call 累积满足情况。
    LLM 想 finish 时调 has_unsatisfied / inject_repair_messages 协调下一步。
    """

    obligations: List[ObligationState] = field(default_factory=list)

    def register(self, matched_contracts: List[ActionContract]) -> None:
        """从 SkillContract.matches_action_triggers 拿到的命中清单注册。

        重复 register 同 contract.id 视为幂等（不重复添加）—— 防 Harness 误调多次。
        """
        existing_ids = {o.id for o in self.obligations}
        for c in matched_contracts:
            if c.id in existing_ids:
                continue
            self.obligations.append(ObligationState(contract=c))

    def notice_tool_call(
        self, tool: str, kwargs: Dict, output_is_failure: bool
    ) -> None:
        """每次 tool_call 处理完后调一次。匹配的 obligation 标记 satisfied。"""
        for obl in self.obligations:
            if obl.status == "satisfied":
                continue
            if obl.contract.obligation.matches_call(tool, kwargs, output_is_failure):
                obl.mark_satisfied()
                _logger.info(
                    f"[ObligationTracker] obligation {obl.id} satisfied by "
                    f"tool={tool} args={kwargs.get('args', {})}"
                )

    def unsatisfied(self) -> List[ObligationState]:
        """返回当前未满足且**未 repair 过**的 obligations。

        已 repair 过的不再返回 —— 一次 repair 后还不满足就允许 finish + 记 violation。
        """
        return [o for o in self.obligations if o.status == "open" and not o.repair_emitted]

    def has_unsatisfied(self) -> bool:
        return bool(self.unsatisfied())

    def all_unsatisfied_including_repaired(self) -> List[ObligationState]:
        """供 emit violation event 用 —— 含 repair 过仍不满足的。"""
        return [o for o in self.obligations if o.status == "open"]

    def build_repair_message(self) -> Optional[str]:
        """组装一条人类可读的 repair message 给 LLM。

        多 obligation 未满足时合并成一条：每条 contract.on_missing_message
        作为独立 bullet。返回后调 mark_repair_emitted 防重复 inject。
        """
        unsatisfied = self.unsatisfied()
        if not unsatisfied:
            return None
        bullets = []
        for obl in unsatisfied:
            msg = (obl.contract.on_missing_message or "").strip()
            if not msg:
                msg = (
                    f"obligation {obl.id} 未满足：要求调 tool={obl.contract.obligation.tool}"
                    f" / args 含 {obl.contract.obligation.args_match}"
                )
            bullets.append(f"- ({obl.id}) {msg}")
        return (
            "[harness-required-action] 你即将完成本轮但未满足以下声明的 action contract：\n"
            + "\n".join(bullets)
            + "\n请调用对应工具完成；如果你判断不应调（例如本轮没产生需 mark 的内容），"
            "请在最终回答里**明确说明跳过原因**，再 finish。"
        )

    def mark_repair_emitted(self) -> None:
        """调 build_repair_message 之后必调一次，防 main_loop 同 turn 内反复 inject。"""
        for obl in self.obligations:
            if obl.status == "open":
                obl.repair_emitted = True
