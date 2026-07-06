"""阶段五 · 用户轴方法论沉淀产出端。

把够格的失败-修复经验沉淀回 skill：读 backend 里**已 PROMOTED** 的 lesson（飞轮认定
"这条修复真实有效且可复用"的最高信号），用副 LLM 蒸馏成一段自然语言方法论，写进该
skill 目录下一个固定文件名的旁挂文件——**绝不碰 SKILL.md 本体**（本体永远是用户的，
机器写的在旁边）。

本轮只做**产出端落盘**（docs/v3 阶段五）：
- 不注入、不消费——写出来的 methodology.md 本轮不进 SKILL 渲染 / prompt（消费端留后）。
- 不进 user-turn 热路径——经 debug 通道手动触发（`python -m
  nanoagent.evolution.runtime_memory.methodology_distiller`），攒够才落。

"够格"判定复用 lesson_score 的结论（lesson 已 active,即 score≥T）；写文件那步保持哑：
只把够格 lesson 蒸出来复写整文件，不再自己判阈值，除了"每 skill 至少 N 条 active
才值得落"这一个落盘下限（min_lessons）。

带 1.2 那把刀喂副 LLM：只留"换个场景 / 换个用户也成立"的通用做法，丢弃一次性细节与
个人口味——产出是方法论，不是把 lesson 原文搬家。

评估留接口（本轮不做）：LLM-as-a-judge / 小规模盲评（产出的方法论是否真比无更好）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from nanoagent.core.logger import get_logger
from nanoagent.core.prompt_assets import load_prompt
from nanoagent.evolution.runtime_memory.backend import MemoryBackend
from nanoagent.evolution.runtime_memory.lesson_score import is_active
from nanoagent.evolution.runtime_memory.schema import RuntimeLesson
from nanoagent.runtime.subagent import SubAgentContextBuilder, SubAgentRunner

_logger = get_logger("methodology_distiller")

# skill 目录下的旁挂文件名。固定名（不每次新建）+ 整文件复写。绝不是 SKILL.md。
DEFAULT_FILENAME = "methodology.md"

# 每个 skill 至少攒够 N 条 promoted lesson 才值得蒸一次（落盘下限，非晋升阈值）。
DEFAULT_MIN_LESSONS = 3

_DISTILL_SYSTEM_PROMPT = """你是一名把"踩坑修复经验"提炼成**可复用方法论**的资深工程师。

下面给你一个 skill 在实战中反复验证过的若干条失败-修复经验（每条含：什么情况触发、
当时该怎么做、可选的结构化示例）。请把它们提炼成一份简洁的 Markdown 方法论清单。

铁律（提炼，不是搬运）：
- 只写"换个场景 / 换个具体参数也成立"的通用做法——丢弃一次性的具体值、个人口味、
  与本次任务强绑定的细节。
- 同类经验合并成一条，不要逐条复述原文。
- 每条方法论是一句可操作的指导（"遇到 X 先做 Y，因为 Z"），不写空话套话。
- 若这些经验凑不出像样的通用方法论，宁可少写也不硬凑。

输出：直接给 Markdown 正文（用 `-` 列表），不要任何前言/解释/代码围栏。"""


@dataclass
class DistillOutcome:
    """单个 skill 一次蒸馏的结果摘要（run() 返回 list，供 debug 打印 / 单测断言）。"""
    skill: str
    lessons_used: int
    written_path: Optional[str] = None
    skipped_reason: str = ""  # below_min / no_skill_dir / distill_failed / empty_body


def _skill_of_lesson(lesson: RuntimeLesson) -> Optional[str]:
    """把一条 lesson 归属到某个 skill；归不到（通用工具如 fetch/glob 的 lesson）→ None。

    优先读 trigger.scope 的正式键 `skill:<name>`；否则从 op 的
    `skill_exec:<skill>/<script>` 前缀解析（与 op 同源）。
    """
    scope = (lesson.trigger.scope or "").strip()
    if scope.startswith("skill:"):
        name = scope[len("skill:"):].strip()
        if name:
            return name
    op = (lesson.trigger.op or "").strip()
    if op.startswith("skill_exec:") and "/" in op:
        # skill_exec:<skill>/<script>
        skill = op[len("skill_exec:"):].split("/", 1)[0].strip()
        if skill:
            return skill
    return None


class MethodologyDistiller:
    """读 PROMOTED lesson → 副 LLM 蒸方法论 → 复写 skill 旁挂文件。产出端 only。"""

    def __init__(
        self,
        backend: MemoryBackend,
        runner: SubAgentRunner,
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        min_lessons: int = DEFAULT_MIN_LESSONS,
        skills_dir: Path = Path("skills"),
        filename: str = DEFAULT_FILENAME,
    ):
        if filename.lower() == "skill.md":
            raise ValueError("旁挂方法论文件名不得是 SKILL.md（本体永远是用户的）")
        self._backend = backend
        self._runner = runner
        self._model = model
        self._provider = provider
        self._min_lessons = min_lessons
        self._skills_dir = Path(skills_dir)
        self._filename = filename

    # ------------------------------------------------------------------

    def collect_promoted_by_skill(self) -> Dict[str, List[RuntimeLesson]]:
        """扫 backend 全部 active（score≥T）lesson，按归属 skill 分组（归不到 skill 的丢弃）。"""
        out: Dict[str, List[RuntimeLesson]] = {}
        lessons = [
            l for l in self._backend.search_lessons(limit=1000) if is_active(l)
        ]
        for lesson in lessons:
            skill = _skill_of_lesson(lesson)
            if skill is None:
                continue
            out.setdefault(skill, []).append(lesson)
        return out

    def distill_skill(self, skill: str, lessons: List[RuntimeLesson]) -> Optional[str]:
        """把一个 skill 的 promoted lessons 蒸成 NL 方法论正文。fail-open → None。"""
        blocks = [("已验证的失败-修复经验", _render_lessons(lessons))]
        request = SubAgentContextBuilder.build(
            system_prompt=load_prompt("methodology_distiller", _DISTILL_SYSTEM_PROMPT),
            blocks=blocks,
            allowed_tools=(),          # 纯蒸馏，零工具
            model=self._model,
            provider=self._provider,
            max_iterations=1,
        )
        result = self._runner.run(request)
        if not result.ok or not (result.text or "").strip():
            _logger.warning(f"[methodology] distill 失败 skill={skill}: {result.error}")
            return None
        return result.text.strip()

    def write_methodology(self, skill: str, body: str) -> Optional[Path]:
        """复写 `skills/<skill>/<filename>`（整文件）。skill 目录不存在 → 跳过返 None。

        文件头标机器生成 + 指回 SKILL.md 本体；写的是旁挂文件，从不触碰 SKILL.md。
        """
        skill_dir = self._skills_dir / skill
        if not skill_dir.is_dir():
            _logger.warning(f"[methodology] skill 目录不存在，跳过写入: {skill_dir}")
            return None
        path = skill_dir / self._filename
        header = (
            f"<!-- 机器生成（MethodologyDistiller），整文件复写、勿手改。"
            f"skill 本体在 SKILL.md；本文件是失败-修复经验沉淀的旁挂方法论。"
            f"generated_at: {datetime.now().isoformat(timespec='seconds')} -->\n\n"
            f"# {skill} · 沉淀方法论\n\n"
        )
        try:
            path.write_text(header + body.rstrip() + "\n", encoding="utf-8")
        except OSError as e:
            _logger.warning(f"[methodology] 写入失败 {path}: {e}")
            return None
        return path

    def run(self, *, min_lessons: Optional[int] = None) -> List[DistillOutcome]:
        """对每个攒够 min_lessons 条 promoted lesson 的 skill 蒸 + 写。返回逐 skill 摘要。"""
        floor = self._min_lessons if min_lessons is None else min_lessons
        outcomes: List[DistillOutcome] = []
        for skill, lessons in self.collect_promoted_by_skill().items():
            if len(lessons) < floor:
                outcomes.append(DistillOutcome(skill, len(lessons), skipped_reason="below_min"))
                continue
            body = self.distill_skill(skill, lessons)
            if not body:
                outcomes.append(DistillOutcome(skill, len(lessons), skipped_reason="distill_failed"))
                continue
            path = self.write_methodology(skill, body)
            if path is None:
                outcomes.append(DistillOutcome(skill, len(lessons), skipped_reason="no_skill_dir"))
                continue
            outcomes.append(DistillOutcome(skill, len(lessons), written_path=str(path)))
        return outcomes


def _render_lessons(lessons: List[RuntimeLesson]) -> str:
    """promoted lessons → 喂副 LLM 的纯文本块（advice + 触发条件 + 可选结构化示例）。"""
    lines: List[str] = []
    for i, lesson in enumerate(lessons, 1):
        t = lesson.trigger
        cond = f"failure_class={t.failure_class}"
        if t.failure_reason:
            cond += f", cause={t.failure_reason}"
        lines.append(f"{i}. 【触发：{cond}】{lesson.advice}")
        if isinstance(lesson.example, dict) and lesson.example:
            lines.append(f"   结构化示例：{json.dumps(lesson.example, ensure_ascii=False)}")
    return "\n".join(lines)


# ============================================================
# Debug 通道（手动触发，不进 user-turn 热路径）
# ============================================================

def main() -> int:
    import atexit
    import os
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from nanoagent.evolution.runtime_memory.sqlite_backend import (
        DEFAULT_DB_PATH,
        SqliteMemoryBackend,
    )
    from nanoagent.runtime.subagent import build_subagent_llm

    min_lessons = int(os.getenv("NANOAGENT_METHODOLOGY_MIN_LESSONS", str(DEFAULT_MIN_LESSONS)))
    backend = SqliteMemoryBackend(db_path=DEFAULT_DB_PATH)
    atexit.register(backend.close)
    distiller = MethodologyDistiller(
        backend,
        SubAgentRunner(llm_builder=build_subagent_llm),
        min_lessons=min_lessons,
    )
    outcomes = distiller.run()
    if not outcomes:
        print("[methodology] 无 PROMOTED lesson 可归属到任何 skill，未产出。")
        return 0
    for o in outcomes:
        if o.written_path:
            print(f"[methodology] {o.skill}: {o.lessons_used} 条 → 写入 {o.written_path}")
        else:
            print(f"[methodology] {o.skill}: {o.lessons_used} 条 → 跳过（{o.skipped_reason}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
