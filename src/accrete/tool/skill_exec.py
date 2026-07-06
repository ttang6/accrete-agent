"""SkillExecTool — 执行 skill 自带的 Python 脚本。

设计哲学：
  SkillLoader 只管 metadata + body 呈现，不管执行（对齐 HelloAgents 生产版）。
  真正执行 skill 的 scripts/ 脚本由本 tool 负责，权限收敛到当前激活 skill 的 scripts/
  目录下，防跨 skill 误调、防路径穿越、防任意文件执行。

LLM 调用契约：
  skill_exec(skill="ai-digest", script="fetch_rss", args={"max_results": 15})
    → 执行 `python skills/ai-digest/scripts/fetch_rss.py`
    → JSON-dumped args 通过 stdin 传给脚本
    → stdout 作为 tool_result 返回（超过 max_output_chars 截断）
    → 非 0 退出码 → 附 stderr 片段

传参走 stdin JSON 而非命令行：避免 shlex 注入风险，嵌套参数天然支持。
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from accrete.tool.base import BaseTool, ValidationOutcome
from accrete.tool._skill_arg_validator import (
    load_skill_script_schema,
    validate_skill_args,
)


class SkillExecTool(BaseTool):
    """执行当前 skill 的 scripts/ 目录下指定 python 脚本。"""

    def __init__(
        self,
        skills_dir: Path,
        timeout_seconds: int = 60,
        max_output_chars: int = 5000,
    ):
        self._skills_dir = Path(skills_dir).resolve()
        self._timeout = timeout_seconds
        self._max_output = max_output_chars

    # 尝试开 OpenAI strict 但**实测不兼容**。OpenAI API 直接 400 拒绝：
    #   Invalid schema for function 'skill_exec': In context=('properties', 'args'),
    #   'additionalProperties' is required to be supplied and to be false.
    #
    # 根因：args 内层本质是 free-form object（fetch_* 无字段 / dup_check 有 action /
    # verify_secret 有 magic_phrase / ...）——OpenAI strict 要求**每个**嵌套 object
    # 都有 additionalProperties:false，跟 free-form dict 物理不兼容。无 escape hatch。
    #
    # 长期解是 **virtual script tools**：把每个 script 暴露成独立 tool（schema 完全
    # 静态可控），runtime 内部转回 skill_exec。
    #
    # BaseTool.strict_mode 基础设施仍保留——其他 schema 完全静态的 tool（如 calc）
    # 可独立开启受益。
    strict_mode = False

    @property
    def name(self) -> str:
        return "skill_exec"

    def call_key(self, kwargs: dict) -> str:
        """skill_exec 的 op 投影 = skill/script:args-hash（工具自声明）。

        熔断/计数粒度到"具体脚本 + 具体 args"：同一 turn 内反复以同一组 args 调同
        一脚本（典型的 identical_retry 死循环）才累加；换 args（如 dup_check 从
        action=check 改 mark）算新 op。lesson 召回仍按 skill/script 粗匹配（见
        failure_memory._operation_key 的 recall_key），两者粒度有意不同。
        """
        from accrete.tool.base import _args_hash
        d = kwargs or {}
        skill = str(d.get("skill", "") or "?").strip() or "?"
        script = str(d.get("script", "") or "?").strip() or "?"
        inner = d.get("args")
        h = _args_hash(inner if isinstance(inner, dict) else {})
        return f"skill_exec:{skill}/{script}:{h}"

    def op(self, kwargs: dict) -> str:
        """lesson 粗键 = skill/script（不含 args-hash）——同一脚本的各种失败归一条 lesson。
        与 call_key 同前缀、去掉 args-hash 尾巴。"""
        d = kwargs or {}
        skill = str(d.get("skill", "") or "?").strip() or "?"
        script = str(d.get("script", "") or "?").strip() or "?"
        return f"skill_exec:{skill}/{script}"

    @property
    def description(self) -> str:
        return (
            "执行指定 skill 自带的 Python 脚本。"
            "参数通过 stdin 以 JSON 传给脚本，脚本的 stdout 作为结果返回。"
            "只能执行 `skills/<skill>/scripts/<script>.py`，不允许跨 skill 或路径穿越。"
            "若不确定 script 的参数 shape（尤其是 args 必填或分派型参数的 script），"
            "先调 `describe_script` 拿结构化 schema 再调本工具。"
        )

    @property
    def parameters(self) -> dict:
        # args 包装的 schema 硬化（防 LLM 漏 args wrapper / flat hoisting）：
        # - `args` 加入 required —— 防止 LLM 漏 args wrapper 直接发
        #   `skill_exec({"skill":..., "script":...})`（function calling dispatch
        #   层会拒绝 schema 不合的请求）
        # - `additionalProperties: false` —— 拒绝 LLM 把 inner 字段（如
        #   `magic_phrase`）冒到 skill_exec 顶层；强制只有"args 包装"一种正确形态
        # - 无参脚本（fetch_*）调用方需显式传 `args: {}`，
        #   对模型更稳定：永远记住 script inputs live under args
        return {
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "skill 名（通常就是当前激活的 skill）",
                },
                "script": {
                    "type": "string",
                    "description": "scripts/ 目录下的脚本名，不含 .py 后缀",
                },
                "args": {
                    "type": "object",
                    "description": (
                        "传给脚本的参数对象（必填），会以 JSON 通过 stdin 传入。"
                        "无参脚本传 `{}`；有参脚本必须把所有字段塞这里——"
                        "**不要把字段冒到 skill_exec 顶层**。具体字段调 "
                        "describe_script 看 inner schema。"
                    ),
                },
            },
            "required": ["skill", "script", "args"],
            "additionalProperties": False,
        }

    def validate(self, **kwargs) -> Optional[str]:
        if not (kwargs.get("skill") or "").strip():
            return "skill 参数不能为空"
        if not (kwargs.get("script") or "").strip():
            return "script 参数不能为空"
        return None

    def check_args(self, **kwargs) -> Optional[ValidationOutcome]:
        """ToolCallRepairGate：subprocess 前对 args 做 jsonschema 校验（从 _execute 搬来）。

        fail-open：schema 不存在 / args 非 dict → None（照常走 _execute → subprocess）。
        校验失败 → ValidationOutcome，run() 直接把 repair_text（以 [gate-invalid-call]
        开头）当 tool_result 返回：MainLoop 检测前缀 emit ACTION_TOOL_CALL_REPAIR_REQUIRED，
        FailureMemory 把它当 failure 走 lesson 召回 / 重复失败保护链。飞轮契约零改动。
        """
        skill = (kwargs.get("skill") or "").strip()
        script = (kwargs.get("script") or "").strip()
        args = kwargs.get("args") or {}
        if not skill or not script or not isinstance(args, dict):
            return None
        schema = load_skill_script_schema(self._skills_dir, skill, script)
        if schema is None:
            return None
        return validate_skill_args(schema, skill, script, args)

    def _execute(self, **kwargs) -> str:
        skill = kwargs["skill"].strip()
        script = kwargs["script"].strip()
        args = kwargs.get("args") or {}
        if not isinstance(args, dict):
            return f"[skill_exec 错误] args 必须是 JSON 对象，收到 {type(args).__name__}"

        script_path = self._resolve_script_path(skill, script)
        if script_path is None:
            return (
                f"[skill_exec 错误] 非法路径或脚本不存在: "
                f"skills/{skill}/scripts/{script}.py"
            )

        # arg 校验（RepairGate）已上移到 check_args，在 run() 里 _execute 之前跑。

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                input=json.dumps(args, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=self._timeout,
                encoding="utf-8",
            )
        except subprocess.TimeoutExpired:
            return f"[skill_exec 错误] 脚本执行超时（{self._timeout}s）"
        except Exception as e:
            return f"[skill_exec 错误] 启动子进程失败: {e}"

        if result.returncode != 0:
            stderr = (result.stderr or "")[:500] or "(no stderr)"
            return f"[skill_exec 错误] 退出码 {result.returncode}\nstderr: {stderr}"

        output = result.stdout or ""
        if len(output) > self._max_output:
            output = (
                output[: self._max_output]
                + f"\n...(stdout 共 {len(output)} 字，已截断)"
            )
        return output

    def _resolve_script_path(self, skill: str, script: str) -> Optional[Path]:
        """安全解析脚本路径：必须落在 skills/<skill>/scripts/ 实体目录下。

        校验步骤：
          1. 拼出目标路径并 resolve（展开软链 / 解相对）
          2. 确认 resolve 后的路径仍在 self._skills_dir/<skill>/scripts/ 根下
          3. 确认是文件（不是目录 / 软链到目录）且以 .py 结尾
        """
        expected_root = (self._skills_dir / skill / "scripts").resolve()
        candidate = (expected_root / f"{script}.py").resolve()
        try:
            candidate.relative_to(expected_root)
        except ValueError:
            return None
        if not candidate.is_file() or candidate.suffix != ".py":
            return None
        return candidate
