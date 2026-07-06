"""ToolOutputStore——超限 tool output 落盘 + head_tail preview 注入 prompt。

替代 MainLoop 内联的 `result[:5000] + "...(截断)"`（main_loop.py:326-327）。
两层动机：

1. **prompt 端**：把无界 tool 输出限制到 max_single_tokens（默认 2000），用
   head_tail preview 保留头尾——错误信息常在尾部，head-only 截断诊断不友好
2. **复盘端**：原文落盘到 `data/tool_outputs/<date>/<trace_id>/<tool_call_id>.txt`，
   EpisodeExtractor 复盘 trace 时能拿到完整版本——和飞轮直接相关

入门视角：
- tool 输出 ≤ max_single_tokens（且本 run 总量未超）→ 不落盘，inline 原文返回
- 超 max_single_tokens 或 force_store=True → 落盘 + 返回 head_tail preview + path
- preview 格式见 _build_preview docstring；token 数估算用注入的 TokenCounter
- per call 一次性写文件，无常驻句柄——不需 atexit close

不做的事：
- 不做按行 / 字节截断（tiktoken token 数才是真实约束，行/字节是 HelloAgents
  ObservationTruncator 的次优近似）
- 不做 metadata JSON wrapper（原文就是原文，元信息进 trace 不需 JSON 包装）
- 不做自动清理（按 date 子目录，未来加 cron 即可，本期不做）
- 不做并发写防护（tool_call_id 由 LLM 生成，trace 内唯一；rerun 同 trace_id
  同 call_id 时覆写——这是合理行为不是 bug）

下游：
- MainLoop._run_inner 替换 `[:5000]` 为 `store_if_needed`
- 调用方拿到 `StoredOutput` 后 emit ACTION_TOOL_OUTPUT_SAVED / TRUNCATED trace
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from accrete.runtime.token_counter import TokenCounter

_logger = logging.getLogger("accrete.tool_output_store")


# ============================================================
# 返回结果
# ============================================================


@dataclass(frozen=True)
class StoredOutput:
    """store_if_needed 的返回值。caller 用 `preview` 喂进 messages。"""

    truncated: bool
    """是否触发了截断（无论 single_limit 还是 total_limit）。"""

    preview: str
    """要喂给 LLM 的内容。truncated=False 时即原文；True 时是 head_tail 摘要 +
    "完整原文见 <path>" reference 句。"""

    full_path: Optional[Path]
    """落盘文件绝对路径（truncated=False 时为 None）。Path 而非 str 便于 caller
    直接 read_text 验证。"""

    original_chars: int
    """原始输出字符数（不是 token 数；落盘成本估算靠这个）。"""

    original_tokens: int
    """原始输出 token 数（TokenCounter 估）。"""

    preview_tokens: int
    """preview 的 token 数。truncated=False 时等于 original_tokens。"""

    reason: str
    """截断原因，三种值之一：
      - "" (空字符串): 未截断
      - "single_limit": 单 tool output 超 max_single_tokens
      - "total_limit": 累计 tool tokens 已超 ContextBudget 阈值（force_store=True）
    """


# ============================================================
# Store
# ============================================================


class ToolOutputStore:
    """tool output 落盘 + preview 构造。

    生命周期：在 main.py 装配区构造一份，注入 MainLoop。无可变状态（base_dir
    / counter / 阈值都是 immutable），可被多 MainLoop 实例共享。

    base_dir 不要求预先 mkdir——每次写文件时按 trace 创建子目录。这样
    `data/tool_outputs/` 即使外层不存在也能跑（参考 paths.py::data_dir 模式）。
    """

    def __init__(
        self,
        base_dir: Path,
        token_counter: TokenCounter,
        max_single_tokens: int = 2_000,
        head_ratio: float = 0.7,
    ):
        """
        Args:
            base_dir: 落盘根目录。约定 `data/tool_outputs/`，main.py 走 data_dir()
                解析。即使不存在，写文件时会自动 mkdir。
            token_counter: 共享的 TokenCounter。preview 长度估算靠它。
            max_single_tokens: 单 tool output 的 token 上限。超过即落盘。默认
                2000 ≈ 8000 chars，比当前 max_tool_output_chars=5000 放宽 60%。
            head_ratio: head_tail preview 的 head 占比。默认 0.7——前 70% +
                后 30%，错误信息常在尾部所以 tail 留 30% 比 HelloAgents
                ObservationTruncator 的 head-only 友好。范围 [0.0, 1.0]，
                极端值（0.0=纯 tail / 1.0=纯 head）也支持。
        """
        if not 0.0 <= head_ratio <= 1.0:
            raise ValueError(f"head_ratio must be in [0.0, 1.0], got {head_ratio}")
        if max_single_tokens <= 0:
            raise ValueError(f"max_single_tokens must be > 0, got {max_single_tokens}")
        self._base_dir = Path(base_dir)
        self._counter = token_counter
        self._max_single = max_single_tokens
        self._head_ratio = head_ratio

    def store_if_needed(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        trace_id: str,
        output: str,
        force_store: bool = False,
    ) -> StoredOutput:
        """主入口：根据阈值 + force_store 决定是否落盘。

        Args:
            tool_name: 用于 trace 字段（落盘文件名不带，因 tool_call_id 已唯一）
            tool_call_id: LLM 生成的 call id，文件名用它
            trace_id: 来自 RunTracer._trace_path.stem（如 'assistant_153012'），
                作为子目录名隔离不同 run
            output: 原始 tool 输出文本
            force_store: 即使 token 数 ≤ max_single_tokens 也强制落盘——
                ContextBudget 累计超总额度时调用方传 True

        Returns: StoredOutput（含 preview / path / token 数 / reason）
        """
        original_chars = len(output) if output else 0
        original_tokens = self._counter.count_text(output)

        # 决定是否落盘
        if force_store:
            reason = "total_limit"
            should_store = True
        elif original_tokens > self._max_single:
            reason = "single_limit"
            should_store = True
        else:
            reason = ""
            should_store = False

        if not should_store:
            return StoredOutput(
                truncated=False,
                preview=output,
                full_path=None,
                original_chars=original_chars,
                original_tokens=original_tokens,
                preview_tokens=original_tokens,
                reason="",
            )

        # 落盘（可能失败 → full_path=None）
        full_path = self._write_full_output(
            trace_id=trace_id, tool_call_id=tool_call_id, output=output
        )
        preview = self._build_preview(
            output=output,
            original_tokens=original_tokens,
            full_path=full_path,
            reason=reason,
        )
        preview_tokens = self._counter.count_text(preview)

        return StoredOutput(
            truncated=True,
            preview=preview,
            full_path=full_path,  # None 时 caller 应跳过 ACTION_TOOL_OUTPUT_SAVED
            original_chars=original_chars,
            original_tokens=original_tokens,
            preview_tokens=preview_tokens,
            reason=reason,
        )

    # ============================================================
    # 内部
    # ============================================================

    def _write_full_output(
        self, *, trace_id: str, tool_call_id: str, output: str
    ) -> Optional[Path]:
        """落盘到 base_dir/<date>/<trace_id>/<tool_call_id>.txt。

        date 从 trace_id 推不出（trace 文件名是 'assistant_HHMMSS' 格式），
        所以用当前日期。同 trace 跨 day 的极端情况（凌晨 0 点跨过）会写到不同
        日期目录——这是边缘 case，不修。

        Returns:
            写盘成功 → Path；写盘失败（OSError / mkdir 异常）→ None。
            caller 据此决定是否 emit ACTION_TOOL_OUTPUT_SAVED + preview 文案。
            返回 None 时 _logger.warning 已记录，调用方不再重复 log。
        """
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        # 清洁 tool_call_id 防路径穿越（LLM 不可控值，保险）
        safe_call_id = self._sanitize_filename(tool_call_id) or "unknown_call"
        target_dir = self._base_dir / date_str / trace_id
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / f"{safe_call_id}.txt"
            target_file.write_text(output, encoding="utf-8")
        except OSError as e:
            _logger.warning(
                f"tool output 落盘失败 base_dir={self._base_dir} "
                f"trace={trace_id} call={safe_call_id}: {e}"
            )
            return None
        return target_file

    def _build_preview(
        self,
        *,
        output: str,
        original_tokens: int,
        full_path: Optional[Path],
        reason: str,
    ) -> str:
        """构造 head_tail preview + reference 句。

        策略：
        1. 头尾按 head_ratio 切原文字符数（不是 token 数——按 token 二分查找
           overengineered）
        2. 中间插入 "...(中间省略 N tokens)..." 提示
        3. 末尾追加完整原文路径 + 落盘原因

        极端情况：
        - output 极短但 force_store=True（total_limit 触发）→ 头尾切完接近
          原文，preview 几乎不省，仍写出 reference 句
        - head_ratio=1.0 → 退化成 head-only + reference
        - head_ratio=0.0 → 退化成 tail-only + reference
        - full_path=None（落盘失败）→ reference 句改成"落盘失败仅保留预览"
          避免误导 LLM / 后续 EpisodeExtractor 去找不存在的文件
        """
        # 按 max_single_tokens 估算 preview 字符预算
        # 目标：preview tokens ≤ max_single_tokens（留 200 token buffer 给 reference）
        budget_tokens = max(200, self._max_single - 200)
        # token → char 粗换：按原文密度估算
        if original_tokens > 0:
            chars_per_token = max(1, len(output) // original_tokens)
        else:
            chars_per_token = 4
        budget_chars = budget_tokens * chars_per_token

        if budget_chars >= len(output):
            # budget 比原文还大（罕见，主要是 force_store + 短 output）
            head_text = output
            tail_text = ""
            omitted_tokens = 0
        else:
            head_chars = int(budget_chars * self._head_ratio)
            tail_chars = budget_chars - head_chars
            head_text = output[:head_chars] if head_chars > 0 else ""
            tail_text = output[-tail_chars:] if tail_chars > 0 else ""
            # omitted = 原文中间被切掉的部分
            omitted_chars = len(output) - head_chars - tail_chars
            omitted_tokens = self._counter.count_text(
                output[head_chars : len(output) - tail_chars]
            ) if omitted_chars > 0 else 0

        # 路径展示——落盘失败时不能给路径（会误导 EpisodeExtractor 去找不存在的文件）
        if full_path is None:
            ref_phrase = f"落盘失败，仅保留预览, reason={reason}"
        else:
            path_display = "/".join(full_path.parts[-3:]) if len(full_path.parts) >= 3 else str(full_path)
            ref_phrase = f"完整原文见 {path_display}, reason={reason}"

        # 组装
        parts: list[str] = []
        if head_text:
            parts.append(head_text)
        if omitted_tokens > 0:
            parts.append(f"\n...(中间省略 ~{omitted_tokens} tokens，{ref_phrase})...")
        else:
            parts.append(f"\n...({ref_phrase})...")
        if tail_text:
            parts.append(tail_text)

        return "".join(parts)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """剥离路径字符 / 非法 Windows 文件名字符。tool_call_id 通常形如
        'call_abc123' 或 'tooluse_XYZ'，正常情况无需处理；防御 LLM 异常输入。"""
        if not name:
            return ""
        bad = '<>:"/\\|?*\0\r\n\t'
        cleaned = "".join("_" if c in bad else c for c in name)
        # 限长（NTFS 单段 ≤ 255）
        return cleaned[:200]
