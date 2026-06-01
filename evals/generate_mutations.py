"""Generate raw mutation fixtures for ai-digest eval cold start.

This is a one-shot helper for docs/_fixture_mutation_plan.md Step 1.
It writes raw LLM output for human review and does not affect run_eval,
because load_task_specs only reads top-level evals/tasks/*.yaml files.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# ============================================================
# 运行参数（改这里就改行为，不需要命令行）
# ============================================================
SEED_TASK_DIR: Path = Path("evals/tasks")
OUTPUT_DIR: Path = Path("evals/tasks/mutations")
TARGET_MIN: int = 20
TARGET_MAX: int = 30


# ============================================================
# LLM 参数
# ============================================================
ANTHROPIC_API_URL: str = "https://api.anthropic.com/v1/messages"
MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")
MAX_TOKENS: int = 12000
TEMPERATURE: float | None = None


def _read_seed_tasks(tasks_dir: Path) -> str:
    parts: list[str] = []
    for path in sorted(tasks_dir.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        parts.append(f"# {path.name}\n{text.strip()}\n")
    if not parts:
        raise FileNotFoundError(f"未找到 seed task YAML: {tasks_dir}")
    return "\n---\n".join(parts)


def _build_prompt(seed_yaml: str) -> str:
    return f"""你在帮一个叫 nanoagent 的 self-improving agent 项目设计 eval fixture。

agent 背景：
- 调一个叫 ai-digest 的 skill，用工具 fetch_hf（HF 论文）/ fetch_rss（多源新闻）/
  fetch_github（trending repos）/ dup_check（去重 / mark）
- ObligationTracker：用户说 "记下来" / "记一下" / "标记" / "保存" / "记录" 等关键词
  会强制 agent 必须调 dup_check action=mark，不调即 obligation violation
- coverage 检查：默认要求 paper≥3, oss≥2, news≥1，不达标会触发 evaluator retry
- 失败分类（trace_error_classifier）：
  - schema_mismatch（args 错 / 缺必填字段）
  - repeated_same_args_failure（同 args 反复失败）
  - coverage_gap（类目数量不达标）
  - soft_quality_issue（evaluator 判低质量）
  - tool_runtime_error（工具运行时异常）

种子 fixtures：
{seed_yaml}

任务：
为每种失败类型生成约 3 个变种 query。每个变种要满足：
1. 自然像真实用户会发的话，不要明显造作，不要明显 adversarial
2. 大概率触发声明的 failure_type
3. 不要简单改写种子（70% 以上重叠会被丢弃），来自不同真实使用场景

输出要求：
- 输出单个 YAML 文档流，用 --- 分隔每条 fixture
- 不要输出 markdown 代码围栏，不要输出解释文字
- 输出 {TARGET_MIN}-{TARGET_MAX} 条
- 每条字段：
  - id: <短 snake_case，必须以 mut_ 开头>
  - description: <10-30 字中文描述>
  - query: <用户实际会输入的话>
  - expected_obligations: [<action_contract id 列表，无则空 []>]
  - expected_tool_calls: [<{{tool, skill, script}} 列表>]
  - expected_coverage: [<paper/oss/news 子集>]
  - max_iterations: <int，难 case 给 30-40>
  - success_keywords: [<final answer 含的关键词>]
  - expected_failure_type: <schema_mismatch/repeated_same_args_failure/coverage_gap/soft_quality_issue/tool_runtime_error/obligation_miss/multi_constraint_conflict>

注意：
- expected_failure_type 是人工筛选用注释字段；正式 fixture 可能会删掉。
- query 要像用户真的会输入，不要写成测试说明。
- 至少覆盖 schema_mismatch / repeated_same_args_failure / coverage_gap / soft_quality_issue / tool_runtime_error。
"""


def _post_anthropic(prompt: str) -> dict[str, Any]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 ANTHROPIC_API_KEY")

    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }
    if TEMPERATURE is not None:
        payload["temperature"] = TEMPERATURE
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=body,
        headers={
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic API HTTP {exc.code}: {detail}") from exc


def _extract_text(response: dict[str, Any]) -> str:
    chunks: list[str] = []
    for block in response.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            chunks.append(str(block.get("text", "")))
    text = "\n".join(chunks).strip()
    if not text:
        raise RuntimeError("Anthropic API 返回中没有 text content")
    return text


def _count_yaml_docs(raw_yaml: str) -> int | None:
    try:
        docs = [doc for doc in yaml.safe_load_all(raw_yaml) if doc is not None]
    except yaml.YAMLError:
        return None
    return len(docs)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    load_dotenv()

    seed_yaml = _read_seed_tasks(SEED_TASK_DIR)
    prompt = _build_prompt(seed_yaml)
    response = _post_anthropic(prompt)
    raw_yaml = _extract_text(response)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    output_path = OUTPUT_DIR / f"raw_{timestamp}.yaml"
    output_path.write_text(raw_yaml + "\n", encoding="utf-8")

    usage = response.get("usage") or {}
    doc_count = _count_yaml_docs(raw_yaml)
    count_text = "unknown" if doc_count is None else str(doc_count)
    print(f"[mutation] wrote: {output_path}")
    print(f"[mutation] generated_docs: {count_text}")
    print(
        "[mutation] tokens: "
        f"input={usage.get('input_tokens', 'unknown')} "
        f"output={usage.get('output_tokens', 'unknown')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
