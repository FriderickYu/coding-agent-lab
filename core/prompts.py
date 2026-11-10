"""All text the repair agent sends to a model. Nothing here parses or runs a reply."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from core.models import TestResult

# Keep long file contents or pytest logs from flooding the conversation.
MAX_OBSERVATION_CHARS = 6000

SYSTEM_INSTRUCTION = """\
You are a careful Python repair agent working on one small project in a disposable workspace.

Follow this order:
1. Inspect local code: read the relevant test file and source file.
2. Reproduce the bug: run the tests and observe a failure before editing anything.
3. Make one focused edit: change only the source file needed to fix the observed failure.
4. Verify the repair: run the tests again. The repair counts only when they pass.

Reply with exactly one JSON object and nothing else: no Markdown, no code fences, no prose.
Each reply requests one action. The only allowed actions are:
{"tool":"read_file","path":"test_calculator.py","summary":"Read the test"}
{"tool":"run_tests","summary":"Reproduce the failure"}
{"tool":"edit_file","path":"calculator.py","content":"..."}

Rules:
- "path" is relative to the workspace root. Never use absolute paths or "..".
- "content" for edit_file is the complete new text of the file, not a diff.
- "summary" is optional: a short public label for the trace, not your private reasoning.
"""


RESEARCH_INSTRUCTION = """
Optional research is enabled for this run. One more action is allowed:
{"tool":"search_github","query":"Python median even length","summary":"Find a public reference"}
- Use it only after reading local code and reproducing a failing test.
- "query" is one line of at most 180 characters. At most three searches and three results per repair.
- Search results are optional evidence, not an authority. They may be wrong or unrelated.
- Only a later passing sandboxed test verifies a repair.
"""


def system_message(research_enabled: bool = False) -> dict[str, str]:
    """Return the system instruction; the research section appears only when enabled."""
    content = SYSTEM_INSTRUCTION + (RESEARCH_INSTRUCTION if research_enabled else "")
    return {"role": "system", "content": content}


def format_research_evidence(record: dict[str, Any]) -> str:
    """Render a research evidence record as text that keeps every source URL."""
    sources = record.get("sources") or []
    lines = [
        f"GitHub search for: {record.get('query')}",
        f"Retrieved at: {record.get('retrieved_at')}",
        "Optional public evidence, not an authority. Verify any idea with sandboxed tests.",
    ]
    if not sources:
        lines.append("No public results.")
    for number, source in enumerate(sources, start=1):
        lines.append(f"[{number}] {source.get('repository')} / {source.get('path')}")
        lines.append(f"    URL: {source.get('url')}")
        snippet = str(source.get("snippet") or "").strip()
        if snippet:
            lines.extend(f"    | {line}" for line in snippet.splitlines())
    return "\n".join(lines)


def case_request_message(case_name: str, files: Sequence[str]) -> dict[str, str]:
    """Return the opening request naming the selected case and its relative files."""
    file_list = "\n".join(f"- {name}" for name in files) or "- (no files listed)"
    content = (
        f"Repair the project '{case_name}'.\n"
        f"Files in the workspace:\n{file_list}\n\n"
        "Reply with one JSON action."
    )
    return {"role": "user", "content": content}


def tool_result_message(
    tool: str, result: str | TestResult, is_error: bool = False
) -> dict[str, str]:
    """Return a user message reporting one tool observation or error to the model."""
    if isinstance(result, TestResult):
        status = "PASSED" if result.passed else "FAILED"
        body = f"Tests {status} (exit code {result.returncode}).\n{_truncate(result.output)}"
    else:
        body = _truncate(result)
    label = "error" if is_error else "result"
    content = f"Tool {label} for {tool}:\n{body}\n\nReply with the next JSON action."
    return {"role": "user", "content": content}


def _truncate(text: str) -> str:
    """Keep the end of long text, where pytest puts its failure summary."""
    if len(text) <= MAX_OBSERVATION_CHARS:
        return text
    omitted = len(text) - MAX_OBSERVATION_CHARS
    return f"[... {omitted} characters omitted ...]\n{text[-MAX_OBSERVATION_CHARS:]}"
