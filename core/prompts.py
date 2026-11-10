"""All text the repair agent sends to a model. Nothing here parses or runs a reply."""

from __future__ import annotations

from collections.abc import Sequence

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


def system_message() -> dict[str, str]:
    """Return the system instruction that defines the workflow and JSON actions."""
    return {"role": "system", "content": SYSTEM_INSTRUCTION}


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
