"""Gradio presentation layer: pick a supplied project, run the existing service, show the result.

The page never builds prompts, calls Docker, or makes repair decisions; it only
calls RepairService and displays the RepairResult it returns.
"""

from __future__ import annotations

import difflib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # No telemetry calls.

import gradio as gr  # noqa: E402

from core.models import RepairResult  # noqa: E402
from core.repair_service import RepairService  # noqa: E402

CASES = ("arithmetic", "balanced_brackets", "median", "multi_function", "unique_words")
OBSERVATION_LINES = 12


def format_trace_entry(entry: dict[str, Any]) -> str:
    """Render one public trace event as readable text."""
    event, step = entry.get("event"), entry.get("decision")
    if event == "repair_started":
        return f"[start] project={entry.get('case')} budget={entry.get('max_decisions')}"
    if event == "model_response":
        if not entry.get("valid"):
            return f"[{step}] model -> INVALID reply: {entry.get('error')}"
        path = f" {entry['path']}" if entry.get("path") else ""
        summary = f"  ({entry['summary']})" if entry.get("summary") else ""
        return f"[{step}] model -> {entry.get('tool')}{path}{summary}"
    if event == "tool_result":
        if "passed" in entry:
            header = f"[{step}] {entry.get('tool')} {'PASSED' if entry['passed'] else 'FAILED'} (exit {entry.get('returncode')})"
        else:
            header = f"[{step}] {entry.get('tool')} {str(entry.get('status')).upper()}"
        return f"{header}\n{_indent(str(entry.get('observation', '')))}"
    if event == "repair_finished":
        verdict = "REPAIRED" if entry.get("passed") else "NOT REPAIRED"
        return f"[done] {verdict}: {entry.get('reason')}"
    return f"[{event}]"


def _indent(text: str) -> str:
    """Indent an observation, keeping its last lines where pytest puts the summary."""
    lines = text.rstrip().splitlines() or ["(empty)"]
    if len(lines) > OBSERVATION_LINES:
        lines = [f"... ({len(lines) - OBSERVATION_LINES} earlier lines)", *lines[-OBSERVATION_LINES:]]
    return "\n".join(f"    {line}" for line in lines)


def _inside(root: Path, relative: str) -> Path:
    """Resolve a relative path and require it to stay inside root."""
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes its folder: {relative}")
    return target


def describe_changes(result: RepairResult, original_dir: Path) -> str:
    """Markdown showing each changed file as a diff against the untouched evals/ original."""
    if not result.changed_files or result.workspace is None:
        return "_No files were changed._"
    sections = []
    for relative in result.changed_files:
        new_text = _inside(result.workspace, relative).read_text(encoding="utf-8")
        original = _inside(original_dir, relative)
        old_text = original.read_text(encoding="utf-8") if original.is_file() else ""
        diff = "\n".join(
            difflib.unified_diff(
                old_text.splitlines(),
                new_text.splitlines(),
                fromfile=f"evals/{original_dir.name}/{relative}",
                tofile=f"workspace/{relative}",
                lineterm="",
            )
        )
        sections.append(
            f"### `{relative}`\n\n```diff\n{diff or '(content identical)'}\n```\n\n"
            f"Repaired file:\n\n```python\n{new_text.rstrip()}\n```"
        )
    return "\n\n".join(sections)


def format_test_output(result: RepairResult) -> str:
    """Text of the most recent sandboxed test run."""
    if result.latest_test is None:
        return "(no sandboxed test was run)"
    test = result.latest_test
    return f"{'PASSED' if test.passed else 'FAILED'} (exit {test.returncode})\n\n{test.output.rstrip()}"


def zip_workspace(workspace: Path) -> str:
    """Archive the disposable workspace into a new folder under the system temp directory."""
    out_dir = Path(tempfile.mkdtemp(prefix="coding-agent-lab-zip-"))
    return shutil.make_archive(str(out_dir / Path(workspace).name), "zip", root_dir=workspace)


def create_demo(service: RepairService | None = None) -> gr.Blocks:
    """Build the page; without a service, use the scripted provider from bootstrap."""
    if service is None:
        from bootstrap import build_service  # Wiring stays in the assembly point.

        service = build_service("scripted")
    missing = set(CASES) - set(service.available_cases())
    if missing:
        raise RuntimeError(f"Supplied projects not found under evals/: {', '.join(sorted(missing))}")

    def on_repair(case: str) -> tuple[str, str, str, str, str | None]:
        """Run the existing service for one dropdown choice and format its result."""
        if case not in CASES:
            raise gr.Error("Choose one of the supplied projects.")
        try:
            result = service.repair(case)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        final = result.trace[-1] if result.trace else {}
        icon = "✅ **Repaired**" if result.passed else "❌ **Not repaired**"
        status = f"{icon}: {final.get('reason', '')}  \nWorkspace: `{result.workspace}`"
        trace = "\n".join(format_trace_entry(entry) for entry in result.trace)
        changes = describe_changes(result, service.resolve_case(case))
        archive = zip_workspace(result.workspace) if result.workspace else None
        return status, trace, changes, format_test_output(result), archive

    with gr.Blocks(title="Python Repair Agent", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Python Repair Agent\n"
            "Choose one supplied project. The agent repairs a disposable copy and "
            "verifies it with sandboxed tests; `evals/` is never modified."
        )
        with gr.Row():
            case = gr.Dropdown(
                choices=list(CASES), value=CASES[0], label="Project", allow_custom_value=False
            )
            start = gr.Button("Start repair", variant="primary")
        status = gr.Markdown()
        trace = gr.Textbox(label="Visible trace", lines=20, max_lines=40, interactive=False)
        changes = gr.Markdown(label="Changed files")
        test_output = gr.Textbox(label="Final sandboxed test output", lines=8, interactive=False)
        download = gr.File(label="Repaired workspace (ZIP)", interactive=False)
        start.click(
            on_repair,
            inputs=case,
            outputs=[status, trace, changes, test_output, download],
            api_name="repair",
        )
    return demo
