"""Command-line entry point: python main.py repair evals/<case> --provider scripted."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from bootstrap import PROVIDERS, build_service

PREVIEW_LINES = 12


def _indent(text: str, max_lines: int = PREVIEW_LINES, tail: bool = False) -> str:
    """Indent text for the terminal, keeping only the first or last lines."""
    lines = text.rstrip().splitlines() or ["(empty)"]
    if len(lines) > max_lines:
        hidden = len(lines) - max_lines
        lines = (
            [f"... ({hidden} earlier lines)", *lines[-max_lines:]]
            if tail
            else [*lines[:max_lines], f"... ({hidden} more lines)"]
        )
    return "\n".join(f"      {line}" for line in lines)


def format_event(event: str, payload: dict[str, Any]) -> str:
    """Turn one public trace event into readable terminal text."""
    step = payload.get("decision")
    if event == "repair_started":
        files = ", ".join(payload.get("files", []))
        return f"[start] project={payload.get('case')} budget={payload.get('max_decisions')} files: {files}"
    if event == "model_response":
        if not payload.get("valid"):
            return f"[{step}] model -> INVALID reply: {payload.get('error')}"
        target = f" {payload['path']}" if payload.get("path") else ""
        summary = f"  ({payload['summary']})" if payload.get("summary") else ""
        return f"[{step}] model -> {payload.get('tool')}{target}{summary}"
    if event == "tool_result":
        tool, status = payload.get("tool"), payload.get("status")
        observation = str(payload.get("observation", ""))
        if status != "ok":
            return f"[{step}] {tool} {status.upper()}:\n{_indent(observation)}"
        if "passed" in payload:
            verdict = "PASSED" if payload["passed"] else "FAILED"
            header = f"[{step}] {tool} {verdict} (exit {payload.get('returncode')})"
            return f"{header}\n{_indent(observation, tail=True)}"
        return f"[{step}] {tool} ok:\n{_indent(observation)}"
    if event == "repair_finished":
        verdict = "REPAIRED" if payload.get("passed") else "NOT REPAIRED"
        return f"[done] {verdict}: {payload.get('reason')} (decisions used: {payload.get('decisions_used')})"
    return f"[{event}] {payload}"


class PrintRecorder:
    """TraceRecorder that prints each event as soon as it happens."""

    def record(self, event: str, payload: dict[str, Any]) -> None:
        print(format_event(event, payload), flush=True)


def run_repair(project: str, provider: str) -> int:
    """Run one repair, print the trace and summary, and return the exit status."""
    try:
        service = build_service(provider)
        result = service.repair(project, recorder=PrintRecorder())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print()
    print(f"Workspace: {result.workspace}")
    print("Changed files:")
    for path in result.changed_files or ["(none)"]:
        print(f"  - {path}")
    print("Final sandboxed test result:")
    if result.latest_test is None:
        print("  (no sandboxed test was run)")
    else:
        test = result.latest_test
        print(f"  {'PASSED' if test.passed else 'FAILED'} (exit {test.returncode})")
        print(_indent(test.output, max_lines=5, tail=True))
    return 0 if result.passed else 1


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line interface."""
    parser = argparse.ArgumentParser(description="Teaching-sized Python repair agent.")
    commands = parser.add_subparsers(dest="command", required=True)
    repair = commands.add_parser("repair", help="Repair a copy of one supplied project under evals/.")
    repair.add_argument("project", help="A supplied project, for example evals/arithmetic.")
    repair.add_argument("--provider", choices=PROVIDERS, default="scripted", help="Chat client to use.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch the selected command."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    if args.command == "repair":
        return run_repair(args.project, args.provider)
    return 2


if __name__ == "__main__":
    sys.exit(main())
