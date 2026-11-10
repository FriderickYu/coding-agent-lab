"""The bounded repair loop: the model proposes one action, this code decides."""

from __future__ import annotations

import json
import posixpath
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.models import ModelAction, RepairResult, TestResult
from core.ports import ChatClient, RepairToolsPort, ResearchPort, TraceRecorder
from core.prompts import (
    case_request_message,
    format_research_evidence,
    system_message,
    tool_result_message,
)

DEFAULT_MAX_DECISIONS = 8
TRACE_PREVIEW_CHARS = 1500

# tool name -> (required fields, optional fields), not counting "tool" itself.
ACTION_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "read_file": (frozenset({"path"}), frozenset({"summary"})),
    "run_tests": (frozenset(), frozenset({"summary"})),
    "edit_file": (frozenset({"path", "content"}), frozenset({"summary"})),
}
# Accepted by the parser only when a research port was injected (--research).
RESEARCH_ACTION_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "search_github": (frozenset({"query"}), frozenset({"summary"})),
}

EARLY_EDIT_MESSAGE = (
    "Policy error: edit_file is allowed only after a successful read_file and a "
    "failing run_tests in this repair attempt. The edit was not applied."
)
EARLY_SEARCH_MESSAGE = (
    "Policy error: search_github is allowed only after a successful read_file and a "
    "failing run_tests in this repair attempt. No search was made."
)
# Actions that need local evidence (a successful read and a failing test) first.
EVIDENCE_GATED = {"edit_file": EARLY_EDIT_MESSAGE, "search_github": EARLY_SEARCH_MESSAGE}


class InvalidReplyError(ValueError):
    """Raised when a model reply is not exactly one permitted JSON action."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """json.loads hook that refuses objects with repeated keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidReplyError(f"Duplicate key in JSON object: {key!r}")
        result[key] = value
    return result


def parse_action(raw: str, allow_research: bool = False) -> ModelAction:
    """Parse one raw model reply into a ModelAction, or raise InvalidReplyError."""
    allowed = {**ACTION_FIELDS, **RESEARCH_ACTION_FIELDS} if allow_research else ACTION_FIELDS
    if not isinstance(raw, str):
        raise InvalidReplyError("Reply must be text.")
    text = raw.strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise InvalidReplyError("Reply must be exactly one JSON object, with no prose or Markdown.")
    try:
        data = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise InvalidReplyError(f"Malformed JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise InvalidReplyError("Reply must be a JSON object.")

    tool = data.get("tool")
    if not isinstance(tool, str):
        raise InvalidReplyError("Missing or non-string field: 'tool'.")
    if tool not in allowed:
        raise InvalidReplyError(f"Unknown tool: {tool!r}.")

    required, optional = allowed[tool]
    fields = set(data) - {"tool"}
    missing = required - fields
    if missing:
        raise InvalidReplyError(f"{tool} is missing field(s): {', '.join(sorted(missing))}.")
    extra = fields - required - optional
    if extra:
        raise InvalidReplyError(f"{tool} does not allow field(s): {', '.join(sorted(extra))}.")
    for name in fields:
        if not isinstance(data[name], str):
            raise InvalidReplyError(f"Field {name!r} must be a string.")

    arguments = {name: data[name] for name in sorted(required)}
    return ModelAction(tool=tool, arguments=arguments, summary=data.get("summary"))


def _preview(text: str) -> str:
    """Shorten long text for the visible trace."""
    if len(text) <= TRACE_PREVIEW_CHARS:
        return text
    return f"{text[:TRACE_PREVIEW_CHARS]}\n[... {len(text) - TRACE_PREVIEW_CHARS} more characters]"


class RepairAgent:
    """Runs one bounded repair attempt: ask, validate, call one tool, observe, repeat."""

    def __init__(
        self,
        client: ChatClient,
        tools: RepairToolsPort,
        max_decisions: int = DEFAULT_MAX_DECISIONS,
        recorder: TraceRecorder | None = None,
        research: ResearchPort | None = None,
    ) -> None:
        if max_decisions < 1:
            raise ValueError("max_decisions must be at least 1.")
        self.client = client
        self.tools = tools
        self.max_decisions = max_decisions
        self.recorder = recorder
        self.research = research  # None keeps search_github out of the parser entirely.

    def run(
        self, case_name: str, files: Sequence[str], workspace: Path | None = None
    ) -> RepairResult:
        """Attempt one repair and always return a RepairResult, never raise."""
        attempt = _Attempt(self.recorder)
        research_enabled = self.research is not None
        messages = [system_message(research_enabled), case_request_message(case_name, files)]
        attempt.emit(
            "repair_started",
            case=case_name,
            files=list(files),
            max_decisions=self.max_decisions,
            research_enabled=research_enabled,
        )

        for decision in range(1, self.max_decisions + 1):
            try:
                raw = self.client.chat(list(messages))
            except Exception as exc:  # A client failure ends the attempt, visibly.
                return attempt.finish(False, f"Model client error: {exc}", decision - 1, workspace)

            try:
                action = parse_action(raw, allow_research=research_enabled)
            except InvalidReplyError as exc:
                attempt.emit("model_response", decision=decision, valid=False, error=str(exc))
                return attempt.finish(False, f"Invalid model reply: {exc}", decision, workspace)

            messages.append({"role": "assistant", "content": raw.strip()})
            attempt.emit(
                "model_response",
                decision=decision,
                valid=True,
                tool=action.tool,
                path=action.arguments.get("path"),
                query=action.arguments.get("query"),
                summary=action.summary,
            )

            if action.tool in EVIDENCE_GATED and not attempt.has_local_evidence:
                policy_error = EVIDENCE_GATED[action.tool]
                messages.append(tool_result_message(action.tool, policy_error, is_error=True))
                attempt.emit(
                    "tool_result",
                    decision=decision,
                    tool=action.tool,
                    status="rejected",
                    observation=policy_error,
                )
                continue

            try:
                observation = self._call_tool(action)
            except Exception as exc:  # Tool errors go back to the model as observations.
                error_text = f"{type(exc).__name__}: {exc}"
                messages.append(tool_result_message(action.tool, error_text, is_error=True))
                attempt.emit(
                    "tool_result", decision=decision, tool=action.tool, status="error", observation=error_text
                )
                continue

            if isinstance(observation, dict):  # Research evidence, shown with its source URLs.
                messages.append(tool_result_message(action.tool, format_research_evidence(observation)))
            else:
                messages.append(tool_result_message(action.tool, observation))
            attempt.record_success(action, observation, decision)

            if isinstance(observation, TestResult) and observation.passed:
                if attempt.edited:
                    return attempt.finish(True, "Tests pass after the edit.", decision, workspace)
                return attempt.finish(
                    False, "Tests passed before any edit: no reproducible bug.", decision, workspace
                )

        return attempt.finish(
            False,
            f"Decision budget of {self.max_decisions} reached without a verified repair.",
            self.max_decisions,
            workspace,
        )

    def _call_tool(self, action: ModelAction) -> str | TestResult | dict[str, Any]:
        """Dispatch one validated action to the matching permitted tool."""
        if action.tool == "search_github" and self.research is not None:
            return self.research.search_github(action.arguments["query"])
        if action.tool == "read_file":
            return self.tools.read_file(action.arguments["path"])
        if action.tool == "edit_file":
            return self.tools.edit_file(action.arguments["path"], action.arguments["content"])
        if action.tool == "run_tests":
            return self.tools.run_tests()
        raise InvalidReplyError(f"Unknown tool: {action.tool!r}.")  # Unreachable after parsing.


class _Attempt:
    """Evidence, trace and changed files gathered during one repair attempt."""

    def __init__(self, recorder: TraceRecorder | None) -> None:
        self.recorder = recorder
        self.trace: list[dict[str, Any]] = []
        self.changed_files: list[str] = []
        self.latest_test: TestResult | None = None
        self.read_ok = False
        self.saw_failing_test = False
        self.edited = False

    @property
    def has_local_evidence(self) -> bool:
        """Edits and searches need a successful read and a failing test first."""
        return self.read_ok and self.saw_failing_test

    def emit(self, event: str, **payload: Any) -> None:
        """Append one public trace event and forward it to the optional recorder."""
        self.trace.append({"event": event, **payload})
        if self.recorder is not None:
            try:
                self.recorder.record(event, payload)
            except Exception:  # Optional observability must never break a repair.
                pass

    def record_success(
        self, action: ModelAction, observation: str | TestResult | dict[str, Any], decision: int
    ) -> None:
        """Update the evidence after a tool call succeeded and log it."""
        if action.tool == "read_file":
            self.read_ok = True
        elif action.tool == "edit_file":
            self.edited = True
            path = posixpath.normpath(action.arguments["path"].replace("\\", "/"))
            if path not in self.changed_files:
                self.changed_files.append(path)

        if isinstance(observation, TestResult):
            self.latest_test = observation
            if not observation.passed:
                self.saw_failing_test = True
            self.emit(
                "tool_result",
                decision=decision,
                tool=action.tool,
                status="ok",
                passed=observation.passed,
                returncode=observation.returncode,
                observation=_preview(observation.output),
            )
        elif isinstance(observation, dict):
            # Keep every source URL even if the text preview is shortened (provenance).
            self.emit(
                "tool_result",
                decision=decision,
                tool=action.tool,
                status="ok",
                query=observation.get("query"),
                sources=[source.get("url") for source in observation.get("sources") or []],
                observation=_preview(format_research_evidence(observation)),
            )
        else:
            self.emit(
                "tool_result", decision=decision, tool=action.tool, status="ok", observation=_preview(observation)
            )

    def finish(self, passed: bool, reason: str, decisions_used: int, workspace: Path | None) -> RepairResult:
        """Log the outcome and build the RepairResult."""
        self.emit("repair_finished", passed=passed, reason=reason, decisions_used=decisions_used)
        return RepairResult(
            passed=passed,
            trace=list(self.trace),
            changed_files=list(self.changed_files),
            workspace=workspace,
            latest_test=self.latest_test,
        )
