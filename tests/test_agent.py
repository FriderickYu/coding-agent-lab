"""Repair-loop tests with ScriptedClient and fake tools: no Docker, no model, no eval code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from adapters.models.scripted import ScriptedClient
from core.models import TestResult
from core.repair_agent import RepairAgent, parse_action

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
FILES = ["calculator.py", "test_calculator.py"]
WORKSPACE = Path("fake-workspace")
FAIL = TestResult(passed=False, output="1 failed", returncode=1)
PASS = TestResult(passed=True, output="2 passed", returncode=0)


def card(**fields: Any) -> str:
    """Build one scripted model reply."""
    return json.dumps(fields)


READ = card(tool="read_file", path="calculator.py", summary="Read the source")
RUN = card(tool="run_tests", summary="Run the tests")
EDIT = card(tool="edit_file", path="calculator.py", content=FIXED, summary="Fix add")


class FakeTools:
    """In-memory stand-in for RepairTools that records every call."""

    def __init__(self, test_results: list[TestResult]) -> None:
        self.files = {"calculator.py": BUGGY, "test_calculator.py": "def test_add(): ...\n"}
        self.test_results = list(test_results)
        self.calls: list[str] = []

    def read_file(self, path: str) -> str:
        self.calls.append(f"read_file:{path}")
        if path.startswith("..") or path.startswith("/"):
            raise ValueError(f"Path escapes the workspace: {path}")
        if path not in self.files:
            raise FileNotFoundError(f"File not found in workspace: {path}")
        return self.files[path]

    def edit_file(self, path: str, content: str) -> str:
        self.calls.append(f"edit_file:{path}")
        self.files[path] = content
        return f"Wrote {len(content)} characters to {path}."

    def run_tests(self) -> TestResult:
        self.calls.append("run_tests")
        return self.test_results.pop(0)


class ListRecorder:
    """TraceRecorder that keeps the event names it receives."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def record(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append(event)


def run_agent(replies: list[str], tests: list[TestResult], **kwargs: Any):
    client = ScriptedClient(replies)
    tools = FakeTools(tests)
    result = RepairAgent(client, tools, **kwargs).run("arithmetic", FILES, WORKSPACE)
    return result, client, tools


def finish_reason(result) -> str:
    final = result.trace[-1]
    assert final["event"] == "repair_finished"
    return final["reason"]


def test_read_fail_edit_pass_is_a_successful_repair() -> None:
    recorder = ListRecorder()
    result, client, tools = run_agent([READ, RUN, EDIT, RUN], [FAIL, PASS], recorder=recorder)

    assert result.passed is True
    assert tools.calls == ["read_file:calculator.py", "run_tests", "edit_file:calculator.py", "run_tests"]
    assert tools.files["calculator.py"] == FIXED
    assert result.changed_files == ["calculator.py"]
    assert result.latest_test == PASS
    assert result.workspace == WORKSPACE
    assert [entry["event"] for entry in result.trace] == [
        "repair_started",
        "model_response", "tool_result",
        "model_response", "tool_result",
        "model_response", "tool_result",
        "model_response", "tool_result",
        "repair_finished",
    ]
    assert result.trace[1]["summary"] == "Read the source"
    assert recorder.events == [entry["event"] for entry in result.trace]
    # Each later decision sees the accepted reply and the tool observation.
    second_call = client.received[1]
    assert [m["role"] for m in second_call] == ["system", "user", "assistant", "user"]
    assert BUGGY in second_call[-1]["content"]


@pytest.mark.parametrize(
    "reply",
    [
        "I will read the file first.",
        'Sure! {"tool":"run_tests"}',
        '```json\n{"tool":"run_tests"}\n```',
        '[{"tool":"run_tests"}]',
        '{"tool":"run_tests"',
        '{"tool":"run_tests"} {"tool":"run_tests"}',
        '{"tool":"read_file","tool":"run_tests","path":"calculator.py"}',
        '{"tool":"shell","command":"ls"}',
        '{"tool":"read_file"}',
        '{"tool":"edit_file","path":"calculator.py"}',
        '{"tool":"run_tests","path":"calculator.py"}',
        '{"tool":"read_file","path":["calculator.py"]}',
        '{"path":"calculator.py"}',
    ],
)
def test_invalid_reply_stops_without_calling_a_tool(reply: str) -> None:
    result, client, tools = run_agent([reply, RUN], [FAIL])

    assert result.passed is False
    assert tools.calls == []
    assert client.remaining == 1
    assert finish_reason(result).startswith("Invalid model reply")
    assert result.trace[-2] == {
        "event": "model_response", "decision": 1, "valid": False, "error": result.trace[-2]["error"]
    }


def test_edit_before_failing_test_is_rejected_then_repair_continues() -> None:
    result, client, tools = run_agent([READ, EDIT, RUN, EDIT, RUN], [FAIL, PASS])

    assert result.passed is True
    assert tools.calls.count("edit_file:calculator.py") == 1
    assert tools.calls[:2] == ["read_file:calculator.py", "run_tests"]
    rejected = [entry for entry in result.trace if entry.get("status") == "rejected"]
    assert len(rejected) == 1 and rejected[0]["decision"] == 2
    assert "Policy error" in client.received[2][-1]["content"]


def test_edit_before_any_read_is_rejected() -> None:
    result, _, tools = run_agent([RUN, EDIT], [FAIL])

    assert result.passed is False
    assert tools.calls == ["run_tests"]
    assert result.changed_files == []


def test_budget_exhaustion_returns_failure() -> None:
    result, client, tools = run_agent([READ] * 9, [], max_decisions=8)

    assert result.passed is False
    assert client.remaining == 1  # Exactly eight decisions were requested.
    assert tools.calls == ["read_file:calculator.py"] * 8
    assert "budget of 8" in finish_reason(result)


def test_still_failing_after_edit_keeps_going() -> None:
    result, _, tools = run_agent([READ, RUN, EDIT, RUN, EDIT, RUN], [FAIL, FAIL, PASS])

    assert result.passed is True
    assert tools.calls.count("run_tests") == 3
    assert result.changed_files == ["calculator.py"]


def test_passing_baseline_is_not_reported_as_a_repair() -> None:
    result, _, tools = run_agent([READ, RUN, EDIT], [PASS])

    assert result.passed is False
    assert tools.calls == ["read_file:calculator.py", "run_tests"]
    assert result.changed_files == []
    assert result.latest_test == PASS
    assert "no reproducible bug" in finish_reason(result)


def test_client_error_still_returns_repair_result() -> None:
    result, _, tools = run_agent([READ], [])

    assert result.passed is False
    assert tools.calls == ["read_file:calculator.py"]
    assert finish_reason(result).startswith("Model client error")
    assert result.workspace == WORKSPACE


@pytest.mark.parametrize("bad_path", ["missing.py", "../secret.txt"])
def test_tool_error_becomes_an_observation(bad_path: str) -> None:
    bad_read = card(tool="read_file", path=bad_path)
    result, client, _ = run_agent([bad_read, READ, RUN, EDIT, RUN], [FAIL, PASS])

    assert result.passed is True
    errors = [entry for entry in result.trace if entry.get("status") == "error"]
    assert len(errors) == 1 and errors[0]["decision"] == 1
    assert "Tool error for read_file" in client.received[1][-1]["content"]


def test_failed_read_does_not_unlock_edit() -> None:
    bad_read = card(tool="read_file", path="missing.py")
    result, _, tools = run_agent([bad_read, RUN, EDIT], [FAIL])

    assert result.passed is False
    assert "edit_file:calculator.py" not in tools.calls


def test_parse_action_builds_model_action() -> None:
    action = parse_action(EDIT)
    assert action.tool == "edit_file"
    assert action.arguments == {"content": FIXED, "path": "calculator.py"}
    assert action.summary == "Fix add"
    assert parse_action('  {"tool":"run_tests"}\n').arguments == {}
