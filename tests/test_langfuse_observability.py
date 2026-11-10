"""Tests for the optional Langfuse recorder with a fake client. They never contact Langfuse."""

from __future__ import annotations

import json
import logging
import socket
from pathlib import Path
from typing import Any

import pytest

import main
from adapters.models.scripted import ScriptedClient
from adapters.observability.langfuse import (
    LangfuseTrajectoryRecorder,
    ObservabilityConfigError,
    read_langfuse_settings,
)
from core.models import RepairResult, TestResult
from core.repair_agent import RepairAgent

SOURCE_MARKER = "SOURCE_CONTENT_MARKER"
REPLACEMENT_MARKER = "REPLACEMENT_CONTENT_MARKER"
OUTPUT_MARKER = "FULL_TEST_OUTPUT_MARKER"
SECRET = "sk-lf-test-secret-value"
FAIL = TestResult(passed=False, output=f"{OUTPUT_MARKER} 1 failed", returncode=1)
PASS = TestResult(passed=True, output=f"{OUTPUT_MARKER} 2 passed", returncode=0)
READ = '{"tool":"read_file","path":"calculator.py","summary":"Read the source"}'
RUN = '{"tool":"run_tests","summary":"Run the tests"}'
EDIT = json.dumps(
    {"tool": "edit_file", "path": "calculator.py", "content": REPLACEMENT_MARKER, "summary": "Fix add"}
)
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that connects beyond this computer."""
    real_connect = socket.socket.connect

    def guarded_connect(sock: socket.socket, address: Any) -> Any:
        if isinstance(address, tuple) and address[0] in LOOPBACK:
            return real_connect(sock, address)
        raise AssertionError(f"Tests must not contact the network: {address!r}")

    def refuse_create_connection(address: Any, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"Tests must not contact the network: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", refuse_create_connection)


# ---------- Fake Langfuse client ----------


class FakeObservation:
    """Records what the recorder sends, like a Langfuse observation."""

    def __init__(self, client: FakeLangfuse, name: str, as_type: str, **fields: Any) -> None:
        self.client, self.name, self.as_type, self.fields = client, name, as_type, fields
        self.trace_id = "trace-123"
        self.ended = False

    def start_observation(self, *, name: str, as_type: str = "span", **fields: Any) -> FakeObservation:
        if self.client.fail_on == "child":
            raise RuntimeError(f"child failed {SECRET}")
        self.client.log.append(("start", name, as_type))
        child = FakeObservation(self.client, name, as_type, **fields)
        self.client.observations.append(child)
        return child

    def update(self, **fields: Any) -> FakeObservation:
        if self.client.fail_on == "update":
            raise RuntimeError("update failed")
        self.client.log.append(("update", self.name, None))
        self.fields.update(fields)
        return self

    def end(self) -> FakeObservation:
        self.client.log.append(("end", self.name, None))
        self.ended = True
        return self


class FakeLangfuse:
    """Stand-in for langfuse.Langfuse that never uses the network."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.log: list[tuple[str, str, str | None]] = []
        self.observations: list[FakeObservation] = []

    def start_observation(self, *, name: str, as_type: str = "span", **fields: Any) -> FakeObservation:
        if self.fail_on == "root":
            raise ConnectionError("Langfuse unreachable")
        self.log.append(("start", name, as_type))
        root = FakeObservation(self, name, as_type, **fields)
        self.observations.append(root)
        return root

    def flush(self) -> None:
        if self.fail_on == "flush":
            raise ConnectionError("flush failed")
        self.log.append(("flush", "", None))

    def get_trace_url(self, *, trace_id: str) -> str:
        return f"https://us.cloud.langfuse.com/project/demo/traces/{trace_id}"

    def sent_payload(self) -> str:
        """Everything the recorder handed to the client, as one JSON string."""
        return json.dumps([[o.name, o.as_type, o.fields] for o in self.observations], default=str)


class FakeTools:
    """In-memory RepairToolsPort."""

    def __init__(self, tests: list[TestResult]) -> None:
        self.tests = list(tests)

    def read_file(self, path: str) -> str:
        return f"def add(a, b):\n    return a - b  # {SOURCE_MARKER}\n"

    def edit_file(self, path: str, content: str) -> str:
        return f"Wrote {len(content)} characters to {path}."

    def run_tests(self) -> TestResult:
        return self.tests.pop(0)


def repair(replies: list[str], tests: list[TestResult], recorder: Any = None) -> RepairResult:
    agent = RepairAgent(ScriptedClient(replies), FakeTools(tests), recorder=recorder)
    return agent.run("arithmetic", ["calculator.py", "test_calculator.py"], Path("ws"))


# ---------- Event order and flush ----------


def test_event_order_is_root_then_closed_children_then_flush() -> None:
    client = FakeLangfuse()
    result = repair([READ, RUN, EDIT, RUN], [FAIL, PASS], LangfuseTrajectoryRecorder(client))

    assert result.passed is True
    assert client.log == [
        ("start", "repair arithmetic", "agent"),
        ("start", "step 1 model", "span"), ("end", "step 1 model", None),
        ("start", "step 1 read_file", "tool"), ("end", "step 1 read_file", None),
        ("start", "step 2 model", "span"), ("end", "step 2 model", None),
        ("start", "step 2 run_tests", "tool"), ("end", "step 2 run_tests", None),
        ("start", "step 3 model", "span"), ("end", "step 3 model", None),
        ("start", "step 3 edit_file", "tool"), ("end", "step 3 edit_file", None),
        ("start", "step 4 model", "span"), ("end", "step 4 model", None),
        ("start", "step 4 run_tests", "tool"), ("end", "step 4 run_tests", None),
        ("update", "repair arithmetic", None),
        ("end", "repair arithmetic", None),
        ("flush", "", None),
    ]
    assert all(observation.ended for observation in client.observations)


def test_flush_happens_only_on_repair_finished() -> None:
    client = FakeLangfuse()
    recorder = LangfuseTrajectoryRecorder(client)
    recorder.record("repair_started", {"case": "median"})
    recorder.record("model_response", {"decision": 1, "valid": True, "tool": "run_tests"})
    recorder.record("tool_result", {"decision": 1, "tool": "run_tests", "status": "ok", "passed": False})
    assert ("flush", "", None) not in client.log

    recorder.record("repair_finished", {"passed": False, "outcome": "budget_exhausted"})
    assert client.log[-3:] == [
        ("update", "repair median", None), ("end", "repair median", None), ("flush", "", None)
    ]


def test_root_records_small_final_outcome() -> None:
    client = FakeLangfuse()
    repair([READ, RUN, EDIT, RUN], [FAIL, PASS], LangfuseTrajectoryRecorder(client))

    root = client.observations[0]
    assert root.fields["input"] == {"case": "arithmetic"}
    assert root.fields["output"] == {
        "case": "arithmetic",
        "passed": True,
        "outcome": "repaired",
        "steps": 4,
        "changed_files": ["calculator.py"],
    }
    run_failed = next(o for o in client.observations if o.name == "step 2 run_tests")
    assert run_failed.fields["output"] == {
        "case": "arithmetic", "step": 2, "tool": "run_tests", "status": "ok", "passed": False, "returncode": 1
    }
    model_step = next(o for o in client.observations if o.name == "step 3 model")
    assert model_step.fields["output"] == {
        "case": "arithmetic", "step": 3, "status": "valid", "tool": "edit_file", "summary": "Fix add"
    }


def test_payload_excludes_prompts_model_text_content_output_and_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", SECRET)
    client = FakeLangfuse()
    repair([READ, RUN, EDIT, RUN], [FAIL, PASS], LangfuseTrajectoryRecorder(client))

    sent = client.sent_payload()
    for forbidden in (
        SOURCE_MARKER,            # source-file content from read_file
        REPLACEMENT_MARKER,       # replacement file content from edit_file
        OUTPUT_MARKER,            # full test output
        "careful Python repair agent",  # raw system prompt
        "Repair the project",     # raw user prompt
        '"tool":"read_file"',     # raw model text
        SECRET,                   # credentials
        "LANGFUSE",               # environment variable names
    ):
        assert forbidden not in sent, forbidden


def test_invalid_reply_sends_status_not_model_text() -> None:
    client = FakeLangfuse()
    bad_reply = '{"tool":"shell","command":"rm -rf SECRET_MODEL_TEXT"}'
    repair([bad_reply], [], LangfuseTrajectoryRecorder(client))

    assert "SECRET_MODEL_TEXT" not in client.sent_payload()
    model_step = next(o for o in client.observations if o.name == "step 1 model")
    assert model_step.fields["output"] == {"case": "arithmetic", "step": 1, "status": "invalid"}
    assert model_step.fields["level"] == "WARNING"
    assert client.observations[0].fields["output"]["outcome"] == "invalid_reply"


# ---------- Best-effort failure handling ----------


@pytest.mark.parametrize("fail_on", ["root", "child", "update", "flush"])
def test_recording_failure_never_changes_or_stops_a_repair(
    fail_on: str, caplog: pytest.LogCaptureFixture
) -> None:
    baseline = repair([READ, RUN, EDIT, RUN], [FAIL, PASS])
    client = FakeLangfuse(fail_on=fail_on)
    recorder = LangfuseTrajectoryRecorder(client)

    with caplog.at_level(logging.WARNING, logger="adapters.observability.langfuse"):
        observed = repair([READ, RUN, EDIT, RUN], [FAIL, PASS], recorder)

    assert observed == baseline
    assert recorder.failures >= 1
    assert "Langfuse recording failed" in caplog.text
    assert SECRET not in caplog.text  # Only the exception type is logged.


def test_update_failure_still_flushes() -> None:
    client = FakeLangfuse(fail_on="update")
    repair([READ, RUN, EDIT, RUN], [FAIL, PASS], LangfuseTrajectoryRecorder(client))
    assert client.log[-1] == ("flush", "", None)


# ---------- Configuration and the --observe flag ----------


FULL_ENV = {
    "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
    "LANGFUSE_SECRET_KEY": SECRET,
    "LANGFUSE_BASE_URL": "https://us.cloud.langfuse.com",
}


@pytest.mark.parametrize("missing", sorted(FULL_ENV))
def test_each_missing_variable_is_named_without_echoing_values(missing: str) -> None:
    environ = {key: value for key, value in FULL_ENV.items() if key != missing}
    with pytest.raises(ObservabilityConfigError) as error:
        read_langfuse_settings(environ)
    assert missing in str(error.value)
    assert SECRET not in str(error.value)


def test_complete_settings_are_accepted() -> None:
    settings = read_langfuse_settings(FULL_ENV)
    assert settings["base_url"] == "https://us.cloud.langfuse.com"


class FakeService:
    """Replays a short trace through the given recorder instead of running Docker."""

    def __init__(self) -> None:
        self.recorders: list[Any] = []

    def repair(self, project: str, recorder: Any = None) -> RepairResult:
        self.recorders.append(recorder)
        events = [
            ("repair_started", {"case": "arithmetic", "files": ["calculator.py"], "max_decisions": 8}),
            ("model_response", {"decision": 1, "valid": True, "tool": "run_tests", "summary": "Run"}),
            ("tool_result", {"decision": 1, "tool": "run_tests", "status": "ok", "passed": True,
                             "returncode": 0, "observation": "2 passed"}),
            ("repair_finished", {"passed": True, "outcome": "repaired", "reason": "Tests pass after the edit.",
                                 "decisions_used": 1, "changed_files": ["calculator.py"]}),
        ]
        for event, payload in events:
            recorder.record(event, payload)
        return RepairResult(True, [], ["calculator.py"], Path("ws"), TestResult(True, "2 passed", 0))


def test_observe_with_missing_variable_fails_before_any_repair(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in FULL_ENV:
        monkeypatch.delenv(name, raising=False)
    service = FakeService()
    monkeypatch.setattr(main, "build_service", lambda *args, **kwargs: service)

    status = main.main(["repair", "evals/arithmetic", "--provider", "scripted", "--observe"])

    assert status == 2
    assert service.recorders == []
    assert "LANGFUSE_PUBLIC_KEY" in capsys.readouterr().err


def test_observe_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_build(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("observer built without --observe")

    service = FakeService()
    monkeypatch.setattr(main, "build_service", lambda *args, **kwargs: service)
    monkeypatch.setattr(main, "build_observer", must_not_build)

    assert main.main(["repair", "evals/arithmetic", "--provider", "scripted"]) == 0
    assert isinstance(service.recorders[0], main.PrintRecorder)


def test_terminal_trace_is_unchanged_when_observing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(main, "build_service", lambda *args, **kwargs: FakeService())
    main.main(["repair", "evals/arithmetic"])
    plain = capsys.readouterr().out

    client = FakeLangfuse()
    monkeypatch.setattr(main, "build_observer", lambda: LangfuseTrajectoryRecorder(client))
    main.main(["repair", "evals/arithmetic", "--observe"])
    observed = capsys.readouterr().out

    assert observed.startswith(plain)
    assert observed[len(plain):].startswith("Langfuse: trajectory recorded.")
    assert client.log[-1] == ("flush", "", None)
