"""Unit tests for the workspace boundary, file tools, and Docker delegation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from adapters.sandbox import runner as runner_module
from adapters.sandbox.repair_tools import RepairTools, UnsafePathError, safe_path
from adapters.sandbox.runner import DockerRunner
from core.models import TestResult


class FakeRunner:
    """Records which workspace it was asked to test and returns a canned result."""

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def run_tests(self, workspace: Path) -> TestResult:
        self.calls.append(workspace)
        return TestResult(passed=False, output="1 failed", returncode=1)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "pkg").mkdir(parents=True)
    (root / "calculator.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (root / "pkg" / "util.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("do not read", encoding="utf-8")
    return root


def test_reads_normal_workspace_file(workspace: Path) -> None:
    tools = RepairTools(workspace, FakeRunner())
    assert tools.read_file("calculator.py") == "def add(a, b):\n    return a - b\n"
    assert tools.read_file("pkg/util.py") == "VALUE = 1\n"


@pytest.mark.parametrize(
    "bad_path",
    ["/etc/passwd", "C:\\Windows\\win.ini", "C:secret.txt", "\\secret.txt"],
)
def test_rejects_absolute_paths(workspace: Path, bad_path: str) -> None:
    with pytest.raises(UnsafePathError):
        safe_path(workspace, bad_path)


@pytest.mark.parametrize("bad_path", ["../secret.txt", "pkg/../../secret.txt", "..", ".", ""])
def test_rejects_paths_that_escape_workspace(workspace: Path, bad_path: str) -> None:
    tools = RepairTools(workspace, FakeRunner())
    with pytest.raises(UnsafePathError):
        tools.read_file(bad_path)
    with pytest.raises(UnsafePathError):
        tools.edit_file(bad_path, "x")
    assert (workspace.parent / "secret.txt").read_text(encoding="utf-8") == "do not read"


def test_edit_is_saved_and_recorded_once(workspace: Path) -> None:
    tools = RepairTools(workspace, FakeRunner())
    fixed = "def add(a, b):\n    return a + b\n"
    tools.edit_file("calculator.py", fixed)
    tools.edit_file("./calculator.py", fixed)
    assert (workspace / "calculator.py").read_text(encoding="utf-8") == fixed
    assert tools.changed_files == ["calculator.py"]


def test_edit_rejects_missing_file(workspace: Path) -> None:
    tools = RepairTools(workspace, FakeRunner())
    with pytest.raises(FileNotFoundError):
        tools.edit_file("new_module.py", "x = 1\n")
    assert tools.changed_files == []


def test_run_tests_delegates_to_runner(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbid(*args: object, **kwargs: object) -> None:
        raise AssertionError("RepairTools must not launch a host process")

    monkeypatch.setattr(subprocess, "run", forbid)
    monkeypatch.setattr(subprocess, "Popen", forbid)
    fake = FakeRunner()
    result = RepairTools(workspace, fake).run_tests()
    assert fake.calls == [workspace.resolve()]
    assert result == TestResult(passed=False, output="1 failed", returncode=1)


def test_docker_runner_uses_locked_down_container(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        return subprocess.CompletedProcess(command, 1, stdout="1 failed\n", stderr="")

    monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
    result = DockerRunner(image="test-image").run_tests(workspace)

    command = captured["command"]
    assert command[:3] == ["docker", "run", "--rm"]
    assert command[command.index("--network") + 1] == "none"
    mount = command[command.index("--mount") + 1]
    assert f"source={workspace.resolve()}" in mount and mount.endswith(",readonly")
    assert command[-5:] == ["test-image", "pytest", "-q", "-p", "no:cacheprovider"]
    assert sys.executable not in command
    assert result == TestResult(passed=False, output="1 failed\n", returncode=1)


def test_create_workspace_copies_without_touching_source(tmp_path: Path) -> None:
    source = tmp_path / "evals" / "demo"
    source.mkdir(parents=True)
    (source / "demo.py").write_text("X = 1\n", encoding="utf-8")
    runner = DockerRunner(workspace_root=tmp_path / "cache")

    copy = runner.create_workspace(source)
    (copy / "demo.py").write_text("X = 2\n", encoding="utf-8")

    assert copy.parent == tmp_path / "cache"
    assert (source / "demo.py").read_text(encoding="utf-8") == "X = 1\n"
    assert runner.create_workspace(source) != copy
