"""The three workspace-confined tools the repair agent may use."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol

from core.models import TestResult


class UnsafePathError(ValueError):
    """Raised when a requested path is absolute or escapes the workspace."""


class WorkspaceTestRunner(Protocol):
    """Anything that can run a workspace's tests in the sandbox."""

    def run_tests(self, workspace: Path) -> TestResult:
        """Run the tests for one workspace and return the result."""
        ...


def safe_path(workspace: Path, requested_path: str) -> Path:
    """Resolve a relative path inside the workspace, rejecting anything outside it."""
    if not isinstance(requested_path, str) or not requested_path.strip():
        raise UnsafePathError("Path must be a non-empty relative path.")
    # Check both styles so "/etc/passwd" and "C:\\x" are rejected on every OS.
    if PurePosixPath(requested_path).is_absolute() or PureWindowsPath(requested_path).anchor:
        raise UnsafePathError(f"Absolute paths are not allowed: {requested_path}")
    root = Path(workspace).resolve()
    resolved = (root / requested_path).resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise UnsafePathError(f"Path escapes the workspace: {requested_path}")
    return resolved


class RepairTools:
    """Reads, edits and tests files in one disposable workspace only."""

    def __init__(self, workspace: Path, runner: WorkspaceTestRunner) -> None:
        self.workspace = Path(workspace).resolve()
        self._runner = runner
        self._changed_files: list[str] = []

    @property
    def changed_files(self) -> list[str]:
        """Relative paths edited so far, each listed once in first-edit order."""
        return list(self._changed_files)

    def read_file(self, path: str) -> str:
        """Return the UTF-8 text of an existing workspace file."""
        target = safe_path(self.workspace, path)
        if not target.is_file():
            raise FileNotFoundError(f"File not found in workspace: {path}")
        return target.read_text(encoding="utf-8")

    def edit_file(self, path: str, content: str) -> str:
        """Replace an existing workspace file's content and record the change."""
        target = safe_path(self.workspace, path)
        if not target.is_file():
            raise FileNotFoundError(f"File not found in workspace: {path}")
        if not isinstance(content, str):
            raise TypeError("File content must be a string.")
        target.write_text(content, encoding="utf-8", newline="")
        relative = target.relative_to(self.workspace).as_posix()
        if relative not in self._changed_files:
            self._changed_files.append(relative)
        return f"Wrote {len(content)} characters to {relative}."

    def run_tests(self) -> TestResult:
        """Delegate the test run to the Docker-backed runner."""
        return self._runner.run_tests(self.workspace)
