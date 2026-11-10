"""Contracts the repair agent depends on, without choosing implementations."""

from __future__ import annotations

from typing import Any, Protocol

from core.models import TestResult


class ChatClient(Protocol):
    """A model client that answers a conversation with raw text."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Return the model's reply to the given role/content messages."""
        ...


class RepairToolsPort(Protocol):
    """The only workspace operations the repair agent may request."""

    def read_file(self, path: str) -> str:
        """Return the UTF-8 text of a file inside the workspace."""
        ...

    def edit_file(self, path: str, content: str) -> str:
        """Replace a workspace file's content and return a short confirmation."""
        ...

    def run_tests(self) -> TestResult:
        """Run the workspace tests in the sandbox and return the result."""
        ...


class TraceRecorder(Protocol):
    """Optional sink for public repair trace events."""

    def record(self, event: str, payload: dict[str, Any]) -> None:
        """Record one trace event with its public payload."""
        ...
