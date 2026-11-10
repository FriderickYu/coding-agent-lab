"""A fake chat client that returns pre-written replies in order, for repeatable tests."""

from __future__ import annotations

from collections.abc import Sequence


class ScriptedClientExhausted(RuntimeError):
    """Raised when chat is called after every scripted reply has been used."""


class ScriptedClient:
    """Answer-card model: each chat call returns the next prepared reply string."""

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)
        self._next = 0
        self.received: list[list[dict[str, str]]] = []

    @property
    def remaining(self) -> int:
        """Number of replies not yet returned."""
        return len(self._replies) - self._next

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Record the conversation and return the next reply; it never calls a model."""
        if self._next >= len(self._replies):
            raise ScriptedClientExhausted(
                f"ScriptedClient has no replies left after {len(self._replies)} call(s)."
            )
        self.received.append([dict(message) for message in messages])
        reply = self._replies[self._next]
        self._next += 1
        return reply
