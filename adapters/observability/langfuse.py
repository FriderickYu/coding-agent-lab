"""Optional Langfuse Cloud recorder for the public repair trajectory.

It implements core.ports.TraceRecorder. It only observes: every SDK call is
best-effort, so a recording failure is logged and never changes or stops a repair.
Only a small allow-listed payload is sent; never prompts, model text, file
content, full test output, or credentials.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

log = logging.getLogger(__name__)

REQUIRED_ENV = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
MAX_SUMMARY_CHARS = 200


class ObservabilityConfigError(ValueError):
    """Raised when --observe is requested but a Langfuse variable is missing."""


def read_langfuse_settings(environ: Mapping[str, str]) -> dict[str, str]:
    """Return the three Langfuse Cloud settings, or name (never print) the missing ones."""
    missing = [name for name in REQUIRED_ENV if not (environ.get(name) or "").strip()]
    if missing:
        raise ObservabilityConfigError(
            f"--observe needs these environment variables: {', '.join(missing)}. "
            "Set them in this terminal only; never commit them."
        )
    return {
        "public_key": environ["LANGFUSE_PUBLIC_KEY"].strip(),
        "secret_key": environ["LANGFUSE_SECRET_KEY"].strip(),
        "base_url": environ["LANGFUSE_BASE_URL"].strip(),
    }


def create_langfuse_client(settings: Mapping[str, str]) -> Any:
    """Create the installed SDK's client for Langfuse Cloud (imported lazily: it is slow)."""
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings["public_key"],
        secret_key=settings["secret_key"],
        base_url=settings["base_url"],
    )


def _summary(value: Any) -> str | None:
    """Short public summary label, or None."""
    return str(value)[:MAX_SUMMARY_CHARS] if value else None


class LangfuseTrajectoryRecorder:
    """One root agent observation per repair, with closed child observations per event."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._root: Any = None
        self._case: str | None = None
        self.trace_id: str | None = None
        self.failures = 0

    def record(self, event: str, payload: dict[str, Any]) -> None:
        """Record one public event; failures are logged and swallowed."""
        handler = {
            "repair_started": self._started,
            "model_response": self._model_response,
            "tool_result": self._tool_result,
            "repair_finished": self._finished,
        }.get(event)
        if handler is None:
            return
        try:
            handler(payload)
        except Exception as exc:  # Observability must never affect the repair.
            self._failed(event, exc)

    def trace_url(self) -> str | None:
        """Best-effort link to this repair's trace in Langfuse Cloud."""
        if not self.trace_id:
            return None
        try:
            return self._client.get_trace_url(trace_id=self.trace_id)
        except Exception as exc:
            self._failed("trace_url", exc)
            return None

    def _failed(self, what: str, exc: Exception) -> None:
        """Count and log a failure by type only, so no credential or payload can leak into logs."""
        self.failures += 1
        log.warning("Langfuse recording failed during %s (%s); the repair continues.", what, type(exc).__name__)

    def _started(self, payload: dict[str, Any]) -> None:
        self._case = str(payload.get("case"))
        self._root = self._client.start_observation(
            name=f"repair {self._case}",
            as_type="agent",
            input={"case": self._case},
        )
        self.trace_id = getattr(self._root, "trace_id", None)

    def _child(self, name: str, as_type: str, data: dict[str, Any], warn: bool = False) -> None:
        """Start and immediately close one child observation under the root."""
        if self._root is None:
            return
        child = self._root.start_observation(
            name=name,
            as_type=as_type,
            output={key: value for key, value in data.items() if value is not None},
            level="WARNING" if warn else "DEFAULT",
        )
        child.end()

    def _model_response(self, payload: dict[str, Any]) -> None:
        step = payload.get("decision")
        valid = bool(payload.get("valid"))
        data = {
            "case": self._case,
            "step": step,
            "status": "valid" if valid else "invalid",
            "tool": payload.get("tool") if valid else None,
            "summary": _summary(payload.get("summary")) if valid else None,
        }
        self._child(f"step {step} model", "span", data, warn=not valid)

    def _tool_result(self, payload: dict[str, Any]) -> None:
        step, tool, status = payload.get("decision"), payload.get("tool"), payload.get("status")
        data = {
            "case": self._case,
            "step": step,
            "tool": tool,
            "status": status,
            "passed": payload.get("passed"),
            "returncode": payload.get("returncode"),
        }
        self._child(f"step {step} {tool}", "tool", data, warn=status != "ok")

    def _finished(self, payload: dict[str, Any]) -> None:
        """Close the root with the final outcome, then always try to flush."""
        try:
            if self._root is not None:
                passed = bool(payload.get("passed"))
                self._root.update(
                    output={
                        "case": self._case,
                        "passed": passed,
                        "outcome": payload.get("outcome"),
                        "steps": payload.get("decisions_used"),
                        "changed_files": list(payload.get("changed_files") or []),
                    },
                    level="DEFAULT" if passed else "WARNING",
                )
                self._root.end()
        finally:
            self._root = None
            try:
                self._client.flush()
            except Exception as exc:
                self._failed("flush", exc)
