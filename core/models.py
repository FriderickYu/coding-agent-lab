"""Shared result objects that move through the repair workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TestResult:
    """Outcome of one sandboxed pytest run."""

    __test__ = False  # Stop pytest from collecting this class as a test.

    passed: bool
    output: str
    returncode: int


@dataclass
class ModelAction:
    """One validated tool request proposed by the model."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None  # Short public description, never private reasoning.


@dataclass
class RepairResult:
    """Final outcome of one repair attempt."""

    passed: bool
    trace: list[dict[str, Any]] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    workspace: Path | None = None
    latest_test: TestResult | None = None
