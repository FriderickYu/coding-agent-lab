"""Select one supplied evaluation project, copy it, and run the repair agent on the copy."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePosixPath, PureWindowsPath

from core.models import RepairResult
from core.ports import TraceRecorder
from core.repair_agent import RepairAgent

# Copies one project directory to a new disposable workspace and returns its path.
WorkspaceFactory = Callable[[Path], Path]
# Builds an agent for (case name, workspace, optional trace recorder).
AgentFactory = Callable[[str, Path, "TraceRecorder | None"], RepairAgent]

IGNORED_PARTS = {"__pycache__", ".pytest_cache"}


class ProjectSelectionError(ValueError):
    """Raised when a requested project is not one of the supplied evaluation projects."""


class RepairService:
    """Accepts only projects directly under evals/ and repairs a disposable copy."""

    def __init__(
        self,
        evals_root: Path,
        create_workspace: WorkspaceFactory,
        make_agent: AgentFactory,
    ) -> None:
        self.evals_root = Path(evals_root).resolve()
        self._create_workspace = create_workspace
        self._make_agent = make_agent

    def available_cases(self) -> list[str]:
        """Names of the supplied projects: folders under evals/ that contain a test file."""
        if not self.evals_root.is_dir():
            return []
        return sorted(
            child.name
            for child in self.evals_root.iterdir()
            if child.is_dir()
            and not child.name.startswith((".", "_"))
            and any(child.glob("test_*.py"))
        )

    def resolve_case(self, selection: str | Path) -> Path:
        """Map 'evals/<case>' or '<case>' to a supplied project directory, or reject it."""
        text = str(selection).strip()
        if not text:
            raise ProjectSelectionError("Choose one project under evals/.")
        if PurePosixPath(text).is_absolute() or PureWindowsPath(text).anchor:
            raise ProjectSelectionError(f"Absolute paths are not allowed: {text}")
        parts = [part for part in PurePosixPath(text.replace("\\", "/")).parts if part != "."]
        if ".." in parts:
            raise ProjectSelectionError(f"Path is outside evals/: {text}")
        if parts and parts[0] == "evals":
            parts = parts[1:]
        if len(parts) != 1:
            raise ProjectSelectionError(f"Choose one project directly under evals/: {text}")
        name = parts[0]
        if name not in self.available_cases():
            known = ", ".join(self.available_cases())
            raise ProjectSelectionError(f"Unknown project {name!r}. Choose one of: {known}")
        project = (self.evals_root / name).resolve()
        if project.parent != self.evals_root:
            raise ProjectSelectionError(f"Path is outside evals/: {text}")
        return project

    def repair(self, selection: str | Path, recorder: TraceRecorder | None = None) -> RepairResult:
        """Copy the selected project, run the agent on the copy, and return its result."""
        project = self.resolve_case(selection)
        workspace = Path(self._create_workspace(project)).resolve()
        if workspace == project or workspace.is_relative_to(self.evals_root):
            raise RuntimeError("Workspace must be a copy outside evals/.")
        files = workspace_files(workspace)
        agent = self._make_agent(project.name, workspace, recorder)
        return agent.run(project.name, files, workspace)


def workspace_files(workspace: Path) -> list[str]:
    """Relative POSIX paths of the files in a workspace, skipping caches."""
    root = Path(workspace)
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not IGNORED_PARTS.intersection(path.relative_to(root).parts)
    )
