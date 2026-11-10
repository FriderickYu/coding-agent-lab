"""Copy an evaluation project to a disposable workspace and test it in Docker."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from core.models import TestResult

DEFAULT_IMAGE = "coding-agent-lab-runner"
# Under the home directory, never the system temp folder: Colima, Rancher
# Desktop and Podman share only the home directory with their Linux VM.
DEFAULT_WORKSPACE_ROOT = Path.home() / ".cache" / "coding-agent-lab"
CONTAINER_WORKSPACE = "/workspace"
PYTEST_COMMAND = ("pytest", "-q", "-p", "no:cacheprovider")


class DockerRunner:
    """Creates disposable workspaces and runs their tests in a locked-down container."""

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
        timeout_seconds: int = 120,
    ) -> None:
        self.image = image
        self.workspace_root = Path(workspace_root)
        self.timeout_seconds = timeout_seconds

    def create_workspace(self, project_dir: Path) -> Path:
        """Copy one project directory into a new folder under the workspace root."""
        source = Path(project_dir).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Project directory not found: {project_dir}")
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix=f"{source.name}-", dir=self.workspace_root))
        shutil.copytree(
            source,
            workspace,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
        )
        return workspace

    def docker_command(self, workspace: Path) -> list[str]:
        """Build the docker run command for one workspace, without running it."""
        source = Path(workspace).resolve()
        return [
            "docker", "run", "--rm",
            "--network", "none",
            "--read-only",
            "--tmpfs", "/tmp",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", "128",
            "--memory", "512m",
            "--mount", f"type=bind,source={source},target={CONTAINER_WORKSPACE},readonly",
            "--workdir", CONTAINER_WORKSPACE,
            self.image,
            *PYTEST_COMMAND,
        ]

    def run_tests(self, workspace: Path) -> TestResult:
        """Run pytest for the workspace inside Docker and return a structured result."""
        try:
            completed = subprocess.run(
                self.docker_command(workspace),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            return TestResult(False, "Docker command not found; is Docker installed?", -1)
        except subprocess.TimeoutExpired:
            return TestResult(False, f"Tests timed out after {self.timeout_seconds} seconds.", -1)
        output = (completed.stdout or "") + (completed.stderr or "")
        return TestResult(completed.returncode == 0, output, completed.returncode)
