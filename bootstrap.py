"""Assembly point: the only module that connects core/ to concrete adapters/."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from adapters.models.scripted import ScriptedClient
from adapters.research.github_research import GitHubCodeSearch, read_token
from adapters.sandbox.repair_tools import RepairTools
from adapters.sandbox.runner import DockerRunner
from core.ports import ChatClient, TraceRecorder
from core.repair_agent import RepairAgent
from core.repair_service import RepairService

PROJECT_ROOT = Path(__file__).resolve().parent
EVALS_ROOT = PROJECT_ROOT / "evals"
PROVIDERS = ("scripted",)


@dataclass(frozen=True)
class ScriptedRepair:
    """Prepared answer cards for one supplied case: files to read and the fixed source."""

    test_file: str
    source_file: str
    fixed_source: str


# Answer cards written after reading each fixture. They are data, not repair logic.
SCRIPTED_REPAIRS: dict[str, ScriptedRepair] = {
    "arithmetic": ScriptedRepair(
        test_file="test_calculator.py",
        source_file="calculator.py",
        fixed_source='''\
def add(left, right):
    """Return the sum of two numbers."""
    return left + right
''',
    ),
    "balanced_brackets": ScriptedRepair(
        test_file="test_brackets.py",
        source_file="brackets.py",
        fixed_source='''\
def is_balanced(text):
    """Return whether parentheses, brackets, and braces are properly nested."""
    opening = "([{"
    closing = ")]}"
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []

    for character in text:
        if character in opening:
            stack.append(character)
        elif character in closing:
            if not stack:
                return False
            if stack.pop() != pairs[character]:
                return False

    return not stack
''',
    ),
    "median": ScriptedRepair(
        test_file="test_statistics_tools.py",
        source_file="statistics_tools.py",
        fixed_source='''\
def median(values):
    """Return the median value without changing the input sequence."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2
    return ordered[middle]
''',
    ),
    "multi_function": ScriptedRepair(
        test_file="test_text_tools.py",
        source_file="text_tools.py",
        fixed_source='''\
def normalize_name(name):
    """Trim surrounding whitespace and lowercase a supplied name."""
    return name.strip().lower()


def greeting(name):
    """Create a greeting from a normalized name."""
    return f"Hello, {normalize_name(name).title()}!"


def initials(name):
    """Return uppercase initials for the whitespace-separated parts of a name."""
    return "".join(part[0].upper() for part in name.split())
''',
    ),
    "unique_words": ScriptedRepair(
        test_file="test_word_tools.py",
        source_file="word_tools.py",
        fixed_source='''\
def unique_words(text):
    """Return lowercase words once, in the order first seen."""
    return list(dict.fromkeys(text.lower().split()))
''',
    ),
}


def scripted_replies(case_name: str) -> list[str]:
    """Ordered replies: read test, read source, failing test, edit, passing test."""
    if case_name not in SCRIPTED_REPAIRS:
        raise ValueError(f"No scripted answer cards for project {case_name!r}.")
    plan = SCRIPTED_REPAIRS[case_name]
    cards = [
        {"tool": "read_file", "path": plan.test_file, "summary": "Read the test"},
        {"tool": "read_file", "path": plan.source_file, "summary": "Read the source"},
        {"tool": "run_tests", "summary": "Reproduce the failure"},
        {"tool": "edit_file", "path": plan.source_file, "content": plan.fixed_source,
         "summary": f"Fix {plan.source_file}"},
        {"tool": "run_tests", "summary": "Verify the repair"},
    ]
    return [json.dumps(card) for card in cards]


def make_client(provider: str, case_name: str) -> ChatClient:
    """Create the chat client for the selected provider and case."""
    if provider == "scripted":
        return ScriptedClient(scripted_replies(case_name))
    raise ValueError(f"Unknown provider {provider!r}. Choose one of: {', '.join(PROVIDERS)}")


def build_service(
    provider: str = "scripted",
    runner: DockerRunner | None = None,
    research: bool = False,
    environ: Mapping[str, str] | None = None,
) -> RepairService:
    """Connect the Docker runner, safe tools, chat client, agent, and service.

    Research stays off unless requested; only then is GITHUB_TOKEN read and the
    GitHub adapter constructed, fresh for each repair so its budgets reset.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}. Choose one of: {', '.join(PROVIDERS)}")
    docker_runner = runner or DockerRunner()
    token = read_token(os.environ if environ is None else environ) if research else None

    def make_agent(case_name: str, workspace: Path, recorder: TraceRecorder | None) -> RepairAgent:
        tools = RepairTools(workspace, docker_runner)
        research_port = GitHubCodeSearch(token) if token is not None else None
        return RepairAgent(
            make_client(provider, case_name), tools, recorder=recorder, research=research_port
        )

    return RepairService(EVALS_ROOT, docker_runner.create_workspace, make_agent)
