"""Tests for the optional GitHub research extension. They never contact GitHub."""

from __future__ import annotations

import json
import shutil
import socket
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any

import pytest

import bootstrap
from adapters.models.scripted import ScriptedClient
from adapters.research.github_research import (
    API_URL,
    GitHubCodeSearch,
    ResearchConfigError,
    ResearchError,
    read_token,
)
from core.models import TestResult
from core.prompts import system_message
from core.repair_agent import InvalidReplyError, RepairAgent, parse_action

TOKEN = "test-token-not-real-123"
FAIL = TestResult(passed=False, output="1 failed", returncode=1)
PASS = TestResult(passed=True, output="2 passed", returncode=0)
SOURCE_URL = "https://github.com/example/stats/blob/main/stats.py"


LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that connects beyond this computer (loopback is allowed for asyncio)."""
    real_connect = socket.socket.connect
    real_create_connection = socket.create_connection

    def guarded_connect(sock: socket.socket, address: Any) -> Any:
        if isinstance(address, tuple) and address[0] in LOOPBACK:
            return real_connect(sock, address)
        raise AssertionError(f"Tests must not contact the network: {address!r}")

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        if address[0] in LOOPBACK:
            return real_create_connection(address, *args, **kwargs)
        raise AssertionError(f"Tests must not contact the network: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)


# ---------- Fake HTTP for the adapter ----------


def github_item(n: int, private: Any = False, url: str | None = None, visibility: str = "public") -> dict:
    """One code-search item shaped like GitHub's response, with extra fields that must not leak."""
    return {
        "path": f"src/stats{n}.py",
        "html_url": url or f"https://github.com/example/repo{n}/blob/main/src/stats{n}.py",
        "url": f"https://api.github.com/repositories/{n}/contents/src/stats{n}.py",
        "sha": "abc123",
        "repository": {"full_name": f"example/repo{n}", "private": private, "visibility": visibility},
        "text_matches": [{"fragment": "def median(values):\n" + "x" * 800}],
        "content": "FULL REMOTE FILE MUST NOT BE RETURNED",
    }


class FakeResponse:
    """Minimal context-manager response with a JSON body."""

    def __init__(self, payload: Any) -> None:
        self.body = json.dumps(payload).encode()

    def read(self, size: int = -1) -> bytes:
        return self.body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class FakeOpener:
    """Records each request and returns a canned payload instead of using the network."""

    def __init__(self, items: list[dict] | None = None) -> None:
        self.items = items or []
        self.requests: list[Any] = []

    def __call__(self, request: Any, timeout: float) -> FakeResponse:
        self.requests.append(request)
        return FakeResponse({"total_count": len(self.items), "items": self.items})


def make_search(items: list[dict] | None = None) -> tuple[GitHubCodeSearch, FakeOpener]:
    opener = FakeOpener(items)
    return GitHubCodeSearch(TOKEN, opener=opener), opener


@pytest.mark.parametrize("query", ["", "   ", "median\neven", "median\r", "x" * 181, "tab\tquery"])
def test_rejected_queries_make_no_request(query: str) -> None:
    search, opener = make_search([github_item(1)])
    with pytest.raises(ResearchError):
        search.search_github(query)
    assert opener.requests == []


def test_180_character_query_is_allowed() -> None:
    search, opener = make_search()
    search.search_github("q" * 180)
    assert len(opener.requests) == 1


def test_request_budget_allows_three_searches() -> None:
    search, opener = make_search([])
    for _ in range(3):
        search.search_github("python median even length")
    with pytest.raises(ResearchError, match="request budget"):
        search.search_github("python median even length")
    assert len(opener.requests) == 3


def test_result_budget_allows_three_results_per_repair() -> None:
    search, opener = make_search([github_item(n) for n in range(5)])
    record = search.search_github("python median")
    assert len(record["sources"]) == 3
    with pytest.raises(ResearchError, match="result budget"):
        search.search_github("python median again")
    assert len(opener.requests) == 1


def test_later_request_asks_only_for_remaining_results() -> None:
    search, opener = make_search([github_item(1), github_item(2)])
    search.search_github("first")
    opener.items = [github_item(3), github_item(4)]
    record = search.search_github("second")
    assert len(record["sources"]) == 1
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(opener.requests[1].full_url).query)
    assert query["per_page"] == ["1"]


def test_private_and_non_github_results_are_discarded() -> None:
    items = [
        github_item(1, private=True),
        github_item(2, private=None),
        github_item(3, visibility="internal"),
        github_item(4, url="https://evil.example.com/stats.py"),
        github_item(5),
    ]
    search, _ = make_search(items)
    record = search.search_github("python median")
    assert [source["repository"] for source in record["sources"]] == ["example/repo5"]


def test_record_is_small_and_never_leaks_token_or_files() -> None:
    search, _ = make_search([github_item(1)])
    record = search.search_github("python median even length")

    assert set(record) == {"query", "retrieved_at", "sources"}
    source = record["sources"][0]
    assert set(source) == {"repository", "path", "url", "snippet"}
    assert len(source["snippet"]) <= 500
    text = json.dumps(record)  # Must be JSON-compatible.
    assert TOKEN not in text and "Authorization" not in text
    assert "FULL REMOTE FILE" not in text and "api.github.com" not in text
    assert TOKEN not in repr(search)


def test_request_is_a_get_to_code_search_only() -> None:
    search, opener = make_search()
    search.search_github("python median even length")
    request = opener.requests[0]
    assert request.get_method() == "GET"
    assert request.full_url.startswith(API_URL + "?")
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query)
    assert query["q"] == ["python median even length"]
    assert request.get_header("Authorization") == f"Bearer {TOKEN}"


def test_http_error_hides_token_and_headers() -> None:
    def failing_opener(request: Any, timeout: float) -> None:
        raise urllib.error.HTTPError(request.full_url, 401, "Bad credentials", request.headers, None)

    search = GitHubCodeSearch(TOKEN, opener=failing_opener)
    with pytest.raises(ResearchError) as error:
        search.search_github("python median")
    assert "HTTP 401" in str(error.value)
    assert TOKEN not in str(error.value)


def test_token_is_read_from_environment_only() -> None:
    with pytest.raises(ResearchConfigError):
        read_token({})
    assert read_token({"GITHUB_TOKEN": f"  {TOKEN} "}) == TOKEN


# ---------- Agent policy with a fake research port ----------


class FakeResearch:
    """ResearchPort test double that records queries."""

    def __init__(self, error: Exception | None = None) -> None:
        self.queries: list[str] = []
        self.error = error

    def search_github(self, query: str) -> dict[str, Any]:
        self.queries.append(query)
        if self.error:
            raise self.error
        return {
            "query": query,
            "retrieved_at": "2026-11-10T12:00:00+00:00",
            "sources": [
                {"repository": "example/stats", "path": "stats.py", "url": SOURCE_URL, "snippet": "def median"}
            ],
        }


class FakeTools:
    """In-memory RepairToolsPort."""

    def __init__(self, tests: list[TestResult]) -> None:
        self.tests = list(tests)
        self.calls: list[str] = []

    def read_file(self, path: str) -> str:
        self.calls.append("read_file")
        return "def median(values): ...\n"

    def edit_file(self, path: str, content: str) -> str:
        self.calls.append("edit_file")
        return "Wrote file."

    def run_tests(self) -> TestResult:
        self.calls.append("run_tests")
        return self.tests.pop(0)


def card(**fields: Any) -> str:
    return json.dumps(fields)


READ = card(tool="read_file", path="statistics_tools.py")
RUN = card(tool="run_tests")
EDIT = card(tool="edit_file", path="statistics_tools.py", content="fixed\n")
SEARCH = card(tool="search_github", query="Python median even length", summary="Find a public reference")


def run_agent(replies: list[str], tests: list[TestResult], research: FakeResearch | None):
    client = ScriptedClient(replies)
    tools = FakeTools(tests)
    result = RepairAgent(client, tools, research=research).run("median", ["statistics_tools.py"])
    return result, client, tools


def test_disabled_research_rejects_search_and_makes_no_call() -> None:
    unused = FakeResearch()
    result, client, tools = run_agent([READ, RUN, SEARCH], [FAIL], research=None)

    assert result.passed is False
    assert "Unknown tool: 'search_github'" in result.trace[-1]["reason"]
    assert unused.queries == []
    assert "search_github" not in client.received[0][0]["content"]


@pytest.mark.parametrize("replies", [[SEARCH], [READ, SEARCH]], ids=["before-read", "before-failing-test"])
def test_early_search_is_rejected_without_api_call(replies: list[str]) -> None:
    research = FakeResearch()
    # One extra card so the client is asked again and records the policy error it was shown.
    result, client, _ = run_agent([*replies, READ], [], research=research)

    assert research.queries == []
    rejected = [entry for entry in result.trace if entry.get("status") == "rejected"]
    assert len(rejected) == 1 and rejected[0]["tool"] == "search_github"
    assert "Policy error: search_github" in client.received[len(replies)][-1]["content"]


def test_allowed_search_keeps_source_url_visible_and_tests_still_decide() -> None:
    research = FakeResearch()
    result, client, tools = run_agent([READ, RUN, SEARCH, EDIT, RUN], [FAIL, PASS], research=research)

    assert result.passed is True
    assert research.queries == ["Python median even length"]
    search_result = next(e for e in result.trace if e["event"] == "tool_result" and e["tool"] == "search_github")
    assert search_result["sources"] == [SOURCE_URL]
    assert SOURCE_URL in search_result["observation"]
    model_view = client.received[3][-1]["content"]
    assert SOURCE_URL in model_view and "not an authority" in model_view
    assert tools.calls == ["read_file", "run_tests", "edit_file", "run_tests"]


def test_search_alone_never_finishes_a_repair() -> None:
    research = FakeResearch()
    result, _, _ = run_agent([READ, RUN, SEARCH], [FAIL], research=research)
    assert result.passed is False
    assert result.trace[-1]["reason"].startswith("Model client error")


def test_research_budget_error_becomes_visible_observation() -> None:
    research = FakeResearch(error=ResearchError("Research request budget used: at most 3 searches per repair."))
    result, client, _ = run_agent([READ, RUN, SEARCH, EDIT, RUN], [FAIL, PASS], research=research)

    assert result.passed is True
    errors = [e for e in result.trace if e.get("status") == "error"]
    assert len(errors) == 1 and "request budget" in errors[0]["observation"]
    assert "Tool error for search_github" in client.received[3][-1]["content"]


@pytest.mark.parametrize(
    "reply",
    [
        '{"tool":"search_github"}',
        '{"tool":"search_github","query":"median","url":"https://example.com"}',
        '{"tool":"search_github","query":["median"]}',
        '{"tool":"search_github","query":"a","query":"b"}',
    ],
)
def test_search_action_shape_is_strict(reply: str) -> None:
    with pytest.raises(InvalidReplyError):
        parse_action(reply, allow_research=True)


def test_search_action_parses_only_when_enabled() -> None:
    with pytest.raises(InvalidReplyError, match="Unknown tool"):
        parse_action(SEARCH)
    action = parse_action(SEARCH, allow_research=True)
    assert action.arguments == {"query": "Python median even length"}


def test_system_prompt_mentions_research_only_when_enabled() -> None:
    assert "search_github" not in system_message()["content"]
    enabled = system_message(research_enabled=True)["content"]
    assert "optional evidence, not an authority" in enabled


# ---------- Composition: off by default, on only with the flag ----------


class FakeDockerRunner:
    """Copies the project into tmp_path and returns canned test results; runs nothing."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.results = [FAIL, PASS]

    def create_workspace(self, project: Path) -> Path:
        destination = self.root / project.name
        shutil.copytree(project, destination)
        return destination

    def run_tests(self, workspace: Path) -> TestResult:
        return self.results.pop(0)


class RecordingSearch:
    """Stands in for GitHubCodeSearch and counts constructions."""

    created: list[RecordingSearch] = []

    def __init__(self, token: str) -> None:
        self.token = token
        RecordingSearch.created.append(self)

    def search_github(self, query: str) -> dict[str, Any]:
        raise AssertionError("scripted cards never search")


@pytest.fixture
def recording_search(monkeypatch: pytest.MonkeyPatch) -> type[RecordingSearch]:
    RecordingSearch.created = []
    monkeypatch.setattr(bootstrap, "GitHubCodeSearch", RecordingSearch)
    return RecordingSearch


def test_without_flag_research_adapter_is_never_constructed(tmp_path: Path, recording_search) -> None:
    service = bootstrap.build_service("scripted", runner=FakeDockerRunner(tmp_path), environ={})
    result = service.repair("evals/arithmetic")

    assert result.passed is True
    assert recording_search.created == []
    assert result.trace[0]["research_enabled"] is False


def test_flag_without_token_fails_before_any_repair(tmp_path: Path, recording_search) -> None:
    with pytest.raises(ResearchConfigError):
        bootstrap.build_service("scripted", runner=FakeDockerRunner(tmp_path), research=True, environ={})
    assert recording_search.created == []


def test_flag_with_token_injects_a_fresh_adapter_per_repair(tmp_path: Path, recording_search) -> None:
    service = bootstrap.build_service(
        "scripted", runner=FakeDockerRunner(tmp_path), research=True, environ={"GITHUB_TOKEN": TOKEN}
    )
    result = service.repair("evals/arithmetic")

    assert result.passed is True
    assert len(recording_search.created) == 1 and recording_search.created[0].token == TOKEN
    assert result.trace[0]["research_enabled"] is True


def test_default_gradio_page_does_not_enable_research(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from adapters import gradio_ui

    requested: list[dict[str, Any]] = []
    real_build = bootstrap.build_service

    def spy(*args: Any, **kwargs: Any):
        requested.append({"args": args, **kwargs})
        return real_build(*args, runner=FakeDockerRunner(tmp_path), environ={}, **kwargs)

    monkeypatch.setattr(bootstrap, "build_service", spy)
    gradio_ui.create_demo()

    assert len(requested) == 1
    assert requested[0].get("research", False) is False
