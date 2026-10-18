from datetime import datetime, timezone

import httpx
import pytest

from copilot.ingest import github_api
from copilot.ingest.github_api import GitHubClient

API = "https://api.github.com"
SINCE = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(github_api, "_sleep", lambda s: None)


def item(number, pr=False, body="b", merged=False, updated="2026-02-01T00:00:00Z"):
    d = {
        "number": number, "title": f"T{number}", "body": body, "state": "closed" if merged else "open",
        "html_url": f"https://github.com/o/r/{'pull' if pr else 'issues'}/{number}",
        "user": {"login": "alice"}, "labels": [{"name": "bug"}],
        "created_at": "2026-01-15T00:00:00Z", "updated_at": updated,
    }
    if pr:
        d["pull_request"] = {"merged_at": "2026-02-01T00:00:00Z" if merged else None}
    return d


def comment(body, at="2026-01-20T00:00:00Z"):
    return {"user": {"login": "bob"}, "body": body, "created_at": at}


def client():
    return GitHubClient("t", http=httpx.Client())


def test_yields_issue_and_pr_with_comments(respx_mock):
    respx_mock.get(f"{API}/repos/o/r/issues").respond(json=[item(1, body=None), item(2, pr=True, merged=True)])
    respx_mock.get(f"{API}/repos/o/r/issues/1/comments").respond(json=[comment("me too")])
    respx_mock.get(f"{API}/repos/o/r/issues/2/comments").respond(json=[comment("lgtm", "2026-01-22T00:00:00Z")])
    respx_mock.get(f"{API}/repos/o/r/pulls/2/comments").respond(json=[comment("nit", "2026-01-21T00:00:00Z")])

    docs = list(client().iter_threads("o/r", SINCE))

    assert [(d.doc_type, d.number) for d in docs] == [("issue", 1), ("pr", 2)]
    assert docs[0].body == "" and [p.body for p in docs[0].posts] == ["me too"]
    assert docs[0].source_id == "issue:1" and docs[0].labels == ["bug"]
    assert docs[1].source_id == "pr:2" and docs[1].state == "merged"
    assert [p.body for p in docs[1].posts] == ["nit", "lgtm"]  # sorted by created_at


def test_sends_since_and_auth(respx_mock):
    route = respx_mock.get(f"{API}/repos/o/r/issues").respond(json=[])
    list(client().iter_threads("o/r", SINCE))
    req = route.calls.last.request
    assert req.url.params["since"] == "2026-01-01T00:00:00Z"
    assert req.url.params["state"] == "all" and req.url.params["direction"] == "asc"
    assert req.headers["Authorization"] == "Bearer t"


def test_follows_link_pagination(respx_mock):
    page2 = f"{API}/repos/o/r/issues?page=2"
    respx_mock.get(page2).respond(json=[item(2)])
    respx_mock.get(f"{API}/repos/o/r/issues").respond(json=[item(1)], headers={"Link": f'<{page2}>; rel="next"'})
    respx_mock.get(url__regex=r".*/comments").respond(json=[])
    assert [d.number for d in client().iter_threads("o/r", SINCE)] == [1, 2]


def test_retries_on_429_then_succeeds(respx_mock):
    route = respx_mock.get(f"{API}/repos/o/r/issues")
    route.side_effect = [httpx.Response(429, headers={"Retry-After": "0"}), httpx.Response(200, json=[item(1)])]
    respx_mock.get(url__regex=r".*/comments").respond(json=[])
    assert len(list(client().iter_threads("o/r", SINCE))) == 1
    assert route.call_count == 2


def test_retries_on_403_rate_limit(respx_mock):
    route = respx_mock.get(f"{API}/repos/o/r/issues")
    route.side_effect = [
        httpx.Response(403, headers={"X-RateLimit-Remaining": "0"}),
        httpx.Response(200, json=[]),
    ]
    list(client().iter_threads("o/r", SINCE))
    assert route.call_count == 2


def test_does_not_retry_plain_403(respx_mock):
    route = respx_mock.get(f"{API}/repos/o/r/issues").respond(403)
    with pytest.raises(httpx.HTTPStatusError):
        list(client().iter_threads("o/r", SINCE))
    assert route.call_count == 1


def test_gives_up_after_5_attempts(respx_mock):
    route = respx_mock.get(f"{API}/repos/o/r/issues").respond(503)
    with pytest.raises(httpx.HTTPStatusError):
        list(client().iter_threads("o/r", SINCE))
    assert route.call_count == 5
