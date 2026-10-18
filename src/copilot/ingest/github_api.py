"""GitHub REST client for issues, PRs, and their comments."""

import random
import time
from collections.abc import Iterator
from datetime import datetime, timezone

import httpx

from copilot.ingest.models import Post, SourceDoc

MAX_ATTEMPTS = 5
MAX_WAIT_SECONDS = 900  # never sleep longer than 15 min for a rate-limit reset

_sleep = time.sleep  # patched in tests


class GitHubClient:
    def __init__(self, token: str, http: httpx.Client | None = None, base_url: str = "https://api.github.com"):
        self.base_url = base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=30)
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def iter_threads(self, repo: str, since: datetime) -> Iterator[SourceDoc]:
        params = {
            "state": "all",
            "since": since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sort": "updated",
            "direction": "asc",
            "per_page": 100,
        }
        # List every page FIRST (fast: 100 items/request), then fetch comments. Paging lazily while
        # spending minutes on comments lets items updated mid-crawl shift pages, so one gets skipped.
        latest: dict[int, dict] = {}
        for raw in self._paginate(f"{self.base_url}/repos/{repo}/issues", params):
            latest[raw["number"]] = raw  # an item that moved pages appears twice; keep the newest copy
        # Ascending updated_at order is what makes the pipeline's progress checkpoints safe.
        for raw in sorted(latest.values(), key=lambda r: r["updated_at"]):
            yield self._to_doc(repo, raw)

    def _to_doc(self, repo: str, raw: dict) -> SourceDoc:
        number = raw["number"]
        pr = raw.get("pull_request")
        posts = self._posts(f"{self.base_url}/repos/{repo}/issues/{number}/comments")
        if pr is not None:
            posts += self._posts(f"{self.base_url}/repos/{repo}/pulls/{number}/comments")
            posts.sort(key=lambda p: p.created_at or datetime.min.replace(tzinfo=timezone.utc))
        state = "merged" if pr and pr.get("merged_at") else raw.get("state")
        doc_type = "pr" if pr is not None else "issue"
        return SourceDoc(
            repo=repo,
            source_id=f"{doc_type}:{number}",
            doc_type=doc_type,
            number=number,
            title=raw.get("title") or "",
            url=raw["html_url"],
            author=(raw.get("user") or {}).get("login"),
            labels=[lbl["name"] for lbl in raw.get("labels", [])],
            state=state,
            created_at=_parse_ts(raw.get("created_at")),
            updated_at=_parse_ts(raw["updated_at"]),
            body=raw.get("body") or "",
            posts=posts,
        )

    def _posts(self, url: str) -> list[Post]:
        return [
            Post(
                author=(c.get("user") or {}).get("login"),
                body=c.get("body") or "",
                created_at=_parse_ts(c.get("created_at")),
            )
            for c in self._paginate(url, {"per_page": 100})
        ]

    def _paginate(self, url: str, params: dict | None) -> Iterator[dict]:
        next_url: str | None = url
        while next_url:
            resp = self._get(next_url, params)
            yield from resp.json()
            next_url = resp.links.get("next", {}).get("url")
            params = None  # the "next" URL already carries the query string

    def _get(self, url: str, params: dict | None) -> httpx.Response:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            resp = self.http.get(url, params=params, headers=self.headers)
            if not _is_retryable(resp) or attempt == MAX_ATTEMPTS:
                resp.raise_for_status()
                return resp
            _sleep(_wait_seconds(resp, attempt))
        raise AssertionError("unreachable")


def _is_retryable(resp: httpx.Response) -> bool:
    if resp.status_code == 429 or resp.status_code >= 500:
        return True
    if resp.status_code != 403:
        return False
    # Primary limit: X-RateLimit-Remaining: 0. Secondary ("abuse") limit: Retry-After and/or a message.
    # A plain 403 (no permission) is not retried — waiting won't help.
    return (
        resp.headers.get("X-RateLimit-Remaining") == "0"
        or "Retry-After" in resp.headers
        or "rate limit" in resp.text.lower()
    )


def _wait_seconds(resp: httpx.Response, attempt: int) -> float:
    if "Retry-After" in resp.headers:
        return min(float(resp.headers["Retry-After"]), MAX_WAIT_SECONDS)
    if "X-RateLimit-Reset" in resp.headers:
        return min(max(0.0, float(resp.headers["X-RateLimit-Reset"]) - time.time()), MAX_WAIT_SECONDS)
    # Exponential backoff with full jitter: random wait in [0, 2^attempt] seconds.
    return random.uniform(0, 2**attempt)


def _parse_ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None
