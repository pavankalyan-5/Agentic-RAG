"""A tiny stand-in for GitHub's MCP server, run as a stdio subprocess in tests.

Same tool names and argument shapes as ghcr.io/github/github-mcp-server, canned data, no network.
Write tools append each call to the JSONL file named by FAKE_MCP_LOG so tests can count executions.
"""

import json
import os

import anyio
from mcp.server import MCPServer

server = MCPServer("fake-github")


def _log(tool: str, args: dict) -> None:
    path = os.environ.get("FAKE_MCP_LOG")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"tool": tool, "args": args}) + "\n")


@server.tool()
def issue_read(method: str, owner: str, repo: str, issue_number: int) -> str:
    if issue_number == 404:
        raise ValueError("issue not found")
    return json.dumps({
        "number": issue_number, "title": f"Issue {issue_number}", "state": "open",
        "html_url": f"https://github.com/{owner}/{repo}/issues/{issue_number}",
        "user": {"login": "alice"}, "labels": [{"name": "bug"}], "comments": 2,
        "updated_at": "2026-10-18T10:00:00Z", "body": "Something breaks.",
    })


@server.tool()
def pull_request_read(method: str, owner: str, repo: str, pullNumber: int) -> str:
    if pullNumber == 404:
        raise ValueError("pull request not found")
    merged = pullNumber % 2 == 0  # even numbers are merged
    return json.dumps({
        "number": pullNumber, "title": f"PR {pullNumber}", "state": "closed",
        "merged": merged, "merged_at": "2026-10-10T00:00:00Z" if merged else None,
        "html_url": f"https://github.com/{owner}/{repo}/pull/{pullNumber}",
        "user": {"login": "Kludex"}, "labels": [], "comments": 1,
        "updated_at": "2026-10-18T10:00:00Z", "body": "Preserve response trailers.",
    })


@server.tool()
def add_issue_comment(owner: str, repo: str, issue_number: int, body: str) -> str:
    _log("add_issue_comment", {"owner": owner, "repo": repo, "issue_number": issue_number, "body": body})
    # Same shape as GitHub MCP v2.0.2: MinimalResponse {"id", "url"}
    return json.dumps({"id": "1", "url": f"https://github.com/{owner}/{repo}/issues/{issue_number}#issuecomment-1"})


@server.tool()
def issue_write(method: str, owner: str, repo: str, title: str, body: str = "") -> str:
    _log("issue_write", {"method": method, "owner": owner, "repo": repo, "title": title, "body": body})
    return json.dumps({"id": "99", "url": f"https://github.com/{owner}/{repo}/issues/99"})


@server.tool()
def crash_tool() -> str:
    """Kill the server process, like a container dying mid-session."""
    os._exit(1)


@server.tool()
def plain_text_tool() -> str:
    return "Not Found"


@server.tool()
async def slow_tool() -> str:
    await anyio.sleep(5)
    return "done"


if __name__ == "__main__":
    server.run()
