import sys
from pathlib import Path

import pytest

from copilot.tools.mcp_client import MCPToolClient, ToolError, open_github_tools

FAKE_SERVER = str(Path(__file__).parent / "fake_mcp_server.py")


@pytest.fixture
def fake_server_cmd(tmp_path):
    return sys.executable, [FAKE_SERVER], {"FAKE_MCP_LOG": str(tmp_path / "calls.jsonl")}


@pytest.fixture
def fake_client(fake_server_cmd):
    with MCPToolClient(*fake_server_cmd, timeout_s=10) as client:
        yield client


def test_lists_tools_and_calls_one(fake_client):
    assert {"issue_read", "pull_request_read", "add_issue_comment"} <= fake_client.tool_names()
    pr = fake_client.call("pull_request_read", {"method": "get", "owner": "o", "repo": "r", "pullNumber": 7})
    assert pr["number"] == 7 and pr["merged"] is False and pr["user"]["login"] == "Kludex"


def test_several_calls_reuse_one_session(fake_client):
    for n in (1, 2, 3):
        assert fake_client.call("issue_read", {"method": "get", "owner": "o", "repo": "r", "issue_number": n})["number"] == n


def test_tool_error_raises(fake_client):
    # MCPServer hides exception details from clients; we guarantee a ToolError carrying the server's text.
    with pytest.raises(ToolError, match="issue_read returned an error: Error executing tool issue_read"):
        fake_client.call("issue_read", {"method": "get", "owner": "o", "repo": "r", "issue_number": 404})


def test_client_still_usable_after_tool_error(fake_client):
    with pytest.raises(ToolError):
        fake_client.call("issue_read", {"method": "get", "owner": "o", "repo": "r", "issue_number": 404})
    assert fake_client.call("issue_read", {"method": "get", "owner": "o", "repo": "r", "issue_number": 1})["number"] == 1


def test_timeout_raises_tool_error(fake_server_cmd):
    with MCPToolClient(*fake_server_cmd, timeout_s=0.5) as client:
        with pytest.raises(ToolError, match="timed out"):
            client.call("slow_tool", {})


def test_unstartable_server_raises_tool_error():
    with pytest.raises(ToolError):
        with MCPToolClient("definitely-not-a-command-xyz", [], {}, timeout_s=2):
            pass


def test_open_github_tools_read_client_is_read_only(settings):
    settings.github_token = "t"
    client = open_github_tools(settings)  # not entered: no subprocess started
    assert client.command == "docker"
    assert "GITHUB_READ_ONLY=1" in client.args
    assert "GITHUB_TOOLS=issue_read,pull_request_read" in client.args
    assert client.args[-2:] == [settings.mcp_image, "stdio"]
    assert client.env == {"GITHUB_PERSONAL_ACCESS_TOKEN": "t"}  # token via env, never in argv


def test_open_github_tools_write_client(settings):
    settings.github_write_token = "w"
    client = open_github_tools(settings, write=True)
    assert "GITHUB_READ_ONLY=1" not in client.args
    assert "GITHUB_TOOLS=add_issue_comment,issue_write" in client.args
    assert client.env == {"GITHUB_PERSONAL_ACCESS_TOKEN": "w"}


def test_missing_tokens_raise(settings):
    settings.github_token = settings.github_write_token = ""
    with pytest.raises(ToolError):
        open_github_tools(settings)
    with pytest.raises(ToolError):
        open_github_tools(settings, write=True)


def test_new_settings_defaults(settings):
    assert settings.mcp_image == "ghcr.io/github/github-mcp-server:v2.0.2"
    assert settings.mcp_timeout_s == 15.0 and settings.sandbox_repo == "" and settings.api_keys == {}


def test_reconnects_after_the_server_dies(fake_server_cmd):
    # Review #5: if the container died, every later call failed until a server restart.
    with MCPToolClient(*fake_server_cmd, timeout_s=10) as client:
        with pytest.raises(ToolError) as err:
            client.call("crash_tool", {})
        assert err.value.uncertain  # a dropped connection can't tell us what happened
        pr = client.call("pull_request_read", {"method": "get", "owner": "o", "repo": "r", "pullNumber": 3})
        assert pr["number"] == 3  # transparently reopened


def test_plain_text_reply_is_returned_as_text(fake_client):
    assert fake_client.call("plain_text_tool", {}) == "Not Found"


def test_concurrent_calls_from_threads_share_one_client(fake_client):
    from concurrent.futures import ThreadPoolExecutor
    args = [{"method": "get", "owner": "o", "repo": "r", "issue_number": n} for n in range(1, 9)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        numbers = list(pool.map(lambda a: fake_client.call("issue_read", a)["number"], args))
    assert numbers == list(range(1, 9))
