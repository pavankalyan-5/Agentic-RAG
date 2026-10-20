"""MCP tool client with a synchronous facade.

The MCP SDK is async; LangGraph nodes here are sync. An anyio *blocking portal* runs an event loop
in a background thread and lets sync code call into it, so one long-lived MCP session (one Docker
container) serves many tool calls.
"""

import json
import os
import threading
from contextlib import ExitStack, asynccontextmanager
from typing import Protocol

import anyio
from anyio.from_thread import start_blocking_portal
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

from copilot.config import Settings

READ_TOOLS = ("issue_read", "pull_request_read")
WRITE_TOOLS = ("add_issue_comment", "issue_write")


class ToolError(Exception):
    """The tool server couldn't be reached, timed out, or the tool returned an error.

    uncertain=True means the tool MAY have run anyway (timeout, dropped connection): for a write,
    "failed" would be a lie, so callers record the outcome as unknown and reconcile later.
    """

    def __init__(self, message: str, uncertain: bool = False):
        super().__init__(message)
        self.uncertain = uncertain


class ToolClient(Protocol):
    def call(self, name: str, arguments: dict) -> dict | list | str: ...

    def tool_names(self) -> set[str]: ...


class MCPToolClient:
    def __init__(self, command: str, args: list[str], env: dict[str, str], timeout_s: float = 15.0):
        self.command, self.args, self.env, self.timeout_s = command, args, env, timeout_s
        self._stack: ExitStack | None = None
        self._portal = None
        self._session: ClientSession | None = None
        self._tools: set[str] = set()
        self._broken = False  # the transport died; reopen before the next call
        self._reconnect_lock = threading.Lock()

    def __enter__(self) -> "MCPToolClient":
        self._start()
        return self

    def _start(self) -> None:
        self._stack = ExitStack()
        try:
            self._portal = self._stack.enter_context(start_blocking_portal())
            self._session = self._stack.enter_context(self._portal.wrap_async_context_manager(self._open()))
            self._tools = self._portal.call(self._initialize)
            self._broken = False
        except Exception as exc:
            self._close()
            raise ToolError(f"could not start tool server {self.command!r}: {exc}", uncertain=True) from exc

    def __exit__(self, *exc_info) -> None:
        self._close()

    def tool_names(self) -> set[str]:
        return set(self._tools)

    def call(self, name: str, arguments: dict) -> dict | list | str:
        if self._broken:
            self._reconnect()

        async def _call():
            with anyio.fail_after(self.timeout_s):
                return await self._session.call_tool(name, arguments)

        try:
            result = self._portal.call(_call)
        except TimeoutError as exc:
            raise ToolError(f"{name} timed out after {self.timeout_s}s", uncertain=True) from exc
        except Exception as exc:
            # The session/transport is unusable (e.g. the container died): reopen on the next call.
            self._broken = True
            raise ToolError(f"{name} failed: {exc}", uncertain=True) from exc
        text = result.content[0].text if result.content else ""
        if result.is_error:  # MCP SDK v2 name (v1 was isError)
            raise ToolError(f"{name} returned an error: {text}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    @asynccontextmanager
    async def _open(self):
        # A provided env replaces the default one entirely, so keep PATH/HOME etc.
        params = StdioServerParameters(
            command=self.command, args=self.args, env={**get_default_environment(), **self.env}
        )
        with open(os.devnull, "w") as devnull:  # the server's INFO logs go to stderr; keep the CLI clean
            async with stdio_client(params, errlog=devnull) as (read, write), ClientSession(read, write) as session:
                yield session

    async def _initialize(self) -> set[str]:
        with anyio.fail_after(self.timeout_s):
            await self._session.initialize()
            return {t.name for t in (await self._session.list_tools()).tools}

    def _reconnect(self) -> None:
        with self._reconnect_lock:  # several worker threads may notice at once; reopen only once
            if not self._broken:
                return
            self._close()
            self._start()

    def _close(self) -> None:
        if self._stack is not None:
            try:
                self._stack.close()
            except Exception:
                pass  # the subprocess may already be gone; nothing useful to report on shutdown
            self._stack = None


def call_tool(tools: "ToolClient", name: str, arguments: dict):
    """Call a tool inside a TOOL span (works for any ToolClient: MCP, fakes, future backends).
    A ToolError propagating out marks the span ERROR; OpenTelemetry records the exception."""
    from copilot.observability.tracing import span

    with span(f"tool.{name}", "TOOL", tool__name=name, input__value=arguments) as s:
        result = tools.call(name, arguments)
        if s.is_recording():
            s.set_attribute("output.value", json.dumps(result, default=str)[:2000])
        return result


def open_github_tools(settings: Settings, *, write: bool = False) -> MCPToolClient:
    """Build (not start) a client for GitHub's MCP server in Docker.

    Read client: read-only mode + read tools only + the read token.
    Write client: two write tools + a token that should be scoped to the sandbox repo only.
    The token is passed through the environment, never on the command line (visible in `ps`).
    """
    token = settings.github_write_token if write else settings.github_token
    if not token:
        raise ToolError(f"no GitHub {'write ' if write else ''}token configured")
    tools = WRITE_TOOLS if write else READ_TOOLS
    args = ["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN", "-e", f"GITHUB_TOOLS={','.join(tools)}"]
    if not write:
        args += ["-e", "GITHUB_READ_ONLY=1"]
    args += [settings.mcp_image, "stdio"]
    return MCPToolClient("docker", args, {"GITHUB_PERSONAL_ACCESS_TOKEN": token}, settings.mcp_timeout_s)
