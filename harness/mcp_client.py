"""Phase 3: MCP — plugging other people's tool servers into our registry.

MCP (Model Context Protocol) is a standard so a tool provider ships ONE
server and any AI app that speaks the protocol can use it. Our harness is
the *client*. Two transports, same protocol:

  - "command" entries → we SPAWN the server as a subprocess and speak
    JSON-RPC over its stdin/stdout ("stdio").
  - "url" entries → the server is HOSTED somewhere; we speak the same
    JSON-RPC over HTTP ("streamable http"). No local install needed.

Either way the handshake is: connect → initialize() → list_tools(), and each
advertised tool gets wrapped as one of OUR Tool objects in the SAME registry.
The agent loop never changes — MCP is just another tool supplier.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.client.streamable_http import streamable_http_client

from .tools import Tool, ToolError, ToolRegistry

CONNECT_TIMEOUT_SECONDS = 30


def expand_env(value: str) -> str:
    """${VAR} substitution, so secrets live in .env — never in the committed config."""
    expanded = os.path.expandvars(value)
    if "${" in expanded:
        raise ValueError(f"unresolved environment variable in config value: {value}")
    return expanded


def _redact(text: str, spec: dict[str, Any]) -> str:
    """Blank out any secret the config pulled from the environment.

    Connection errors often embed the full URL — including an API key
    substituted into it — and failure reasons get printed to the console.
    """
    for raw in [spec.get("url", ""), *(spec.get("env") or {}).values()]:
        for var in re.findall(r"\$\{(\w+)\}", raw):
            value = os.environ.get(var)
            if value:
                text = text.replace(value, "***")
    return text


def bridge_tool(
    server_name: str,
    tool_name: str,
    description: str | None,
    input_schema: dict[str, Any],
    session: ClientSession,
    read_only: bool = False,
) -> Tool:
    """Wrap one MCP tool as a native Tool — the agent can't tell the difference.

    Namespacing (mcp__server__tool) keeps two servers' tools from colliding
    with each other or with built-ins. A read-only tool (only reads, from a
    trusted server's hint) may run in parallel and be given to sub-agents.
    """

    async def handler(tool_input: dict[str, Any]) -> str:
        result = await session.call_tool(tool_name, tool_input)
        parts: list[str] = []
        for item in result.content or []:
            if getattr(item, "type", None) == "text":
                parts.append(item.text)
            else:
                parts.append(f"[{getattr(item, 'type', 'unknown')} content omitted]")
        if not parts and result.structured_content is not None:
            parts.append(json.dumps(result.structured_content, ensure_ascii=False))
        text = "\n".join(parts) or "(no content)"
        if result.is_error:
            raise ToolError(text)  # becomes an is_error tool_result, like any local failure
        return text

    return Tool(
        name=f"mcp__{server_name}__{tool_name}",
        description=description or f"Tool '{tool_name}' from the '{server_name}' MCP server.",
        input_schema=input_schema,
        handler=handler,
        concurrency_safe=read_only,
        read_only=read_only,
    )


class MCPManager:
    """Connects to every configured server and owns those connections."""

    def __init__(self) -> None:
        self._stacks: dict[str, AsyncExitStack] = {}
        self.tool_counts: dict[str, int] = {}   # server -> number of tools bridged
        self.read_only_counts: dict[str, int] = {}  # server -> how many count as read-only
        self.failures: dict[str, str] = {}      # server -> why it didn't connect

    async def connect_all(self, config_path: Path, registry: ToolRegistry) -> None:
        """Best-effort: a server that fails to start must not kill the harness."""
        if not config_path.is_file():
            return
        config = json.loads(config_path.read_text())
        for name, spec in config.get("mcpServers", {}).items():
            try:
                # asyncio.timeout, NOT wait_for: wait_for runs the coroutine in
                # a separate task, but the transports' anyio cancel scopes must
                # be entered and exited from the SAME task (see aclose).
                async with asyncio.timeout(CONNECT_TIMEOUT_SECONDS):
                    await self._connect_one(name, spec, registry)
            except Exception as exc:  # noqa: BLE001 — record and move on
                reason = str(exc) or type(exc).__name__
                self.failures[name] = f"{type(exc).__name__}: {_redact(reason, spec)[:200]}"

    async def _connect_one(self, name: str, spec: dict[str, Any], registry: ToolRegistry) -> None:
        stack = AsyncExitStack()
        try:
            if "url" in spec:
                read, write = await stack.enter_async_context(
                    streamable_http_client(expand_env(spec["url"]))
                )
            elif "command" in spec:
                env = get_default_environment()  # PATH, HOME, ... for the child process
                for key, value in (spec.get("env") or {}).items():
                    env[key] = expand_env(value)
                params = StdioServerParameters(
                    command=spec["command"], args=spec.get("args", []), env=env
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            else:
                raise ValueError("server entry needs either 'url' or 'command'")

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listing = await session.list_tools()
            # Servers can self-declare read-only tools (an MCP annotation).
            # That's only the server's claim, and the MCP spec says not to
            # act on hints from untrusted servers. So it counts only if this
            # server's config entry says "trustReadOnlyHints": true. Even
            # then it never bypasses the permission policy.
            trusted = spec.get("trustReadOnlyHints", False) is True
            read_only = 0
            for mcp_tool in listing.tools:
                annotations = mcp_tool.annotations
                hint = trusted and bool(annotations and annotations.read_only_hint)
                read_only += hint
                registry.register(
                    bridge_tool(
                        name, mcp_tool.name, mcp_tool.description,
                        mcp_tool.input_schema, session, read_only=hint,
                    )
                )
            self.tool_counts[name] = len(listing.tools)
            self.read_only_counts[name] = read_only
        except BaseException:
            await stack.aclose()  # tear down whatever half-connected
            raise
        self._stacks[name] = stack

    def status_lines(self) -> list[str]:
        lines = [
            f"(mcp) {name}: {count} tool{'s' if count != 1 else ''} bridged"
            f" ({self.read_only_counts.get(name, 0)} read-only)"
            for name, count in self.tool_counts.items()
        ]
        lines += [f"(mcp) {name}: FAILED — {reason}" for name, reason in self.failures.items()]
        return lines

    async def aclose(self) -> None:
        # LIFO: anyio cancel scopes must unwind innermost-first, so the last
        # server connected is the first one closed.
        for stack in reversed(self._stacks.values()):
            try:
                await stack.aclose()
            except Exception:  # noqa: BLE001 — a dead server must not break shutdown
                pass
        self._stacks.clear()
