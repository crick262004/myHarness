import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from harness.mcp_client import MCPManager
from harness.tools import ToolRegistry

FIXTURE = Path(__file__).parent.parent / "fixtures" / "echo_server.py"


@asynccontextmanager
async def fixture_server(tmp_path: Path, **options) -> AsyncIterator[tuple[MCPManager, ToolRegistry]]:
    """Connect to the fixture server over stdio. Opened and closed in the same
    task, as the MCP transports require (so not a pytest fixture)."""
    config = tmp_path / "mcp_config.json"
    config.write_text(json.dumps({"mcpServers": {"fixture": {
        "command": sys.executable, "args": [str(FIXTURE)], **options,
    }}}))
    manager, registry = MCPManager(), ToolRegistry()
    await manager.connect_all(config, registry)
    try:
        assert manager.failures == {}
        yield manager, registry
    finally:
        await manager.aclose()


async def test_bridged_tools_run_through_the_registry(tmp_path):
    async with fixture_server(tmp_path) as (_, registry):
        assert registry.names() == ["mcp__fixture__echo", "mcp__fixture__shout"]
        assert await registry.execute("mcp__fixture__echo", {"text": "hi"}) == ("echo: hi", False)


async def test_read_only_hints_are_ignored_unless_the_server_is_trusted(tmp_path):
    async with fixture_server(tmp_path) as (manager, registry):
        echo = registry.get("mcp__fixture__echo")
        # The server marks echo read-only, but nobody said to believe it.
        assert not echo.read_only and not echo.concurrency_safe
        assert manager.status_lines() == ["(mcp) fixture: 2 tools bridged (0 read-only)"]


async def test_a_trusted_servers_hints_count(tmp_path):
    async with fixture_server(tmp_path, trustReadOnlyHints=True) as (manager, registry):
        echo, shout = registry.get("mcp__fixture__echo"), registry.get("mcp__fixture__shout")
        assert echo.read_only and echo.concurrency_safe
        assert not shout.read_only and not shout.concurrency_safe  # no hint: trust changes nothing
        assert manager.status_lines() == ["(mcp) fixture: 2 tools bridged (1 read-only)"]
