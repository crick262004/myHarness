"""A tiny MCP server used as a test fixture.

Speaks MCP over stdio, exposing two tools: echo, which the server marks
read-only, and shout, which it doesn't. Lets us test the whole bridge
(spawn → initialize → list_tools → call_tool) offline, with no Node and no
network. Run directly:  python evals/fixtures/echo_server.py
"""

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer("fixture")


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
def echo(text: str) -> str:
    """Echo the given text back, prefixed with 'echo: '."""
    return f"echo: {text}"


@server.tool()
def shout(text: str) -> str:
    """Echo the given text back in capitals."""
    return text.upper()


if __name__ == "__main__":
    server.run()  # stdio transport by default
