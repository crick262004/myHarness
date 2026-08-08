"""A tiny MCP server used as a test fixture.

Speaks MCP over stdio, exposing one tool. Lets us test the whole bridge
(spawn → initialize → list_tools → call_tool) offline, with no Node and no
network. Run directly:  python evals/fixtures/echo_server.py
"""

from mcp.server import MCPServer

server = MCPServer("fixture")


@server.tool()
def echo(text: str) -> str:
    """Echo the given text back, prefixed with 'echo: '."""
    return f"echo: {text}"


if __name__ == "__main__":
    server.run()  # stdio transport by default
