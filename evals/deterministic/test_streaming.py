from typing import Any

from evals.scripted import ScriptedClient, make_agent, reply, text, tool_use
from harness.tools import Tool


async def _read(tool_input: dict[str, Any]) -> str:
    return "file contents"


READ = Tool(name="read", description="read", input_schema={"type": "object"}, handler=_read)


async def test_text_streams_in_pieces_around_tool_calls():
    events: list[tuple[str, str]] = []
    client = ScriptedClient(
        reply(text("Let me look."), tool_use("a", "read")),
        reply(text("All done here.")),
    )
    agent = make_agent(
        client,
        READ,
        on_text=lambda chunk: events.append(("text", chunk)),
        on_trace=lambda name, tool_input: events.append(("tool", name)),
    )
    result = await agent.run_turn("go")

    kinds = [kind for kind, _ in events]
    split = kinds.index("tool")
    assert "".join(c for k, c in events[:split] if k == "text") == "Let me look."
    assert "".join(c for k, c in events[split + 1 :] if k == "text") == "All done here."
    assert kinds.count("text") > 2  # arrived in pieces, not one blob
    assert result.text == "All done here."


async def test_truncation_notice_is_streamed_with_the_text():
    chunks: list[str] = []
    client = ScriptedClient(reply(text("Half an ans"), stop_reason="max_tokens"))
    result = await make_agent(client, on_text=chunks.append).run_turn("go")

    assert result.text.endswith("[warning: response cut off at max_tokens]")
    assert "".join(chunks) == result.text


async def test_no_listener_still_works():
    client = ScriptedClient(reply(text("quiet")))
    result = await make_agent(client).run_turn("go")
    assert result.text == "quiet"
