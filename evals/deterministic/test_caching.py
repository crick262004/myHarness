from typing import Any

from anthropic.types import Usage

from evals.scripted import ScriptedClient, make_agent, reply, text, tool_use
from harness.agent import TokenUsage
from harness.tools import Tool


async def _read(tool_input: dict[str, Any]) -> str:
    return "file contents"


READ = Tool(name="read", description="read", input_schema={"type": "object"}, handler=_read)


def breakpoints(request: dict[str, Any]) -> list[str]:
    """Where a request asks the API to cache, as readable labels."""
    found = [f"tool:{t['name']}" for t in request["tools"] if "cache_control" in t]
    if isinstance(request["system"], list):
        found += ["system" for b in request["system"] if "cache_control" in b]
    for i, message in enumerate(request["messages"]):
        content = message["content"]
        if isinstance(content, list):
            found += [
                f"message:{i}" for b in content
                if isinstance(b, dict) and "cache_control" in b
            ]
    return found


async def test_requests_mark_tools_system_and_the_last_two_user_messages():
    client = ScriptedClient(reply(tool_use("t", "read")), reply(text("done")))
    await make_agent(client, READ).run_turn("go")

    assert breakpoints(client.requests[0]) == ["tool:read", "system", "message:0"]
    # Second call: messages are [user, assistant(tool_use), user(tool_result)].
    # The previous call ended at message 0, so that stays marked too.
    assert breakpoints(client.requests[1]) == ["tool:read", "system", "message:0", "message:2"]


async def test_marks_never_pile_up_in_stored_history():
    client = ScriptedClient(*(reply(text(f"r{i}")) for i in range(5)))
    agent = make_agent(client, READ)
    for i in range(5):
        await agent.run_turn(f"turn {i}")

    assert len(breakpoints(client.requests[-1])) == 4  # the API's maximum
    assert breakpoints({"tools": [], "system": "", "messages": agent.messages}) == []
    assert all(isinstance(m["content"], str) for m in agent.messages if m["role"] == "user")


async def test_turn_and_session_usage_count_cache_hits():
    client = ScriptedClient(
        reply(tool_use("t", "read"), usage=Usage(
            input_tokens=100, cache_creation_input_tokens=900, output_tokens=20)),
        reply(text("done"), usage=Usage(
            input_tokens=50, cache_read_input_tokens=950, output_tokens=10)),
    )
    agent = make_agent(client, READ)
    result = await agent.run_turn("go")

    u = result.usage
    assert (u.fresh, u.cache_write, u.cache_read, u.output, u.calls) == (150, 900, 950, 30, 2)
    assert u.prompt == 2000
    assert u.cache_hit_rate == 950 / 2000
    assert agent.usage == u  # one turn so far: session == turn


def test_cost_saved_compares_against_no_caching():
    u = TokenUsage(fresh=0, cache_write=0, cache_read=1000)
    assert round(u.input_cost_saved, 2) == 0.90  # reads cost 10%
    u = TokenUsage(fresh=0, cache_write=1000, cache_read=0)
    assert round(u.input_cost_saved, 2) == -0.25  # a write alone costs 25% extra
    assert TokenUsage().input_cost_saved == 0.0
