import asyncio
from typing import Any

from evals.scripted import ScriptedClient, make_agent, reply, text, tool_use
from harness.tools import Tool


class Probe:
    """Counts how many tool calls are in flight at once."""

    def __init__(self) -> None:
        self.now = 0
        self.peak = 0
        self.calls: list[str] = []

    def tool(self, name: str, *, safe: bool, approval: bool = False) -> Tool:
        async def handler(tool_input: dict[str, Any]) -> str:
            self.calls.append(f"{name}:{tool_input['x']}")
            self.now += 1
            self.peak = max(self.peak, self.now)
            await asyncio.sleep(tool_input.get("delay", 0.02))
            self.now -= 1
            return f"{name} got {tool_input['x']}"

        return Tool(
            name=name,
            description="probe",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
            handler=handler,
            requires_approval=approval,
            concurrency_safe=safe,
        )


async def test_safe_calls_run_together():
    probe = Probe()
    client = ScriptedClient(
        reply(tool_use("a", "read", x=1), tool_use("b", "read", x=2), tool_use("c", "read", x=3)),
        reply(text("done")),
    )
    result = await make_agent(client, probe.tool("read", safe=True)).run_turn("go")

    assert probe.peak == 3
    assert result.text == "done"


async def test_unsafe_calls_run_one_at_a_time():
    probe = Probe()
    client = ScriptedClient(
        reply(tool_use("a", "write", x=1), tool_use("b", "write", x=2)),
        reply(text("done")),
    )
    await make_agent(client, probe.tool("write", safe=False)).run_turn("go")

    assert probe.peak == 1
    assert probe.calls == ["write:1", "write:2"]


async def test_unsafe_call_splits_a_batch_of_reads():
    probe = Probe()
    client = ScriptedClient(
        reply(tool_use("a", "read", x=1), tool_use("b", "write", x=2), tool_use("c", "read", x=3)),
        reply(text("done")),
    )
    await make_agent(client, probe.tool("read", safe=True), probe.tool("write", safe=False)).run_turn("go")

    assert probe.peak == 1  # the two reads are not adjacent, so they never overlap


async def test_results_come_back_in_request_order_in_one_message():
    probe = Probe()
    # The first call is the slowest, so it finishes last.
    client = ScriptedClient(
        reply(tool_use("a", "read", x=1, delay=0.08), tool_use("b", "read", x=2, delay=0.01)),
        reply(text("done")),
    )
    result = await make_agent(client, probe.tool("read", safe=True)).run_turn("go")

    tool_results = client.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in tool_results] == ["a", "b"]
    assert [r["content"] for r in tool_results] == ["read got 1", "read got 2"]
    assert result.tool_names() == ["read", "read"]


async def test_declined_call_never_runs():
    probe = Probe()

    async def approve(name: str, tool_input: dict[str, Any]) -> bool:
        return tool_input["x"] != 2

    client = ScriptedClient(
        reply(tool_use("a", "write", x=1), tool_use("b", "write", x=2)),
        reply(text("done")),
    )
    await make_agent(client, probe.tool("write", safe=False, approval=True), approve=approve).run_turn("go")

    assert probe.calls == ["write:1"]
    declined = client.requests[1]["messages"][-1]["content"][1]
    assert declined["is_error"] is True
