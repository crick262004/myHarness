import asyncio
from typing import Any

import anthropic
import httpx

from evals.scripted import ScriptedClient, make_agent, reply, text, tool_use
from harness.agent import Agent
from harness.permissions import DEFAULT_RULES, Policy, Rule
from harness.subagents import build_delegate_tool, subagent_tools
from harness.tools import Tool, ToolRegistry


class Reads:
    """A read-only tool that counts how many calls are in flight at once."""

    def __init__(self) -> None:
        self.now = self.peak = 0

    def tool(self) -> Tool:
        async def handler(tool_input: dict[str, Any]) -> str:
            self.now += 1
            self.peak = max(self.peak, self.now)
            await asyncio.sleep(0.02)
            self.now -= 1
            return f"RAW CONTENTS OF {tool_input['path']}"

        return Tool("read", "read", {"type": "object"}, handler, concurrency_safe=True, read_only=True)


async def _write(tool_input: dict[str, Any]) -> str:
    return "written"


WRITE = Tool("write", "write", {"type": "object"}, _write)


def with_delegate(agent: Agent) -> Agent:
    agent.registry.register(build_delegate_tool(agent, "SUBAGENT PROMPT"))
    return agent


def delegate(id_: str, task: str) -> Any:
    return tool_use(id_, "delegate", description=f"task {id_}", task=task)


def first_text(message: dict[str, Any]) -> str:
    content = message["content"]
    return content if isinstance(content, str) else content[0]["text"]


async def test_subagent_starts_fresh_and_only_its_report_comes_back():
    client = ScriptedClient(
        reply(delegate("d", "Find the answer in a.txt")),          # parent
        reply(tool_use("r", "read", path="a.txt")),                # sub-agent
        reply(text("The answer is 42 (a.txt, line 1).")),          # sub-agent's report
        reply(text("It's 42.")),                                   # parent
    )
    agent = with_delegate(make_agent(client, Reads().tool(), WRITE))
    result = await agent.run_turn("What's the answer? Use a sub-agent.")

    child = client.requests[1]
    assert [first_text(m) for m in child["messages"]] == ["Find the answer in a.txt"]
    assert child["system"][0]["text"] == "SUBAGENT PROMPT"
    # Read-only tools only, and no delegate: it can't write or spawn its own.
    assert [t["name"] for t in child["tools"]] == ["read"]

    report = client.requests[3]["messages"][-1]["content"][0]
    assert report["content"] == "The answer is 42 (a.txt, line 1)."
    assert "RAW CONTENTS" not in str(agent.messages)  # the raw read stayed in the sub-agent
    assert result.tool_names() == ["delegate"] and result.text == "It's 42."


def test_subagents_get_tools_that_are_both_read_only_and_concurrency_safe():
    registry = ToolRegistry()
    for name, safe, read_only in [
        ("both", True, True),
        ("parallel_writer", True, False),   # e.g. appends to separate files
        ("serial_reader", False, True),     # e.g. reads a one-at-a-time API
        ("writer", False, False),
    ]:
        registry.register(Tool(name, name, {"type": "object"}, _write,
                               concurrency_safe=safe, read_only=read_only))
    parent = with_delegate(make_agent(ScriptedClient(), *(registry.get(n) for n in registry.names())))

    assert parent.registry.get("delegate").read_only  # yet still excluded: no recursion
    assert subagent_tools(parent.registry).names() == ["both"]


async def test_subagent_tokens_are_billed_to_the_parent():
    client = ScriptedClient(
        reply(delegate("d", "look")), reply(text("report")), reply(text("done")),
    )
    agent = with_delegate(make_agent(client))
    result = await agent.run_turn("go")

    assert result.usage.calls == 3 and agent.usage.calls == 3
    assert agent.last_prompt_tokens == 10  # the parent's own context, not the child's


async def test_parallel_delegates_run_at_the_same_time():
    reads = Reads()
    client = ScriptedClient(
        reply(delegate("a", "look at a"), delegate("b", "look at b")),
        reply(tool_use("ra", "read", path="a")),  # sub-agent a
        reply(tool_use("rb", "read", path="b")),  # sub-agent b, while a's read is running
        reply(text("report a")),
        reply(text("report b")),
        reply(text("done")),
    )
    agent = with_delegate(make_agent(client, reads.tool()))
    await agent.run_turn("go")

    assert reads.peak == 2
    results = client.requests[5]["messages"][-1]["content"]
    assert [r["content"] for r in results] == ["report a", "report b"]


async def test_subagent_tool_calls_pass_the_parent_policy(tmp_path):
    asked: list[str] = []

    async def approve(name: str, tool_input: dict[str, Any]) -> bool:
        asked.append(name)
        return False

    client = ScriptedClient(
        reply(delegate("d", "read a")),
        reply(tool_use("r", "read", path="a")),
        reply(text("couldn't read it")),
        reply(text("done")),
    )
    reads = Reads()
    agent = with_delegate(make_agent(client, reads.tool(), policy=Policy(tmp_path), approve=approve))
    await agent.run_turn("go")

    assert asked == ["task d › read"]  # "read" is unlisted, so it asks, labelled
    assert reads.peak == 0
    declined = client.requests[2]["messages"][-1]["content"][0]
    assert declined["is_error"] is True


async def test_denied_tools_stay_denied_inside_a_subagent(tmp_path):
    client = ScriptedClient(
        reply(delegate("d", "read a")),
        reply(tool_use("r", "read", path="a")),
        reply(text("blocked")),
        reply(text("done")),
    )
    policy = Policy(tmp_path, (Rule("read", "deny"), *DEFAULT_RULES))
    agent = with_delegate(make_agent(client, Reads().tool(), policy=policy))
    await agent.run_turn("go")

    assert "blocked by policy" in client.requests[2]["messages"][-1]["content"][0]["content"]


async def test_refusal_comes_back_as_an_error():
    client = ScriptedClient(
        reply(delegate("d", "something")), reply(stop_reason="refusal"), reply(text("ok")),
    )
    agent = with_delegate(make_agent(client))
    await agent.run_turn("go")

    result = client.requests[2]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "declined" in result["content"]


async def test_a_crashed_subagent_is_reported_and_still_billed():
    connection_error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api"))
    client = ScriptedClient(
        reply(delegate("d", "read a")),
        reply(tool_use("r", "read", path="a")),
        connection_error,                       # the sub-agent's second call fails
        reply(text("I'll do it myself")),
    )
    agent = with_delegate(make_agent(client, Reads().tool()))
    result = await agent.run_turn("go")

    error = client.requests[3]["messages"][-1]["content"][0]
    assert error["is_error"] is True and "APIConnectionError" in error["content"]
    assert result.usage.calls == 3  # parent 2 + the sub-agent's one successful call
