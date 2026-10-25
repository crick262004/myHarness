from typing import Any

import pytest

from anthropic.types import Usage

from evals.scripted import ScriptedClient, make_agent, reply, text, tool_use
from harness.compaction import (
    CLEARED,
    SUMMARY_PROMPT,
    CompactionError,
    clear_old_tool_results,
    split_history,
)
from harness.tools import Tool


async def _read(tool_input: dict[str, Any]) -> str:
    return "file contents"


READ = Tool(name="read", description="read", input_schema={"type": "object"}, handler=_read)


def history() -> list[dict[str, Any]]:
    """Three turns; the middle one contains a tool call and its result."""
    return [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": [tool_use("t", "read")]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x"}]},
        {"role": "assistant", "content": "reply 2"},
        {"role": "user", "content": "turn 3"},
        {"role": "assistant", "content": "reply 3"},
    ]


def test_split_cuts_only_at_turn_starts():
    old, recent = split_history(history(), keep_turns=2)
    assert old == history()[:2]
    assert recent[0]["content"] == "turn 2"  # the tool pair stayed together


def test_split_with_too_few_turns_keeps_everything():
    old, recent = split_history(history(), keep_turns=3)
    assert old == [] and recent == history()


async def test_auto_compaction_replaces_old_turns_with_a_summary():
    events: list[Any] = []
    client = ScriptedClient(
        reply(text("reply 1")),
        reply(tool_use("t", "read")),
        reply(text("reply 2")),
        reply(text("reply 3")),
        reply(text("The user asked three things.")),  # the summary call
        reply(text("reply 4")),
    )
    # Scripted replies report 10 input tokens, so a threshold of 10 trips
    # before every turn; compaction only acts once there are >2 turns.
    agent = make_agent(client, READ, compact_at=10, on_compact=events.append)
    for i in range(1, 5):
        await agent.run_turn(f"turn {i}")

    assert len(events) == 1 and events[0].turns == 1 and events[0].cleared == 0
    assert agent.messages[0]["content"].endswith("The user asked three things.")
    assert agent.messages[2]["content"] == "turn 2"  # recent turns verbatim, tool pair intact
    assert [m["content"] for m in agent.messages[-2:]][0] == "turn 4"

    summary_request, normal_request = client.requests[4], client.requests[3]
    assert summary_request["messages"][-1]["content"][0]["text"] == SUMMARY_PROMPT
    assert summary_request["tool_choice"] == {"type": "none"}
    # Same tools and system as the normal calls, so the cached prefix is reused.
    assert summary_request["system"] == normal_request["system"]
    assert summary_request["tools"] == normal_request["tools"]


async def test_nothing_to_compact_makes_no_api_call():
    client = ScriptedClient(reply(text("hi")))
    agent = make_agent(client)
    await agent.run_turn("only turn")

    assert await agent.compact() is None
    assert len(client.requests) == 1


@pytest.mark.parametrize("bad_summary", [
    reply(text("")),
    reply(stop_reason="refusal"),  # what Fable 5 did to the first prompt wording
])
async def test_failed_summary_raises_and_leaves_history_untouched(bad_summary):
    client = ScriptedClient(*(reply(text(f"r{i}")) for i in range(3)), bad_summary)
    agent = make_agent(client)
    for i in range(3):
        await agent.run_turn(f"turn {i}")
    before = list(agent.messages)

    with pytest.raises(CompactionError):
        await agent.compact()
    assert agent.messages == before


async def test_auto_compaction_failure_does_not_block_the_turn():
    client = ScriptedClient(
        *(reply(text(f"r{i}")) for i in range(3)),
        reply(stop_reason="refusal"),
        reply(text("still answered")),
    )
    agent = make_agent(client, compact_at=10)
    for i in range(3):
        await agent.run_turn(f"turn {i}")

    result = await agent.run_turn("turn 3")
    assert result.text == "still answered"
    assert agent.messages[0]["content"] == "turn 0"  # full history kept


def test_clearing_blanks_only_old_tool_results():
    messages = history() + [
        {"role": "user", "content": "turn 4"},
        {"role": "assistant", "content": "reply 4"},
    ]
    original = [dict(m) for m in messages]
    cleaned, cleared, chars = clear_old_tool_results(messages, keep_turns=2)

    assert cleared == 1 and chars == 1  # the one result in turn 2, which is now old
    assert cleaned[4]["content"][0]["content"] == CLEARED
    assert cleaned[4]["content"][0]["tool_use_id"] == "t"  # still paired with its tool_use
    assert cleaned[3] == messages[3]  # the tool_use itself is kept
    assert messages == original  # the input history was not edited in place


def big_turn_replies(prompt_size: int) -> list:
    """One turn that calls a tool returning 4,000 chars (~1,000 tokens)."""
    return [
        reply(tool_use("t", "big")),
        reply(text("read it"), usage=Usage(input_tokens=prompt_size, output_tokens=5)),
    ]


async def _big(tool_input: dict[str, Any]) -> str:
    return "x" * 4000


BIG = Tool(name="big", description="big", input_schema={"type": "object"}, handler=_big)


async def test_tier_1_clearing_avoids_the_summary_call():
    events: list[Any] = []
    client = ScriptedClient(
        *big_turn_replies(10),
        reply(text("r2")),
        reply(text("r3"), usage=Usage(input_tokens=1500, output_tokens=5)),
        reply(text("r4")),
    )
    agent = make_agent(client, BIG, compact_at=1000, on_compact=events.append)
    for i in range(1, 5):
        await agent.run_turn(f"turn {i}")

    # 1,500 tokens - ~1,000 freed by clearing is under 1,000: no summary needed.
    assert [(e.cleared, e.summary) for e in events] == [(1, "")]
    assert len(client.requests) == 5  # no extra side call
    assert agent.messages[2]["content"][0]["content"] == CLEARED


async def test_tier_2_runs_when_clearing_is_not_enough():
    events: list[Any] = []
    client = ScriptedClient(
        *big_turn_replies(10),
        reply(text("r2")),
        reply(text("r3"), usage=Usage(input_tokens=5000, output_tokens=5)),
        reply(text("a short summary")),
        reply(text("r4")),
    )
    agent = make_agent(client, BIG, compact_at=1000, on_compact=events.append)
    for i in range(1, 5):
        await agent.run_turn(f"turn {i}")

    assert [(e.cleared, bool(e.summary)) for e in events] == [(1, False), (0, True)]
    summary_request = client.requests[4]
    sent_result = summary_request["messages"][2]["content"][0]
    assert sent_result["content"] == CLEARED  # cleared BEFORE summarizing
