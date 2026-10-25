import json
from typing import Any

import anthropic
import httpx
import pytest

from evals.fakes import WordEmbedder
from evals.scripted import ScriptedClient, make_agent, reply, text
from harness.graph import Step
from harness.llm import LLMError, ask_json, ask_text
from harness.memory import MemoryStore
from harness.workflows import MAX_DRAFTS, MAX_SUBQUESTIONS, describe_step, run_research


def plan(*subquestions: str) -> Any:
    return reply(text(json.dumps({"subquestions": list(subquestions)})))


def verdict(approved: bool, feedback: str = "") -> Any:
    return reply(text(json.dumps({"approved": approved, "feedback": feedback})))


def prompt_of(request: dict[str, Any]) -> str:
    content = request["messages"][0]["content"]
    return content if isinstance(content, str) else content[0]["text"]


async def research(client: ScriptedClient, question: str = "Q?", **kwargs: Any):
    agent = make_agent(client)
    steps: list[Step] = []
    state = await run_research(agent, question, "SUB", on_step=steps.append, **kwargs)
    return agent, state, [describe_step(s) for s in steps]


async def test_plan_research_write_review():
    client = ScriptedClient(
        plan("Where is X set?", "What is Y?"),
        reply(text("X is set in a.py:3")),   # researcher 1
        reply(text("Y is 7")),               # researcher 2, at the same time
        reply(text("X: a.py:3. Y: 7.")),     # write
        verdict(True),                       # review
    )
    agent, state, steps = await research(client, "Where is X and what is Y?")

    assert steps == [
        "plan → 2 sub-questions",
        "research → 2 reports ∥ memory → 0 notes",  # parallel branches, one superstep
        "write → draft 1",
        "review → approved",
    ]
    assert state["draft"] == "X: a.py:3. Y: 7." and state["review"]["approved"]

    planner, r1, r2, writer, reviewer = client.requests
    assert planner["output_config"]["format"]["type"] == "json_schema"
    assert [prompt_of(r) for r in (r1, r2)] == ["Where is X set?", "What is Y?"]  # fresh contexts
    assert "X is set in a.py:3" in prompt_of(writer) and "Y is 7" in prompt_of(writer)
    assert "X: a.py:3. Y: 7." in prompt_of(reviewer)
    assert agent.usage.calls == 5  # every call billed to the agent


async def test_a_rejected_draft_is_rewritten_with_the_feedback():
    client = ScriptedClient(
        plan("Q?"), reply(text("found it")),
        reply(text("draft one")), verdict(False, "Cite the file."),
        reply(text("draft two, citing a.py")), verdict(True),
    )
    _, state, steps = await research(client)

    rewrite = prompt_of(client.requests[4])
    assert "draft one" in rewrite and "Cite the file." in rewrite
    assert steps[-3:] == ["review → changes requested: Cite the file.", "write → draft 2", "review → approved"]
    assert state["draft"] == "draft two, citing a.py"


async def test_rewrites_stop_after_the_last_draft():
    rounds = [r for i in range(MAX_DRAFTS) for r in (reply(text(f"draft {i}")), verdict(False, "no"))]
    client = ScriptedClient(plan("Q?"), reply(text("found")), *rounds)
    _, state, _ = await research(client)

    assert state["drafts"] == MAX_DRAFTS and not state["review"]["approved"]
    assert state["draft"] == f"draft {MAX_DRAFTS - 1}"  # the best we have is returned


async def test_one_failed_researcher_does_not_sink_the_run():
    client = ScriptedClient(
        plan("A?", "B?"),
        anthropic.APIConnectionError(request=httpx.Request("POST", "https://api")),
        reply(text("B is fine")),
        reply(text("B is fine; A unknown")), verdict(True),
    )
    _, state, _ = await research(client)

    a, b = state["findings"]
    assert a["report"].startswith("(research failed: APIConnectionError")
    assert b["report"] == "B is fine"


async def test_saved_notes_reach_the_writer(tmp_path):
    store = MemoryStore(tmp_path / "m", WordEmbedder())
    await store.add("user prefers short answers")
    client = ScriptedClient(plan("Q?"), reply(text("found")), reply(text("short")), verdict(True))
    _, _, steps = await research(client, "Which short answers does the user prefer?", memory=store)

    assert "memory → 1 note" in steps[1]
    assert "Notes about the user:\n- user prefers short answers" in prompt_of(client.requests[2])


@pytest.mark.parametrize("planned, researched", [
    ([], ["Q?"]),  # nothing planned: research the question itself
    ([f"s{i}" for i in range(MAX_SUBQUESTIONS + 2)], [f"s{i}" for i in range(MAX_SUBQUESTIONS)]),
])
async def test_plan_size_is_kept_in_bounds(planned, researched):
    reports = [reply(text("r")) for _ in researched]
    client = ScriptedClient(plan(*planned), *reports, reply(text("d")), verdict(True))
    _, state, _ = await research(client)
    assert state["subquestions"] == researched


async def test_ask_json_and_ask_text_reject_unusable_replies():
    with pytest.raises(LLMError, match="declined"):
        await ask_json(ScriptedClient(reply(stop_reason="refusal")), model="m", prompt="p", schema={})
    with pytest.raises(LLMError, match="cut off"):
        await ask_json(ScriptedClient(reply(text('{"a": '), stop_reason="max_tokens")),
                       model="m", prompt="p", schema={})
    with pytest.raises(LLMError, match="not valid JSON"):
        await ask_json(ScriptedClient(reply(text("sure! {"))), model="m", prompt="p", schema={})
    with pytest.raises(LLMError, match="empty"):
        await ask_text(ScriptedClient(reply(text("  "))), model="m", prompt="p")
