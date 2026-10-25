import logging
from pathlib import Path

import pytest

from evals.fakes import WordEmbedder
from evals.scripted import ScriptedClient, make_agent, reply, text, tool_use
from harness.compaction import turn_starts
from harness.memory import MemoryStore, build_memory_tools, format_memories, recalled_ids
from harness.permissions import Policy
from harness.tools import ToolError


NEOVIM = "user prefers neovim editor"
TABS = "user indents python with tabs"
PYTEST = "project uses pytest"


async def store_with(tmp_path: Path, *notes: str, embedder: WordEmbedder | None = None) -> MemoryStore:
    store = MemoryStore(tmp_path / "memory.db", embedder or WordEmbedder())
    for note in notes:
        await store.add(note)
    return store


async def test_notes_persist_across_sessions(tmp_path):
    (await store_with(tmp_path, NEOVIM)).close()

    reopened = MemoryStore(tmp_path / "memory.db", WordEmbedder())
    assert [m.text for m in reopened.all()] == [NEOVIM]
    assert [m.text for m in await reopened.search("which editor")] == [NEOVIM]


async def test_search_ranks_by_meaning_and_drops_weak_matches(tmp_path):
    store = await store_with(tmp_path, NEOVIM, TABS, PYTEST)
    query = "does the user indent python with tabs"

    ranked = await store.search(query, min_score=-1)
    assert [m.text for m in ranked[:2]] == [TABS, NEOVIM]  # NEOVIM shares only "user"
    assert [m.text for m in await store.search(query)] == [TABS]  # the cutoff drops it


async def test_a_repeated_fact_is_not_stored_twice(tmp_path):
    store = await store_with(tmp_path, NEOVIM)
    memory, created = await store.add("  User prefers NEOVIM editor ")

    assert not created and memory.id == 1
    assert store.count() == 1


async def test_changing_the_embedding_model_reembeds_old_notes(tmp_path):
    (await store_with(tmp_path, NEOVIM)).close()

    new_model = WordEmbedder(name="words-v2")
    reopened = MemoryStore(tmp_path / "memory.db", new_model)
    assert new_model.calls == 1  # every old note re-embedded, in one batch
    # Search only compares vectors from the current model; this finds the
    # note only because its vector was redone.
    assert [m.text for m in await reopened.search("neovim editor")] == [NEOVIM]


async def test_relevant_notes_are_added_to_the_user_message(tmp_path):
    store = await store_with(tmp_path, NEOVIM, PYTEST)
    client = ScriptedClient(reply(text("neovim")))
    agent = make_agent(client, memory=store)

    result = await agent.run_turn("what editor does the user prefer?")

    sent = agent.messages[0]["content"]
    assert sent.startswith("<memories>") and sent.endswith("what editor does the user prefer?")
    assert NEOVIM in sent and PYTEST not in sent
    assert [m.text for m in result.memories] == [NEOVIM]
    assert turn_starts(agent.messages) == [0]  # still a plain-text turn start


async def test_a_note_is_shown_once_per_session(tmp_path):
    store = await store_with(tmp_path, NEOVIM)
    client = ScriptedClient(reply(text("neovim")), reply(text("still neovim")))
    agent = make_agent(client, memory=store)

    await agent.run_turn("what editor does the user prefer?")
    second = await agent.run_turn("and which editor does the user prefer again?")

    assert second.memories == []
    assert agent.messages[2]["content"] == "and which editor does the user prefer again?"


async def test_a_rolled_back_turn_does_not_count_as_shown(tmp_path):
    store = await store_with(tmp_path, NEOVIM)
    client = ScriptedClient()  # no replies: the first turn fails
    agent = make_agent(client, memory=store)

    with pytest.raises(AssertionError):
        await agent.run_turn("what editor does the user prefer?")
    client.messages.replies.append(reply(text("neovim")))
    result = await agent.run_turn("what editor does the user prefer?")

    assert [m.text for m in result.memories] == [NEOVIM]


async def test_a_memory_failure_never_blocks_the_turn(tmp_path, caplog):
    embedder = WordEmbedder()
    store = await store_with(tmp_path, NEOVIM, embedder=embedder)
    embedder.broken = True
    agent = make_agent(ScriptedClient(reply(text("answered anyway"))), memory=store)

    with caplog.at_level(logging.WARNING):
        result = await agent.run_turn("what editor does the user prefer?")

    assert result.text == "answered anyway" and result.memories == []
    assert agent.messages[0]["content"] == "what editor does the user prefer?"
    assert "memory search failed" in caplog.text


async def test_remember_asks_first_and_recall_does_not(tmp_path):
    store = await store_with(tmp_path)
    asked: list[str] = []

    async def approve(name, tool_input):
        asked.append(name)
        return True

    client = ScriptedClient(
        reply(tool_use("a", "remember", text=NEOVIM)),
        reply(tool_use("b", "recall", query="neovim editor")),
        reply(text("saved")),
    )
    agent = make_agent(
        client, *build_memory_tools(store), policy=Policy(tmp_path), approve=approve, memory=store
    )
    result = await agent.run_turn("remember that I like neovim")

    assert asked == ["remember"]
    assert [m.text for m in store.all()] == [NEOVIM]
    assert result.tool_calls[1].output.startswith(f"#1 (score 0.71) {NEOVIM}")


async def test_forget_deletes_by_id(tmp_path):
    store = await store_with(tmp_path, NEOVIM, PYTEST)
    forget = {t.name: t for t in build_memory_tools(store)}["forget"]

    assert await forget.handler({"id": 1}) == "Deleted memory #1."
    assert [m.text for m in store.all()] == [PYTEST]
    with pytest.raises(ToolError):
        await forget.handler({"id": 1})


def test_shown_notes_are_read_off_the_history():
    block = format_memories([])  # header only
    messages = [
        {"role": "user", "content": "<memories>\nheader\n- [#3] a\n- [#7] b\n</memories>\n\nhi"},
        {"role": "assistant", "content": "- [#9] not a memory block"},
        {"role": "user", "content": "- [#4] typed by the user, not a memory block"},
        {"role": "user", "content": f"{block}\n\n- [#5] after the block"},
    ]
    assert recalled_ids(messages) == {3, 7}
