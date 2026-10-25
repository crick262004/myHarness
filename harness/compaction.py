"""Context compaction: keep a long conversation inside a token budget.

Every API call re-sends the whole history, so each turn costs more and runs
slower than the last, and eventually the context window runs out. Compaction
replaces the older turns with a summary the model writes itself, and keeps
the most recent turns word for word.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

KEEP_RECENT_TURNS = 2       # the latest turns stay verbatim
SUMMARY_MAX_TOKENS = 8_000  # room for thinking plus the summary itself

# Framed as a handoff note on purpose. An earlier "summarize so you can
# continue from it" wording was refused by Claude Fable 5's safety filter as
# an attempt to extract the model's reasoning (stop_reason "refusal").
SUMMARY_PROMPT = """\
Write a handoff note for this conversation so a colleague can pick it up in \
a fresh session. Cover: the user's goals and stated preferences; decisions \
made and why; files read or changed and their current state; key facts from \
tool results (exact paths, names, numbers, error messages); open tasks. Skip \
pleasantries, dead ends, and raw tool output that no longer matters.

Plain prose and short lists, under 600 words. Output only the note."""


class CompactionError(Exception):
    """The summary call came back unusable (refused or empty)."""


@dataclass
class Compaction:
    turns: int          # earlier turns folded into the summary
    tokens_before: int  # prompt size that triggered it (0 if unknown)
    summary: str


def prompt_tokens(usage: Any) -> int:
    """Full size of the prompt the API just processed, cached parts included."""
    return (
        usage.input_tokens
        + (getattr(usage, "cache_creation_input_tokens", None) or 0)
        + (getattr(usage, "cache_read_input_tokens", None) or 0)
    )


def turn_starts(messages: list[dict[str, Any]]) -> list[int]:
    """Indices where a turn begins: a user message holding typed text.

    Tool results are user messages too, but their content is a list of
    blocks, so they never count as a turn start.
    """
    return [
        i for i, m in enumerate(messages)
        if m["role"] == "user" and isinstance(m["content"], str)
    ]


def split_history(
    messages: list[dict[str, Any]], keep_turns: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into (old, recent), where recent is the last keep_turns turns.

    The cut is always at a turn start. A tool_use and its tool_result live
    inside one turn, so they always land on the same side: the API rejects
    any history where a tool_use has no matching result.
    """
    starts = turn_starts(messages)
    if len(starts) <= keep_turns:
        return [], messages
    cut = starts[-keep_turns] if keep_turns > 0 else len(messages)
    return messages[:cut], messages[cut:]


async def summarize(
    client: Any,
    *,
    model: str,
    system: str,
    tools: list[dict[str, Any]],
    old: list[dict[str, Any]],
) -> str:
    """Ask the model to summarize `old`. A side call: not shown, not kept.

    It sends the same system prompt, tools and history as the agent's normal
    calls (tools are required anyway once history holds tool_use blocks), so
    the request starts with the same bytes and a prompt cache can be reused.
    tool_choice "none" makes sure it answers in text.
    """
    response = await client.messages.create(
        model=model,
        max_tokens=SUMMARY_MAX_TOKENS,
        system=system,
        tools=tools,
        tool_choice={"type": "none"},
        messages=[*old, {"role": "user", "content": SUMMARY_PROMPT}],
    )
    if response.stop_reason == "refusal":
        raise CompactionError("the model declined to write the summary")
    summary = "".join(b.text for b in response.content if b.type == "text").strip()
    if not summary:
        raise CompactionError("the summary came back empty")
    return summary


def summary_messages(summary: str) -> list[dict[str, Any]]:
    """The two messages that stand in for everything summarized.

    A user/assistant pair keeps the roles alternating: the recent turns that
    follow start with a user message.
    """
    return [
        {"role": "user", "content": f"[Summary of the earlier conversation]\n{summary}"},
        {"role": "assistant", "content": "Understood. I'll continue from this summary."},
    ]
