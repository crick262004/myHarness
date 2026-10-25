"""Context compaction: keep a long conversation inside a token budget.

Every API call re-sends the whole history, so each turn costs more and runs
slower than the last, and eventually the context window runs out. Two tiers,
cheapest first:

  1. Clear old tool results. Tool output is usually most of an agent's
     context, and once the model has read it and replied, its own reply has
     captured what mattered. Blanking it costs nothing — no model call.
  2. Summarize. Only if clearing wasn't enough: the model writes a summary
     that replaces the older turns.

Either way the most recent turns stay word for word.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .caching import cached_messages, cached_system, cached_tools

KEEP_RECENT_TURNS = 2       # the latest turns stay verbatim
SUMMARY_MAX_TOKENS = 8_000  # room for thinking plus the summary itself
CHARS_PER_TOKEN = 4         # rough rule of thumb, only used to pick a tier
CLEARED = "[old tool result cleared to save context]"

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
    """What one compaction step did. Tier 1 fills `cleared`, tier 2 `turns`."""

    tokens_before: int       # prompt size that triggered it (0 if unknown)
    cleared: int = 0         # tier 1: old tool results blanked
    tokens_freed: int = 0    # tier 1: estimated tokens that removed
    turns: int = 0           # tier 2: earlier turns folded into the summary
    summary: str = ""        # tier 2: the summary itself


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


def clear_old_tool_results(
    messages: list[dict[str, Any]], keep_turns: int
) -> tuple[list[dict[str, Any]], int, int]:
    """Tier 1: blank the output of every tool call older than the last turns.

    The tool_use block stays (what was called, with which input), so the
    model still knows what it did; only the result body is swapped for a
    short marker. Builds new message dicts rather than editing in place.
    Returns (new_messages, results_cleared, chars_removed).
    """
    old, recent = split_history(messages, keep_turns)
    cleared = chars = 0
    out: list[dict[str, Any]] = []
    for message in old:
        if message["role"] != "user" or isinstance(message["content"], str):
            out.append(message)
            continue
        blocks = []
        for block in message["content"]:
            if block.get("type") == "tool_result" and block.get("content") != CLEARED:
                chars += len(str(block.get("content", "")))
                block = {**block, "content": CLEARED}
                cleared += 1
            blocks.append(block)
        out.append({**message, "content": blocks})
    return out + recent, cleared, chars


async def summarize(
    client: Any,
    *,
    model: str,
    system: str,
    tools: list[dict[str, Any]],
    old: list[dict[str, Any]],
) -> tuple[str, Any]:
    """Tier 2: ask the model to summarize `old`. Returns (summary, usage).

    A side call: not shown, not kept. It sends the same tools, system prompt,
    history and cache breakpoints as the agent's normal calls (tools are
    required anyway once history holds tool_use blocks), so it starts with
    the same bytes and reads the history from the prompt cache. tool_choice
    "none" makes sure it answers in text.
    """
    response = await client.messages.create(
        model=model,
        max_tokens=SUMMARY_MAX_TOKENS,
        system=cached_system(system),
        tools=cached_tools(tools),
        tool_choice={"type": "none"},
        messages=cached_messages([*old, {"role": "user", "content": SUMMARY_PROMPT}]),
    )
    if response.stop_reason == "refusal":
        raise CompactionError("the model declined to write the summary")
    summary = "".join(b.text for b in response.content if b.type == "text").strip()
    if not summary:
        raise CompactionError("the summary came back empty")
    return summary, response.usage


def summary_messages(summary: str) -> list[dict[str, Any]]:
    """The two messages that stand in for everything summarized.

    A user/assistant pair keeps the roles alternating: the recent turns that
    follow start with a user message.
    """
    return [
        {"role": "user", "content": f"[Summary of the earlier conversation]\n{summary}"},
        {"role": "assistant", "content": "Understood. I'll continue from this summary."},
    ]
