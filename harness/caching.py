"""Prompt caching: tell the API which prefix of each request to keep warm.

The API reads a request in a fixed order — tools, then system, then messages
— and the cache matches an exact prefix of that. Our requests only ever grow
by appending, so each call starts with the previous call's exact bytes. Mark
where the reusable part ends ("breakpoints") and the next call reads it back
at a tenth of the price instead of paying full price to process it again.
"""

from __future__ import annotations

from typing import Any

CACHE = {"type": "ephemeral"}  # 5-minute lifetime, refreshed on every hit

# What the cache does to the price of a prompt token (5-minute cache).
WRITE_PRICE = 1.25  # storing a prefix costs a bit extra, once
READ_PRICE = 0.10   # every reuse after that costs 10%


def cached_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Breakpoint after the last tool: the tool list survives even when the
    system prompt or history changes."""
    if not tools:
        return tools
    return [*tools[:-1], {**tools[-1], "cache_control": CACHE}]


def cached_system(system: str) -> str | list[dict[str, Any]]:
    """Breakpoint after the system prompt: tools + system survive a history
    rewrite (compaction, cleared tool results)."""
    if not system:
        return system
    return [{"type": "text", "text": system, "cache_control": CACHE}]


def cached_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy of `messages` with breakpoints on the last two user messages.

    The last one stores the whole conversation for the next call. The one
    before it is where the PREVIOUS call ended, so this call is sure to find
    the cache that call stored: the API only searches back 20 blocks from a
    breakpoint, and a reply with many parallel tool calls can add more.

    Only the copy is marked. A request may hold at most 4 breakpoints, so
    marks must never pile up in the stored history.
    """
    marked = list(messages)
    user_turns = [i for i, m in enumerate(messages) if m["role"] == "user"]
    for i in user_turns[-2:]:
        marked[i] = {**messages[i], "content": _mark_last_block(messages[i]["content"])}
    return marked


def _mark_last_block(content: str | list[Any]) -> list[Any]:
    if isinstance(content, str):
        return [{"type": "text", "text": content, "cache_control": CACHE}]
    last = content[-1]
    if not isinstance(last, dict):  # an SDK block object
        last = last.model_dump(exclude_none=True)
    return [*content[:-1], {**last, "cache_control": CACHE}]
