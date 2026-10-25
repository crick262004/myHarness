"""A scripted stand-in for AsyncAnthropic: deterministic evals with no API calls.

Each messages.create() returns the next canned reply and records the request,
so a test can check what the agent did and what it sent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from harness.agent import Agent
from harness.tools import Tool, ToolRegistry


def text(t: str) -> TextBlock:
    return TextBlock(type="text", text=t)


def tool_use(id_: str, name: str, **tool_input: Any) -> ToolUseBlock:
    return ToolUseBlock(type="tool_use", id=id_, name=name, input=tool_input)


Scripted = Message | Exception  # an Exception is raised in place of a reply


def reply(
    *blocks: TextBlock | ToolUseBlock,
    stop_reason: str | None = None,
    usage: Usage | None = None,
) -> Message:
    if stop_reason is None:
        stop_reason = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return Message(
        id="msg_scripted",
        type="message",
        role="assistant",
        model="scripted",
        content=list(blocks),
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=usage or Usage(input_tokens=10, output_tokens=5),
    )


class _ScriptedStream:
    """Mimics the SDK's message stream: text in small chunks, then the full Message."""

    def __init__(self, message: Message) -> None:
        self._message = message

    async def __aenter__(self) -> _ScriptedStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    @property
    def text_stream(self) -> AsyncIterator[str]:
        return self._chunks()

    async def _chunks(self) -> AsyncIterator[str]:
        for block in self._message.content:
            if block.type == "text":
                for i in range(0, len(block.text), 4):
                    yield block.text[i : i + 4]

    async def get_final_message(self) -> Message:
        return self._message


class _ScriptedMessages:
    def __init__(self, replies: list[Scripted]) -> None:
        self.replies = replies
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Message:
        return self._next(kwargs)

    def stream(self, **kwargs: Any) -> _ScriptedStream:
        return _ScriptedStream(self._next(kwargs))

    def _next(self, kwargs: dict[str, Any]) -> Message:
        # Snapshot the history: the agent keeps appending to the same list.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self.replies:
            raise AssertionError("agent made more API calls than the script has replies")
        nxt = self.replies.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class ScriptedClient:
    def __init__(self, *replies: Scripted) -> None:
        self.messages = _ScriptedMessages(list(replies))

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.messages.requests


def make_agent(client: ScriptedClient, *tools: Tool, **hooks: Any) -> Agent:
    """An Agent over the given tools, talking to a scripted model."""
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return Agent(client, registry, "test", "scripted", 1000, **hooks)
