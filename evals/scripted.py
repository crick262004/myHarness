"""A scripted stand-in for AsyncAnthropic: deterministic evals with no API calls.

Each messages.create() returns the next canned reply and records the request,
so a test can check what the agent did and what it sent.
"""

from __future__ import annotations

from typing import Any

from anthropic.types import Message, TextBlock, ToolUseBlock, Usage


def text(t: str) -> TextBlock:
    return TextBlock(type="text", text=t)


def tool_use(id_: str, name: str, **tool_input: Any) -> ToolUseBlock:
    return ToolUseBlock(type="tool_use", id=id_, name=name, input=tool_input)


def reply(*blocks: TextBlock | ToolUseBlock, stop_reason: str | None = None) -> Message:
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
        usage=Usage(input_tokens=10, output_tokens=5),
    )


class _ScriptedMessages:
    def __init__(self, replies: list[Message]) -> None:
        self.replies = replies
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Message:
        # Snapshot the history: the agent keeps appending to the same list.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self.replies:
            raise AssertionError("agent made more API calls than the script has replies")
        return self.replies.pop(0)


class ScriptedClient:
    def __init__(self, *replies: Message) -> None:
        self.messages = _ScriptedMessages(list(replies))

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.messages.requests
