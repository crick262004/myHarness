"""Phases 1+2: the conversation, and the agentic loop.

Phase 1 idea — the Messages API is STATELESS. What feels like a conversation
is `self.messages`, a plain Python list we keep and re-send in full on every
call. Delete the list and the "relationship" is gone.

Phase 2 idea — the model can't run anything. It replies with tool_use blocks
("please run bash with this command") and stop_reason "tool_use". The harness
runs the real function, appends the result, and calls the API again:

    while the model says "I want to run tools":
        run them, append results, ask again

That loop is the whole trick behind every agent product.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from anthropic import AsyncAnthropic

from .tools import ToolRegistry


@dataclass
class ToolCallRecord:
    """One tool execution, as it actually happened (evals assert on these)."""

    name: str
    input: dict[str, Any]
    output: str
    is_error: bool


@dataclass
class TurnResult:
    """Everything one user turn produced. The REPL prints .text; evals read the rest."""

    text: str
    stop_reason: str | None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    def tool_names(self) -> list[str]:
        return [call.name for call in self.tool_calls]


class Agent:
    def __init__(
        self,
        client: AsyncAnthropic,
        registry: ToolRegistry,
        system_prompt: str,
        model: str,
        max_tokens: int,
        *,
        auto_approve: bool = False,  # evals set True to skip the y/N prompt
        quiet: bool = False,         # suppress the ⏺ trace lines
    ) -> None:
        self.client = client
        self.registry = registry
        # Standing instructions, sent with every request (who you are, where
        # you're running). Separate from the message history.
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens
        self.auto_approve = auto_approve
        self.quiet = quiet
        # THE conversation. Entries: {"role": "user"|"assistant", "content": ...}
        self.messages: list[dict[str, Any]] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def clear(self) -> None:
        """Instant amnesia — proves the memory was only ever this list."""
        self.messages = []

    async def run_turn(self, user_input: str) -> TurnResult:
        """One user turn — which may now span MANY model calls and tool runs.

        History must only ever contain completed turns. A turn that dies
        midway (API error, Ctrl-C during a tool) is rolled back wholesale:
        everything appended since the turn started is discarded.
        """
        start = len(self.messages)
        try:
            return await self._turn(user_input)
        except BaseException:
            del self.messages[start:]
            raise

    async def _turn(self, user_input: str) -> TurnResult:
        self.messages.append({"role": "user", "content": user_input})
        records: list[ToolCallRecord] = []
        turn_in = turn_out = 0

        while True:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=self.messages,          # the ENTIRE history, every time
                tools=self.registry.schemas(),   # "these tools exist"
            )
            turn_in += response.usage.input_tokens
            turn_out += response.usage.output_tokens

            # Append the reply verbatim. content is a list of blocks — text,
            # tool_use, and (on this model) thinking blocks. The API wants
            # them passed back unchanged on later calls; never cherry-pick.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break  # the model is done talking — end_turn/max_tokens/refusal

            # The model may request SEVERAL tools in one reply (parallel
            # calls). Run them all, then send ALL results in ONE user message
            # — splitting them across messages trains it to stop parallelizing.
            results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                self._trace(block.name, block.input)
                output, is_error = await self._run_tool(block.name, dict(block.input))
                records.append(ToolCallRecord(block.name, dict(block.input), output, is_error))
                result: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": block.id,  # ties this result to that request
                    "content": output,
                }
                if is_error:
                    result["is_error"] = True
                results.append(result)
            self.messages.append({"role": "user", "content": results})

        self.total_input_tokens += turn_in
        self.total_output_tokens += turn_out

        text = "".join(block.text for block in response.content if block.type == "text")
        if response.stop_reason == "refusal":
            text = text or "[The model declined this request.]"
        elif response.stop_reason == "max_tokens":
            text += "\n[warning: response cut off at max_tokens]"

        return TurnResult(
            text=text,
            stop_reason=response.stop_reason,
            tool_calls=records,
            input_tokens=turn_in,
            output_tokens=turn_out,
        )

    async def _run_tool(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        """The permission gate: the model proposes, the harness decides."""
        tool = self.registry.get(name)
        if tool is not None and tool.requires_approval and not self.auto_approve:
            if not self._confirm(name, tool_input):
                # The model sees the denial as an error result and moves on.
                return "User declined to run this tool.", True
        return await self.registry.execute(name, tool_input)

    def _confirm(self, name: str, tool_input: dict[str, Any]) -> bool:
        print(f"\n  {name} wants to run:\n    {_summarize(tool_input, limit=500)}")
        answer = input("  allow? [y/N] ").strip().lower()
        return answer in ("y", "yes")

    def _trace(self, name: str, tool_input: Any) -> None:
        if not self.quiet:
            print(f"  ⏺ {name}({_summarize(dict(tool_input))})")


def _summarize(tool_input: dict[str, Any], limit: int = 120) -> str:
    text = json.dumps(tool_input, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."
