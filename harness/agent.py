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

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from anthropic import AsyncAnthropic

from .tools import ToolRegistry

# Safety valve: total tool executions allowed in ONE user turn, summed across
# every loop iteration (not per-reply parallelism). A model stuck in a tool
# loop stops getting tools after this and is forced to answer in text.
MAX_TOOL_CALLS_PER_TURN = 50

# Frontend hooks. The Agent does no console I/O itself — the REPL passes
# console implementations; evals pass nothing (approve=None runs gated tools
# without asking, on_trace=None is silent).
ApproveFn = Callable[[str, dict[str, Any]], Awaitable[bool]]
TraceFn = Callable[[str, dict[str, Any]], None]


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
        approve: ApproveFn | None = None,
        on_trace: TraceFn | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        # Standing instructions, sent with every request (who you are, where
        # you're running). Separate from the message history.
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens
        self.approve = approve
        self.on_trace = on_trace
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
            # Once the budget is spent, tool_choice "none" forces a text
            # answer — the loop is guaranteed to terminate on the next reply.
            budget_spent = len(records) >= MAX_TOOL_CALLS_PER_TURN
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=self.messages,          # the ENTIRE history, every time
                tools=self.registry.schemas(),   # "these tools exist"
                tool_choice={"type": "none" if budget_spent else "auto"},
            )
            turn_in += response.usage.input_tokens
            turn_out += response.usage.output_tokens

            # Append the reply verbatim. content is a list of blocks — text,
            # tool_use, and (on this model) thinking blocks. The API wants
            # them passed back unchanged on later calls; never cherry-pick.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                # Done talking — end_turn/max_tokens/refusal. But max_tokens
                # can cut the model off mid-tool-call, leaving tool_use blocks
                # in content. Every tool_use MUST be answered in the next
                # message or all later API calls 400 — so answer them here.
                dangling = [b for b in response.content if b.type == "tool_use"]
                if dangling:
                    self.messages.append({"role": "user", "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": b.id,
                            "content": "Not executed — the response was truncated.",
                            "is_error": True,
                        }
                        for b in dangling
                    ]})
                break

            # The model may request SEVERAL tools in one reply (parallel
            # calls). Run them all, then send ALL results in ONE user message
            # — splitting them across messages trains it to stop parallelizing.
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            results = await self._run_tools(tool_uses, records)
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

    async def _run_tools(
        self, blocks: list[Any], records: list[ToolCallRecord]
    ) -> list[dict[str, Any]]:
        """Run one reply's tool calls; return their tool_results in request order.

        Pass 1 (one at a time) settles what can't run: over budget, or declined
        by the user — a human can only answer one y/N prompt at a time.
        Pass 2 runs the rest in groups from _schedule: a group of reads goes
        out together via asyncio.gather, so the reply waits for the slowest
        read instead of the sum of all of them.
        """
        outcomes: list[tuple[str, bool] | None] = [None] * len(blocks)
        for i, block in enumerate(blocks):
            if len(records) + i >= MAX_TOOL_CALLS_PER_TURN:
                outcomes[i] = (
                    f"Not executed: tool-call limit ({MAX_TOOL_CALLS_PER_TURN}) "
                    "reached for this turn. Answer with what you have.",
                    True,
                )
                continue
            if self.on_trace is not None:
                self.on_trace(block.name, dict(block.input))
            if not await self._approved(block.name, dict(block.input)):
                # The model sees the denial as an error result and moves on.
                outcomes[i] = ("User declined to run this tool.", True)

        for group in self._schedule(blocks, outcomes):
            done = await asyncio.gather(
                *(self.registry.execute(blocks[i].name, dict(blocks[i].input)) for i in group)
            )
            for i, outcome in zip(group, done):
                outcomes[i] = outcome

        results: list[dict[str, Any]] = []
        for block, outcome in zip(blocks, outcomes):
            assert outcome is not None  # every call was settled in pass 1 or 2
            output, is_error = outcome
            records.append(ToolCallRecord(block.name, dict(block.input), output, is_error))
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.id,  # ties this result to that request
                "content": output,
            }
            if is_error:
                result["is_error"] = True
            results.append(result)
        return results

    def _schedule(
        self, blocks: list[Any], outcomes: list[tuple[str, bool] | None]
    ) -> list[list[int]]:
        """Group the still-pending calls (by index) into execution steps, in order.

        Back-to-back concurrency-safe calls share one group and run together.
        Every other call gets a group of its own. So [read, read, edit, read]
        becomes [[0, 1], [2], [3]]: the edit waits for both reads, and the last
        read waits for the edit, exactly as if run one by one — only faster.
        """
        groups: list[list[int]] = []
        joinable = False  # can the next safe call join groups[-1]?
        for i, block in enumerate(blocks):
            if outcomes[i] is not None:
                continue
            tool = self.registry.get(block.name)
            safe = tool is not None and tool.concurrency_safe
            if safe and joinable:
                groups[-1].append(i)
            else:
                groups.append([i])
            joinable = safe
        return groups

    async def _approved(self, name: str, tool_input: dict[str, Any]) -> bool:
        """The permission gate: the model proposes, the harness decides."""
        tool = self.registry.get(name)
        if tool is None or not tool.requires_approval or self.approve is None:
            return True
        return await self.approve(name, tool_input)
