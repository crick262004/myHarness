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

import anthropic
from anthropic import AsyncAnthropic

from .caching import READ_PRICE, WRITE_PRICE, cached_messages, cached_system, cached_tools
from .compaction import (
    CHARS_PER_TOKEN,
    KEEP_RECENT_TURNS,
    Compaction,
    CompactionError,
    clear_old_tool_results,
    prompt_tokens,
    split_history,
    summarize,
    summary_messages,
    turn_starts,
)
from .permissions import Policy
from .tools import ToolRegistry

# Safety valve: total tool executions allowed in ONE user turn, summed across
# every loop iteration (not per-reply parallelism). A model stuck in a tool
# loop stops getting tools after this and is forced to answer in text.
MAX_TOOL_CALLS_PER_TURN = 50

# Frontend hooks. The Agent does no console I/O itself — the REPL passes
# console implementations; evals pass nothing (on_trace=None and on_text=None
# are silent). policy=None allows every call; an "ask" verdict with no
# approve function counts as declined.
ApproveFn = Callable[[str, dict[str, Any]], Awaitable[bool]]
TraceFn = Callable[[str, dict[str, Any]], None]
TextFn = Callable[[str], None]  # receives each chunk of model text as it arrives
CompactFn = Callable[[Compaction], None]


@dataclass
class ToolCallRecord:
    """One tool execution, as it actually happened (evals assert on these)."""

    name: str
    input: dict[str, Any]
    output: str
    is_error: bool


@dataclass
class TokenUsage:
    """Token counts as the API reports them, added up over many calls.

    A prompt's tokens split three ways: `fresh` (processed at full price),
    `cache_write` (processed, then stored for reuse) and `cache_read`
    (served from the cache).
    """

    fresh: int = 0
    cache_write: int = 0
    cache_read: int = 0
    output: int = 0
    calls: int = 0

    def add(self, usage: Any) -> None:
        self.fresh += usage.input_tokens
        self.cache_write += usage.cache_creation_input_tokens or 0
        self.cache_read += usage.cache_read_input_tokens or 0
        self.output += usage.output_tokens
        self.calls += 1

    @property
    def prompt(self) -> int:
        return self.fresh + self.cache_write + self.cache_read

    @property
    def cache_hit_rate(self) -> float:
        """Share of prompt tokens served from the cache."""
        return self.cache_read / self.prompt if self.prompt else 0.0

    @property
    def input_cost_saved(self) -> float:
        """Share of input cost saved compared to sending the same prompts uncached."""
        if not self.prompt:
            return 0.0
        paid = self.fresh + WRITE_PRICE * self.cache_write + READ_PRICE * self.cache_read
        return 1 - paid / self.prompt


@dataclass
class TurnResult:
    """Everything one user turn produced. The REPL prints .text; evals read the rest."""

    text: str
    stop_reason: str | None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)

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
        policy: Policy | None = None,
        approve: ApproveFn | None = None,
        on_trace: TraceFn | None = None,
        on_text: TextFn | None = None,
        compact_at: int | None = None,
        on_compact: CompactFn | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        # Standing instructions, sent with every request (who you are, where
        # you're running). Separate from the message history.
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens
        self.policy = policy
        self.approve = approve
        self.on_trace = on_trace
        self.on_text = on_text
        # Compact before a turn once the last prompt reached this many tokens
        # (None = never automatically).
        self.compact_at = compact_at
        self.on_compact = on_compact
        # THE conversation. Entries: {"role": "user"|"assistant", "content": ...}
        self.messages: list[dict[str, Any]] = []
        self.usage = TokenUsage()    # whole session, side calls included
        self.last_prompt_tokens = 0  # size of the most recent request, as the API measured it

    def clear(self) -> None:
        """Instant amnesia — proves the memory was only ever this list."""
        self.messages = []
        self.last_prompt_tokens = 0

    async def compact(self, keep_turns: int = KEEP_RECENT_TURNS) -> Compaction | None:
        """Replace all but the last `keep_turns` turns with a model-written summary.

        Returns None if there is nothing old enough to summarize. Raises
        CompactionError (or an API error) if the summary call fails. The swap
        happens in one assignment at the end, so a failure or Ctrl-C midway
        leaves the history exactly as it was.
        """
        old, recent = split_history(self.messages, keep_turns)
        if not old:
            return None
        summary, usage = await summarize(
            self.client,
            model=self.model,
            system=self.system_prompt,
            tools=self.registry.schemas(),
            old=old,
        )
        self.usage.add(usage)
        result = Compaction(
            tokens_before=self.last_prompt_tokens, turns=len(turn_starts(old)), summary=summary
        )
        self.messages = summary_messages(summary) + recent
        self.last_prompt_tokens = 0  # unknown until the next call measures it
        if self.on_compact is not None:
            self.on_compact(result)
        return result

    async def _shrink_context(self) -> None:
        """Get back under the compaction threshold, cheapest tier first.

        Tier 1 clears old tool results — no model call. If the estimated size
        is still over the threshold, tier 2 summarizes, now from the cleared
        (smaller) history. The estimate only picks the tier; the next real
        call measures the true size.
        """
        before = self.last_prompt_tokens
        cleaned, cleared, chars = clear_old_tool_results(self.messages, KEEP_RECENT_TURNS)
        if cleared:
            self.messages = cleaned
            freed = chars // CHARS_PER_TOKEN
            self.last_prompt_tokens = max(before - freed, 0)
            if self.on_compact is not None:
                self.on_compact(Compaction(tokens_before=before, cleared=cleared, tokens_freed=freed))
            if self.last_prompt_tokens < self.compact_at:
                return
        try:
            await self.compact()
        except (anthropic.APIError, CompactionError):
            pass  # it's an optimization: carry on with the history we have

    async def run_turn(self, user_input: str) -> TurnResult:
        """One user turn — which may now span MANY model calls and tool runs.

        History must only ever contain completed turns. A turn that dies
        midway (API error, Ctrl-C during a tool) is rolled back wholesale:
        everything appended since the turn started is discarded.
        """
        # Compaction runs BETWEEN turns: it only ever sees completed turns,
        # and never touches a tool chain that is still in flight.
        if self.compact_at is not None and self.last_prompt_tokens >= self.compact_at:
            await self._shrink_context()
        start = len(self.messages)
        try:
            return await self._turn(user_input)
        except BaseException:
            del self.messages[start:]
            raise

    async def _turn(self, user_input: str) -> TurnResult:
        self.messages.append({"role": "user", "content": user_input})
        records: list[ToolCallRecord] = []
        turn_usage = TokenUsage()

        while True:
            # Once the budget is spent, tool_choice "none" forces a text
            # answer — the loop is guaranteed to terminate on the next reply.
            budget_spent = len(records) >= MAX_TOOL_CALLS_PER_TURN
            response = await self._call_model(tool_choice="none" if budget_spent else "auto")
            turn_usage.add(response.usage)
            self.usage.add(response.usage)
            self.last_prompt_tokens = prompt_tokens(response.usage)

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

        text ="".join(block.text for block in response.content if block.type == "text")
        notice = ""
        if response.stop_reason == "refusal" and not text:
            notice = "[The model declined this request.]"
        elif response.stop_reason == "max_tokens":
            notice = "\n[warning: response cut off at max_tokens]"
        if notice and self.on_text is not None:
            self.on_text(notice)  # the reply was streamed live; the notice goes the same way
        text += notice

        return TurnResult(
            text=text,
            stop_reason=response.stop_reason,
            tool_calls=records,
            usage=turn_usage,
        )

    async def _call_model(self, tool_choice: str) -> Any:
        """One model call, streamed: text reaches on_text as it is generated.

        The API sends the reply as a series of small events (server-sent
        events). The SDK stream adds them up as they arrive, so
        get_final_message() returns the same complete Message a non-streaming
        create() would — the rest of the loop doesn't change at all.

        The cached_* wrappers add prompt-cache breakpoints to copies, so the
        part of the prompt the previous call already sent is billed at 10%.
        """
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=cached_system(self.system_prompt),
            messages=cached_messages(self.messages),     # the ENTIRE history, every time
            tools=cached_tools(self.registry.schemas()),  # "these tools exist"
            tool_choice={"type": tool_choice},
        ) as stream:
            async for chunk in stream.text_stream:
                if self.on_text is not None:
                    self.on_text(chunk)
            return await stream.get_final_message()

    async def _run_tools(
        self, blocks: list[Any], records: list[ToolCallRecord]
    ) -> list[dict[str, Any]]:
        """Run one reply's tool calls; return their tool_results in request order.

        Pass 1 (one at a time) settles what can't run: over budget, refused
        by validation, policy, or the user — a human can only answer one y/N
        prompt at a time.
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
            blocked = await self._gate(block.name, dict(block.input))
            if blocked is not None:
                # The model sees the refusal as an error result and moves on.
                outcomes[i] = (blocked, True)

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

    async def _gate(self, name: str, tool_input: dict[str, Any]) -> str | None:
        """The permission gate: the model proposes, the harness decides.

        Returns None to let the call run, or the reason it must not. Input is
        validated first, so the user is never asked about a malformed call.
        """
        if (error := self.registry.validate(name, tool_input)) is not None:
            return error
        if self.policy is None or self.registry.get(name) is None:
            return None  # an unknown tool is reported by execute()
        verdict = self.policy.check(name, tool_input)
        if verdict.decision == "deny":
            return verdict.reason
        if verdict.decision == "ask":
            if self.approve is None or not await self.approve(name, tool_input):
                return "User declined to run this tool."
        return None
