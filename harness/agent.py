"""Phase 1: the conversation.

The one idea that matters here: the Messages API is STATELESS. It has no
session, no conversation id, no server-side memory. What feels like a
conversation is `self.messages` — a plain Python list we keep, and re-send
in full on every single call. Delete the list and the "relationship" is gone.
"""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic


class Agent:
    def __init__(
        self,
        client: AsyncAnthropic,
        system_prompt: str,
        model: str,
        max_tokens: int,
    ) -> None:
        self.client = client
        # Standing instructions, sent with every request (who you are, where
        # you're running). Separate from the message history.
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens
        # THE conversation. Entries: {"role": "user"|"assistant", "content": ...}
        self.messages: list[dict[str, Any]] = []

    def clear(self) -> None:
        """Instant amnesia — proves the memory was only ever this list."""
        self.messages = []

    async def run_turn(self, user_input: str) -> str:
        """One exchange: your message in, Claude's reply out."""
        self.messages.append({"role": "user", "content": user_input})

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=self.messages,  # the ENTIRE history, every time
            )
        except BaseException:
            # The turn never completed — history must only hold finished
            # exchanges, or the failed message leaks into future context.
            self.messages.pop()
            raise

        # Append the reply verbatim. `response.content` is a list of blocks —
        # text blocks, and (on this model) thinking blocks too. The API wants
        # them passed back unchanged on later calls, so we never cherry-pick.
        self.messages.append({"role": "assistant", "content": response.content})

        # The human-readable part is just the text blocks joined together.
        text = "".join(block.text for block in response.content if block.type == "text")

        # stop_reason answers "why did the model stop talking?"
        #   end_turn   — finished normally
        #   max_tokens — ran out of output budget mid-sentence
        #   refusal    — declined for safety; content may be empty
        # (Phase 2 adds the interesting one: tool_use.)
        if response.stop_reason == "refusal":
            text = text or "[The model declined this request.]"
        elif response.stop_reason == "max_tokens":
            text += "\n[warning: response cut off at max_tokens]"

        return text
