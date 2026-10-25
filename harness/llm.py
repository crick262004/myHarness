"""One-shot model calls, for code that needs an answer rather than a conversation.

Graph nodes and the eval judge ask once and need to know what comes back.
ask_json uses the API's structured outputs: decoding is constrained to the
JSON schema, so a finished reply always parses into that shape. (A forced
tool call does the same job, but some models reject a forced tool_choice.)
"""

from __future__ import annotations

import json
from typing import Any

DEFAULT_MAX_TOKENS = 8_000


class LLMError(Exception):
    """The model gave no usable answer (refused, cut off, or empty)."""


async def ask_text(
    client: Any, *, model: str, prompt: str, system: str = "", max_tokens: int = DEFAULT_MAX_TOKENS
) -> tuple[str, Any]:
    """Returns (reply text, usage)."""
    response = await _create(client, model, prompt, system, max_tokens)
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise LLMError("the reply was empty")
    return text, response.usage


async def ask_json(
    client: Any,
    *,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    system: str = "",
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[dict[str, Any], Any]:
    """Returns (the reply parsed as JSON matching `schema`, usage)."""
    response = await _create(
        client, model, prompt, system, max_tokens,
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    try:
        return json.loads(text), response.usage
    except json.JSONDecodeError:
        raise LLMError(f"the reply was not valid JSON: {text[:200]!r}") from None


async def _create(
    client: Any, model: str, prompt: str, system: str, max_tokens: int, **extra: Any
) -> Any:
    kwargs: dict[str, Any] = dict(
        model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}], **extra
    )
    if system:
        kwargs["system"] = system
    response = await client.messages.create(**kwargs)
    # Constrained decoding guarantees the shape only for a reply that finished.
    if response.stop_reason == "refusal":
        raise LLMError("the model declined")
    if response.stop_reason == "max_tokens":
        raise LLMError(f"the reply was cut off at max_tokens ({max_tokens})")
    return response
