"""Phase 1: the REPL — read your input, run one turn, print the reply.

Run with:  uv run python -m harness.main
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from .agent import Agent

# Application policy lives here, not in the Agent. Values come from the
# environment (a local .env is loaded below), with sensible defaults.
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16_000


def build_system_prompt(workdir: Path) -> str:
    return (
        "You are MyHarness, a minimal personal AI agent running in a local terminal REPL.\n"
        f"Working directory: {workdir}\n"
        "Be direct and concise."
    )


async def main() -> None:
    load_dotenv()  # reads .env in the working directory; real env vars win

    if "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit("Set ANTHROPIC_API_KEY first (in .env or the environment).")

    model = os.environ.get("MYHARNESS_MODEL", DEFAULT_MODEL)
    max_tokens = int(os.environ.get("MYHARNESS_MAX_TOKENS", DEFAULT_MAX_TOKENS))

    async with anthropic.AsyncAnthropic() as client:  # picks up ANTHROPIC_API_KEY itself
        agent = Agent(client, build_system_prompt(Path.cwd()), model, max_tokens)

        print(f"MyHarness · {model} · /clear wipes history · /exit quits")
        while True:
            try:
                user_input = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):  # Ctrl-D / Ctrl-C
                print()
                break

            if not user_input:
                continue
            if user_input == "/exit":
                break
            if user_input == "/clear":
                agent.clear()
                print("(history cleared — the model now remembers nothing)")
                continue

            try:
                reply = await agent.run_turn(user_input)
            except KeyboardInterrupt:  # Ctrl-C while waiting on the model
                print("\n(interrupted — turn discarded)")
                continue
            except anthropic.APIStatusError as exc:  # the API said no (4xx/5xx)
                print(f"[api error {exc.status_code}] {exc.message}")
                continue
            except anthropic.APIConnectionError:  # never reached the API
                print("[network error — check your connection]")
                continue

            print(f"\nclaude> {reply}")


if __name__ == "__main__":
    asyncio.run(main())
