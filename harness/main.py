"""The REPL — read your input, run one turn, print the reply.

Run with:  uv run python -m harness.main
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from .agent import Agent
from .tools import ToolRegistry, build_builtin_tools

# Application policy lives here, not in the Agent. Values come from the
# environment (a local .env is loaded below), with sensible defaults.
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16_000


def build_system_prompt(workdir: Path) -> str:
    return (
        "You are MyHarness, a minimal personal AI agent running in a local terminal REPL.\n"
        f"Working directory: {workdir}\n"
        "Use your tools to inspect files before answering questions about them.\n"
        "Be direct and concise."
    )


def console_trace(name: str, tool_input: dict[str, Any]) -> None:
    text = json.dumps(tool_input, ensure_ascii=False)
    if len(text) > 120:
        text = text[:117] + "..."
    print(f"  ⏺ {name}({text})")


async def console_approve(name: str, tool_input: dict[str, Any]) -> bool:
    # This is a security boundary: show the FULL input, never truncated —
    # anything hidden past a cutoff is exactly where a bad payload would sit.
    print(f"\n  {name} wants to run:")
    for line in json.dumps(tool_input, indent=2, ensure_ascii=False).splitlines():
        print(f"    {line}")
    return input("  allow? [y/N] ").strip().lower() in ("y", "yes")


def build_registry(workdir: Path) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in build_builtin_tools(workdir):
        registry.register(tool)
    return registry


async def main() -> None:
    load_dotenv()  # reads .env in the working directory; real env vars win

    if "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit("Set ANTHROPIC_API_KEY first (in .env or the environment).")

    model = os.environ.get("MYHARNESS_MODEL", DEFAULT_MODEL)
    max_tokens = int(os.environ.get("MYHARNESS_MAX_TOKENS", DEFAULT_MAX_TOKENS))

    workdir = Path.cwd()
    registry = build_registry(workdir)

    async with anthropic.AsyncAnthropic() as client:  # picks up ANTHROPIC_API_KEY itself
        agent = Agent(
            client,
            registry,
            build_system_prompt(workdir),
            model,
            max_tokens,
            approve=console_approve,
            on_trace=console_trace,
        )

        print(f"MyHarness · {model} · /tools lists tools · /clear wipes history · /exit quits")
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
            if user_input == "/tools":
                for name in registry.names():
                    print(f"  {name}")
                continue

            try:
                result = await agent.run_turn(user_input)
            except KeyboardInterrupt:  # Ctrl-C while waiting on the model
                print("\n(interrupted — turn discarded)")
                continue
            except anthropic.APIStatusError as exc:  # the API said no (4xx/5xx)
                print(f"[api error {exc.status_code}] {exc.message}")
                continue
            except anthropic.APIConnectionError:  # never reached the API
                print("[network error — check your connection]")
                continue

            print(f"\nmyHarnessAgent> {result.text}")


if __name__ == "__main__":
    asyncio.run(main())
