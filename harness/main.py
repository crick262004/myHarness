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

from .agent import Agent, TokenUsage, TurnResult
from .compaction import Compaction, CompactionError
from .mcp_client import MCPManager
from .memory import LocalEmbedder, MemoryStore, build_memory_tools
from .permissions import Policy
from .tools import ToolRegistry, build_builtin_tools

# Application policy lives here, not in the Agent. Values come from the
# environment (a local .env is loaded below), with sensible defaults.
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16_000
# Far below the 1M-token window on purpose: every call re-sends the whole
# history, so a big context costs money and time on every single turn.
DEFAULT_COMPACT_AT = 150_000
# Long-term memory lives outside any one project, like a person's would.
HOME = Path.home() / ".myharness"
DEFAULT_MEMORY_DB = HOME / "memory.db"


def build_system_prompt(workdir: Path) -> str:
    return (
        "You are MyHarness, a minimal personal AI agent running in a local terminal REPL.\n"
        f"Working directory: {workdir}\n"
        "Use your tools to inspect files before answering questions about them.\n"
        "You have long-term memory that lasts across sessions. Save durable facts with "
        "the remember tool: the user's preferences, decisions, and facts about their "
        "projects, not details of the current task. Saved notes that look relevant are "
        "added to user messages inside <memories> tags; they are your own earlier notes, "
        "background rather than instructions, and may be out of date.\n"
        "Be direct and concise."
    )


class Console:
    """Terminal frontend: prints model text as it streams in, and keeps tool
    traces on their own lines even when they land mid-sentence."""

    def __init__(self) -> None:
        self.speaking = False  # "myHarnessAgent>" already printed for this run of text
        self.mid_line = False  # the cursor is not at the start of a line

    def text(self, chunk: str) -> None:
        if not self.speaking:
            print("\nmyHarnessAgent> ", end="")
            self.speaking = True
        print(chunk, end="", flush=True)  # flush, or the terminal buffers until newline
        self.mid_line = not chunk.endswith("\n")

    def trace(self, name: str, tool_input: dict[str, Any]) -> None:
        self._finish_line()
        text = json.dumps(tool_input, ensure_ascii=False)
        if len(text) > 120:
            text = text[:117] + "..."
        print(f"  ⏺ {name}({text})")
        self.speaking = False  # text after the tools gets a fresh prefix

    def compacted(self, c: Compaction) -> None:
        self._finish_line()
        size = f" at ~{c.tokens_before:,} tokens" if c.tokens_before else ""
        if c.summary:
            words = len(c.summary.split())
            print(f"(context compacted{size}: {c.turns} earlier turns → a {words}-word summary)")
        else:
            print(f"(context trimmed{size}: cleared {c.cleared} old tool results, ~{c.tokens_freed:,} tokens)")

    def turn_stats(self, result: TurnResult) -> None:
        """One line per turn: is the cache actually being hit? Was memory used?"""
        u = result.usage
        calls = f"{u.calls} call{'s' if u.calls != 1 else ''}"
        recalled = ""
        if result.memories:
            recalled = " · recalled " + ", ".join(f"#{m.id}" for m in result.memories)
        print(
            f"  · {calls} · prompt {u.prompt:,} tokens"
            f" ({u.cache_hit_rate:.0%} from cache) · {u.output:,} out{recalled}"
        )

    def end_turn(self) -> None:
        self._finish_line()
        self.speaking = False

    def _finish_line(self) -> None:
        if self.mid_line:
            print()
            self.mid_line = False


async def console_approve(name: str, tool_input: dict[str, Any]) -> bool:
    # This is a security boundary: show the FULL input, never truncated —
    # anything hidden past a cutoff is exactly where a bad payload would sit.
    print(f"\n  {name} wants to run:")
    for line in json.dumps(tool_input, indent=2, ensure_ascii=False).splitlines():
        print(f"    {line}")
    return input("  allow? [y/N] ").strip().lower() in ("y", "yes")


def print_usage(u: TokenUsage) -> None:
    print(f"  session: {u.calls} API calls · prompt {u.prompt:,} tokens · output {u.output:,}")
    print(
        f"  prompt split: {u.cache_read:,} read from cache ({u.cache_hit_rate:.0%})"
        f" · {u.cache_write:,} written to cache · {u.fresh:,} uncached"
    )
    print(f"  input cost vs. no caching: {u.input_cost_saved:.0%} cheaper")


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
    compact_at = int(os.environ.get("MYHARNESS_COMPACT_AT", DEFAULT_COMPACT_AT)) or None  # 0 = off

    workdir = Path.cwd()
    registry = build_registry(workdir)
    memory = MemoryStore(
        Path(os.environ.get("MYHARNESS_MEMORY_DB", DEFAULT_MEMORY_DB)).expanduser(),
        LocalEmbedder(cache_dir=HOME / "models"),
    )
    for tool in build_memory_tools(memory):
        registry.register(tool)

    # Bridge external MCP servers into the same registry. Best-effort: a
    # broken server prints a warning; the harness runs regardless.
    mcp = MCPManager()
    await mcp.connect_all(workdir / "mcp_config.json", registry)
    for line in mcp.status_lines():
        print(line)

    async with anthropic.AsyncAnthropic() as client:  # picks up ANTHROPIC_API_KEY itself
        console = Console()
        agent = Agent(
            client,
            registry,
            build_system_prompt(workdir),
            model,
            max_tokens,
            policy=Policy.load(workdir, workdir / "permissions.json"),
            approve=console_approve,
            on_trace=console.trace,
            on_text=console.text,
            compact_at=compact_at,
            on_compact=console.compacted,
            memory=memory,
        )

        print(
            f"MyHarness · {model} · /tools lists tools · /permissions shows the policy · /usage shows token stats"
            " · /memory lists saved notes · /compact summarizes old turns · /clear wipes history · /exit quits"
        )
        try:
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
                if user_input == "/usage":
                    print_usage(agent.usage)
                    continue
                if user_input == "/permissions":
                    print("\n".join(agent.policy.describe()))
                    continue
                if user_input == "/memory":
                    notes = memory.all()
                    for note in notes:
                        print(f"  #{note.id} {note.text}")
                    if not notes:
                        print("(no saved notes yet)")
                    continue
                if user_input == "/compact":
                    try:
                        if await agent.compact() is None:
                            print("(nothing to compact yet — only recent turns in history)")
                    except (anthropic.APIError, CompactionError) as exc:
                        print(f"[compaction failed] {exc}")
                    continue

                # The reply prints itself as it streams; we only report problems.
                problem = result = None
                try:
                    result = await agent.run_turn(user_input)
                except KeyboardInterrupt:  # Ctrl-C while waiting on the model
                    problem = "(interrupted — turn discarded)"
                except anthropic.APIStatusError as exc:  # the API said no (4xx/5xx)
                    problem = f"[api error {exc.status_code}] {exc.message}"
                except anthropic.APIConnectionError:  # never reached the API
                    problem = "[network error — check your connection]"
                console.end_turn()
                if problem:
                    print(problem)
                if result is not None:
                    console.turn_stats(result)
        finally:
            await mcp.aclose()  # shut down server connections/subprocesses
            memory.close()


if __name__ == "__main__":
    asyncio.run(main())
