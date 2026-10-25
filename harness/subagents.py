"""Sub-agent delegation: hand a self-contained task to a fresh Agent.

The parent's context is precious: all of it is re-sent on every call, and
the more it holds, the less attention each part gets. A task like "find
every place the timeout is set" can mean reading twenty files the parent
only needs a summary of. A sub-agent does that work in its own, empty
history and hands back only its final reply, so the parent pays for one
tool call and a short report instead of all the raw material.

Sub-agents get only the read-only tools, and no delegate tool:
  - read-only, so several can run at once (delegate is concurrency-safe)
    and none can change anything behind the parent's back;
  - no delegate, so a sub-agent can't spawn sub-agents of its own.
Their tool calls still pass the parent's permission policy: delegating is
never a way around it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .agent import Agent
from .tools import Tool, ToolError, ToolRegistry

DELEGATE = "delegate"


def subagent_prompt(workdir: Path) -> str:
    return (
        "You are a sub-agent of MyHarness. The main agent gave you one task. You can't "
        "see its conversation and it can't see yours: your final reply is the only "
        "thing it receives.\n"
        f"Working directory: {workdir}\n"
        "Your tools only read; you can't change anything. Investigate with them before "
        "answering, and make independent lookups in parallel.\n"
        "End with a report for the main agent: findings first, with exact paths, names, "
        "line numbers and figures; then anything you could not find out. No preamble."
    )


def read_only_tools(registry: ToolRegistry) -> ToolRegistry:
    """The parent's concurrency-safe (read-only) tools, minus delegate itself."""
    child = ToolRegistry()
    for name in registry.names():
        tool = registry.get(name)
        if tool is not None and tool.concurrency_safe and name != DELEGATE:
            child.register(tool)
    return child


def _labelled(hook: Callable[..., Any] | None, label: str) -> Callable[..., Any] | None:
    """Prefix the tool name a frontend hook shows with the sub-agent's label."""
    if hook is None:
        return None
    return lambda name, tool_input: hook(f"{label} › {name}", tool_input)


def build_delegate_tool(parent: Agent, system_prompt: str, model: str | None = None) -> Tool:
    """A tool that runs a task in a new child Agent and returns its report.

    `model` lets sub-agents run on a different (e.g. faster, cheaper) model
    than the parent; by default they use the parent's.
    """

    async def delegate(tool_input: dict[str, Any]) -> str:
        label = tool_input["description"]
        child = Agent(
            parent.client,
            read_only_tools(parent.registry),
            system_prompt,
            model or parent.model,
            parent.max_tokens,
            policy=parent.policy,
            approve=_labelled(parent.approve, label),
            on_trace=_labelled(parent.on_trace, label),
        )
        try:
            result = await child.run_turn(tool_input["task"])
        finally:
            parent.bill(child.usage)  # tokens a failed sub-agent spent still cost money
        if result.stop_reason == "refusal":
            raise ToolError(f"The sub-agent declined the task. {result.text}".strip())
        return result.text or "(the sub-agent finished without a report)"

    return Tool(
        name=DELEGATE,
        description=(
            "Hand a self-contained task to a sub-agent that starts with an empty context "
            "and this agent's read-only tools (reading and searching files; web search and "
            "memory recall when available). It can't change anything. Only its final "
            "report comes back, so the files and pages it reads never fill this "
            "conversation. Several delegate calls in one reply run at the same time. Use "
            "it for broad research or reading many files; for a call or two, use the "
            "tools directly."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "minLength": 1,
                    "description": "A 3-5 word label for the task, shown to the user.",
                },
                "task": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Complete instructions. The sub-agent sees nothing of this "
                        "conversation: include every path, name and detail it needs, "
                        "and say what its report should contain."
                    ),
                },
            },
            "required": ["description", "task"],
        },
        handler=delegate,
        concurrency_safe=True,
    )
