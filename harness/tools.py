"""Phase 2: tools — the agent's hands.

A tool is three things bundled together:
  1. a name + description  — what the MODEL reads to decide "should I call this?"
  2. an input JSON Schema  — the shape of the arguments it must supply
  3. a handler             — the plain Python function the HARNESS runs

The model never executes anything. It only *asks*; the harness decides and runs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from jsonschema import validators
from jsonschema.exceptions import SchemaError, best_match

# Tool output is unbounded, untrusted text; cap it so one giant file or a
# runaway command can't flood the context window.
MAX_OUTPUT_CHARS = 50_000
BASH_TIMEOUT_SECONDS = 120

Handler = Callable[[dict[str, Any]], Awaitable[str]]


class ToolError(Exception):
    """A tool failure whose message should go back to the model verbatim."""


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Handler
    # Safe to run at the same time as other calls: it only reads, so order
    # can't matter. Anything that writes stays serialized.
    concurrency_safe: bool = False


class ToolRegistry:
    """All available tools, by name. Built-ins now; MCP tools join in Phase 3."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """Exactly the `tools=` parameter the Messages API expects."""
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self._tools.values()
        ]

    def validate(self, name: str, tool_input: dict[str, Any]) -> str | None:
        """Check the model's arguments against the tool's input schema.

        Returns an error message, or None if the input is fine (or the tool is
        unknown — execute() reports that). The model writes these arguments,
        so wrong types and missing fields are expected; the message goes back
        to it so it can try again.
        """
        tool = self._tools.get(name)
        if tool is None:
            return None
        try:
            cls = validators.validator_for(tool.input_schema)
            cls.check_schema(tool.input_schema)
        except SchemaError:
            return None  # a server's broken schema shouldn't make the tool unusable
        error = best_match(cls(tool.input_schema).iter_errors(tool_input))
        if error is None:
            return None
        where = ".".join(str(p) for p in error.absolute_path)
        return f"Invalid input for {name}{' at ' + where if where else ''}: {error.message}"

    async def execute(self, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
        """Run a tool. Returns (output, is_error) — NEVER raises.

        A failing tool is not a crash: the error text goes back to the model
        as a tool_result with is_error=True, and the model reads it and
        adjusts. That feedback loop is how agents self-correct.
        (KeyboardInterrupt still propagates — the user aborting is not an
        error the model should see.)
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"Unknown tool: {name}", True
        try:
            output = await tool.handler(tool_input)
        except ToolError as exc:
            return str(exc), True
        except Exception as exc:  # noqa: BLE001 — the loop must survive any tool bug
            return f"{type(exc).__name__}: {exc}", True
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + f"\n... (truncated at {MAX_OUTPUT_CHARS} chars)"
        return output, False


def resolve_path(workdir: Path, raw: str) -> Path:
    """Resolve a model-supplied path, confined to the working directory.

    Tool input is untrusted model output: reject anything that escapes the
    workdir via `..`, symlinks, or absolute paths. `resolve()` flattens all
    of those tricks before we check containment.
    """
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = workdir / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(workdir.resolve()):
        raise ToolError(f"Path escapes the working directory: {raw}")
    return resolved


def build_builtin_tools(workdir: Path) -> list[Tool]:
    workdir = workdir.resolve()

    async def bash(tool_input: dict[str, Any]) -> str:
        command = tool_input["command"]
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=workdir,
            stdin=asyncio.subprocess.DEVNULL,  # a command that prompts fails fast instead of hanging
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # one interleaved stream, like a terminal
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=BASH_TIMEOUT_SECONDS)
        except TimeoutError:
            raise ToolError(f"Command timed out after {BASH_TIMEOUT_SECONDS}s: {command}")
        finally:
            # Runs on timeout AND on Ctrl-C/cancellation mid-command: kill the
            # process (or it runs on orphaned) and wait so it's actually reaped.
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        output = stdout.decode(errors="replace")
        if proc.returncode != 0:
            output += f"\n(exit code {proc.returncode})"
        return output or "(no output)"

    async def read_file(tool_input: dict[str, Any]) -> str:
        path = resolve_path(workdir, tool_input["path"])
        if not path.is_file():
            raise ToolError(f"File not found: {tool_input['path']}")
        return path.read_text(errors="replace")

    async def write_file(tool_input: dict[str, Any]) -> str:
        path = resolve_path(workdir, tool_input["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        content = tool_input["content"]
        path.write_text(content)
        return f"Wrote {len(content)} chars to {path.relative_to(workdir)}"

    async def edit_file(tool_input: dict[str, Any]) -> str:
        path = resolve_path(workdir, tool_input["path"])
        if not path.is_file():
            raise ToolError(f"File not found: {tool_input['path']}")
        content = path.read_text()
        old, new = tool_input["old_string"], tool_input["new_string"]
        count = content.count(old)
        if count == 0:
            raise ToolError("old_string not found in file. Read the file and match its content exactly.")
        if count > 1:
            raise ToolError(f"old_string appears {count} times; include more surrounding context so it is unique.")
        path.write_text(content.replace(old, new, 1))
        return f"Edited {path.relative_to(workdir)}"

    return [
        Tool(
            name="bash",
            description=(
                "Run a shell command in the working directory and return its combined "
                "stdout/stderr. Call this to explore the filesystem, run programs, or "
                "use CLI tools. Commands time out after 120 seconds."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run."}
                },
                "required": ["command"],
            },
            handler=bash,
        ),
        Tool(
            name="read_file",
            description=(
                "Read a text file and return its contents. Call this before answering "
                "questions about a file's contents and before editing a file."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the working directory."}
                },
                "required": ["path"],
            },
            handler=read_file,
            concurrency_safe=True,
        ),
        Tool(
            name="write_file",
            description=(
                "Create or overwrite a file with the given content. Parent directories "
                "are created automatically. Use edit_file for small changes to existing files."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the working directory."},
                    "content": {"type": "string", "description": "Full file content to write."},
                },
                "required": ["path", "content"],
            },
            handler=write_file,
        ),
        Tool(
            name="edit_file",
            description=(
                "Replace an exact string in a file with a new string. old_string must "
                "match the file contents exactly and appear exactly once — read the "
                "file first. Use write_file to create new files."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the working directory."},
                    "old_string": {"type": "string", "description": "Exact text to replace (must be unique in the file)."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_string", "new_string"],
            },
            handler=edit_file,
        ),
    ]
