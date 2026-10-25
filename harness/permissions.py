"""Permissions: the harness decides what a tool call may do, not the model.

The model's tool input is untrusted: a web page or a tool result can talk it
into asking for anything. So every call is checked against a policy first.
Each rule maps a tool-name pattern (and optionally a path pattern) to one of
three decisions: allow, ask the user, or deny outright.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Literal

Decision = Literal["allow", "ask", "deny"]
DECISIONS = ("allow", "ask", "deny")

# Files a tool may never touch, whatever else the rules say. Patterns without
# a "/" match the file name; the rest match the full resolved path.
SECRET_PATHS = (
    ".env", ".env.local", "*.pem", "*.key", "id_rsa*", "id_ed25519*",
    "*/.ssh/*", "*/.certs/*", "*/.aws/*",
)


@dataclass(frozen=True)
class Rule:
    tool: str                  # fnmatch pattern on the tool name, e.g. "mcp__tavily__*"
    decision: Decision
    paths: tuple[str, ...] = ()  # if set, applies only when input["path"] matches one

    def matches(self, name: str, path: Path | None) -> bool:
        if not fnmatch(name, self.tool):
            return False
        if not self.paths:
            return True
        return path is not None and path_matches(path, self.paths)


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reason: str = ""


def path_matches(path: Path, patterns: tuple[str, ...]) -> bool:
    return any(
        fnmatch(path.name if "/" not in p else str(path), p) for p in patterns
    )


# Used when no rule matches: anything unlisted (new built-ins, every MCP tool)
# asks. Trust is something you grant on purpose, in permissions.json.
# remember/forget ask too: a web page could talk the model into saving an
# instruction that then comes back in every future session.
DEFAULT_RULES = (
    Rule("*", "deny", SECRET_PATHS),
    Rule("read_file", "allow"),
    Rule("recall", "allow"),
)


class Policy:
    def __init__(self, workdir: Path, rules: tuple[Rule, ...] = DEFAULT_RULES) -> None:
        self.workdir = workdir.resolve()
        self.rules = rules

    @classmethod
    def load(cls, workdir: Path, config_path: Path) -> Policy:
        """Defaults plus the rules in permissions.json. Yours are checked first."""
        rules: list[Rule] = []
        if config_path.is_file():
            for entry in json.loads(config_path.read_text())["rules"]:
                if entry["decision"] not in DECISIONS:
                    raise ValueError(f"bad decision in {config_path.name}: {entry['decision']!r}")
                rules.append(Rule(entry["tool"], entry["decision"], tuple(entry.get("paths", ()))))
        return cls(workdir, (*rules, *DEFAULT_RULES))

    def check(self, name: str, tool_input: dict[str, Any]) -> Verdict:
        """Any matching deny wins. Otherwise the first matching rule. Otherwise ask."""
        path = self._resolve(tool_input.get("path"))
        matched = [r for r in self.rules if r.matches(name, path)]
        for rule in matched:
            if rule.decision == "deny":
                return Verdict("deny", f"blocked by policy: {name} may not use this path")
        if matched:
            return Verdict(matched[0].decision)
        return Verdict("ask")

    def _resolve(self, raw: Any) -> Path | None:
        """Flatten `..`, symlinks and relative paths before matching, like resolve_path."""
        if not isinstance(raw, str) or not raw:
            return None
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.workdir / candidate
        return candidate.resolve()

    def describe(self) -> list[str]:
        lines = []
        for r in self.rules:
            where = f" on {', '.join(r.paths)}" if r.paths else ""
            lines.append(f"  {r.decision:5} {r.tool}{where}")
        lines.append("  ask   everything else")
        return lines
