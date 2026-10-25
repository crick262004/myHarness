from pathlib import Path
from typing import Any

from evals.scripted import ScriptedClient, make_agent, reply, text, tool_use
from harness.permissions import Policy, Rule
from harness.tools import Tool, ToolRegistry, build_builtin_tools


def probe_tool(name: str, ran: list[str]) -> Tool:
    async def handler(tool_input: dict[str, Any]) -> str:
        ran.append(name)
        return "ok"

    return Tool(
        name=name,
        description="probe",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "n": {"type": "integer"}},
            "required": ["path"],
        },
        handler=handler,
    )


async def run_one(tmp_path: Path, tool_name: str, policy: Policy, approve=None, **tool_input: Any):
    """One scripted turn calling one tool; returns (tool_result, tools_that_ran)."""
    ran: list[str] = []
    client = ScriptedClient(reply(tool_use("a", tool_name, **tool_input)), reply(text("done")))
    agent = make_agent(client, probe_tool(tool_name, ran), policy=policy, approve=approve)
    await agent.run_turn("go")
    return client.requests[1]["messages"][-1]["content"][0], ran


async def test_deny_blocks_without_asking(tmp_path):
    asked = []

    async def approve(name, tool_input):
        asked.append(name)
        return True

    policy = Policy(tmp_path, (Rule("danger", "deny"),))
    result, ran = await run_one(tmp_path, "danger", policy, approve, path="x")

    assert ran == [] and asked == []
    assert result["is_error"] is True


async def test_allow_runs_without_asking(tmp_path):
    policy = Policy(tmp_path, (Rule("safe", "allow"),))
    result, ran = await run_one(tmp_path, "safe", policy, path="x")

    assert ran == ["safe"]
    assert "is_error" not in result


async def test_ask_calls_approve_and_honours_no(tmp_path):
    asked = []

    async def approve(name, tool_input):
        asked.append(name)
        return False

    result, ran = await run_one(tmp_path, "unlisted", Policy(tmp_path), approve, path="x")

    assert asked == ["unlisted"] and ran == []
    assert result["is_error"] is True


async def test_ask_with_no_approver_is_declined(tmp_path):
    _, ran = await run_one(tmp_path, "unlisted", Policy(tmp_path), path="x")
    assert ran == []


async def test_invalid_input_is_rejected_before_any_prompt(tmp_path):
    asked = []

    async def approve(name, tool_input):
        asked.append(name)
        return True

    policy = Policy(tmp_path)  # would ask for this tool
    result, ran = await run_one(tmp_path, "unlisted", policy, approve, path="x", n="not a number")

    assert asked == [] and ran == []
    assert "Invalid input for unlisted at n" in result["content"]


async def test_missing_required_field_is_rejected(tmp_path):
    result, ran = await run_one(tmp_path, "unlisted", Policy(tmp_path), n=1)
    assert ran == []
    assert "'path' is a required property" in result["content"]


def test_secret_files_are_denied_however_the_path_is_spelled(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / ".env").write_text("KEY=1")
    (tmp_path / "link").symlink_to(tmp_path / ".env")
    # even an explicit allow can't override a deny
    policy = Policy(tmp_path, (Rule("read_file", "allow"), *Policy(tmp_path).rules))

    for path in [".env", "./.env", "sub/../.env", str(tmp_path / ".env"), "link", "secrets/server.pem"]:
        assert policy.check("read_file", {"path": path}).decision == "deny", path
    assert policy.check("read_file", {"path": "notes.txt"}).decision == "allow"


def test_defaults_ask_for_unlisted_and_mcp_tools(tmp_path):
    policy = Policy(tmp_path)
    assert policy.check("read_file", {"path": "a.txt"}).decision == "allow"
    for name in ["bash", "write_file", "edit_file", "mcp__tavily__tavily_search"]:
        assert policy.check(name, {}).decision == "ask", name


def test_config_rules_come_before_defaults(tmp_path):
    config = tmp_path / "permissions.json"
    config.write_text('{"rules": [{"tool": "mcp__tavily__*", "decision": "allow"}]}')
    policy = Policy.load(tmp_path, config)
    assert policy.check("mcp__tavily__tavily_search", {}).decision == "allow"
    assert policy.check("mcp__other__thing", {}).decision == "ask"


async def test_builtin_read_file_cannot_read_env(tmp_path):
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=secret")
    registry_tools = build_builtin_tools(tmp_path)
    client = ScriptedClient(reply(tool_use("a", "read_file", path=".env")), reply(text("done")))
    agent = make_agent(client, *registry_tools, policy=Policy(tmp_path))
    await agent.run_turn("go")

    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "secret" not in result["content"]


def test_registry_validate_ignores_unknown_tools_and_broken_schemas():
    registry = ToolRegistry()
    registry.register(Tool("odd", "d", {"type": "nonsense"}, handler=None))  # type: ignore[arg-type]
    assert registry.validate("missing", {}) is None
    assert registry.validate("odd", {"anything": 1}) is None
