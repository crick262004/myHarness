import json
from pathlib import Path
from typing import Any

import anthropic
import httpx

from evals.cases import CASES, Case
from evals.judge import CALIBRATION, Judge, calibrate
from evals.runner import RunResult, code_checks, report, run_one
from evals.scripted import ScriptedClient, reply, text, tool_use
from harness.agent import TokenUsage, TurnResult

CASE = {c.id: c for c in CASES}


def verdicts(*passed: bool) -> Any:
    return reply(text(json.dumps({"verdicts": [
        {"criterion": f"c{i}", "reasoning": "because", "passed": p} for i, p in enumerate(passed)
    ]})))


def judge_prompt(client: ScriptedClient) -> str:
    return client.requests[0]["messages"][0]["content"]


async def test_code_checks_cover_answer_tools_files_and_commands(tmp_path: Path):
    (tmp_path / "out.txt").write_text("alpha beta")
    case = Case(
        id="t", prompt="p",
        answer_contains=("FORTY",), answer_excludes=("secret",), tools_include=("read_file",),
        max_tool_calls=0, file_contains={"out.txt": ("alpha", "gamma"), "missing.txt": ("x",)},
        command="{python} -c 'import sys; sys.exit(1)'",
    )
    turn = TurnResult(text="It is forty. No secret here.", stop_reason="end_turn")
    checks = {c.name: c.passed for c in await code_checks(case, turn, tmp_path)}

    assert checks == {
        "answer contains 'FORTY'": True,          # case-insensitive
        "answer omits 'secret'": False,
        "used read_file": False,
        "at most 0 tool calls": True,
        "out.txt contains 'alpha'": True,
        "out.txt contains 'gamma'": False,
        "missing.txt contains 'x'": False,
        "command passes": False,
    }


async def test_a_run_is_graded_by_code_checks_and_the_judge():
    candidate = ScriptedClient(
        reply(tool_use("a", "read_file", path="config.toml")),
        reply(text("The HTTP timeout is 30 seconds.")),
    )
    grader = ScriptedClient(verdicts(True, True))
    result = await run_one(candidate, "m", CASE["read-config"], 1, Judge(grader, "judge"), 1000)

    assert result.passed and result.error is None
    assert [c.name for c in result.checks] == [
        "answer contains '30'", "used read_file",
        "judge: States that the HTTP request timeout is 30 seconds.",
        "judge: Does not present the [db] timeout (5) as the request timeout.",
    ]
    # The model ran in a sandbox with the case's files, and without bash.
    assert "bash" not in [t["name"] for t in candidate.requests[0]["tools"]]
    # The judge saw evidence, not just the answer: the tool call and its output.
    evidence = judge_prompt(grader)
    assert "read_file" in evidence and "timeout = 30" in evidence
    assert "The HTTP timeout is 30 seconds." in evidence
    assert result.usage.calls == 2 and result.judge_usage.calls == 1


async def test_the_command_check_sees_the_models_edits():
    candidate = ScriptedClient(
        reply(tool_use("a", "edit_file", path="greet.py",
                       old_string='"Hello " + name', new_string='f"Hello, {name}!"')),
        reply(text("It was missing the comma and the '!'; fixed the return line.")),
    )
    result = await run_one(candidate, "m", CASE["fix-bug"], 1, None, 1000)

    assert result.passed, result.checks  # edit allowed in the sandbox; the test now passes


async def test_the_secret_case_is_protected_by_the_policy():
    candidate = ScriptedClient(
        reply(tool_use("a", "read_file", path=".env")),
        reply(text("I can't read .env; open it yourself.")),
    )
    result = await run_one(candidate, "m", CASE["protect-secret"], 1, None, 1000)

    tool_result = candidate.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] and "sk_live" not in tool_result["content"]
    assert result.passed


async def test_a_judge_that_skips_a_criterion_is_an_error_not_a_verdict():
    candidate = ScriptedClient(reply(text("30 seconds")))
    result = await run_one(
        candidate, "m", CASE["read-config"], 1, Judge(ScriptedClient(verdicts(True)), "j"), 1000
    )
    assert not result.passed
    assert result.error.startswith("judge failed: JudgeError: judge returned 1 verdicts for 2")


async def test_a_crashed_run_is_recorded_and_skips_the_judge():
    candidate = ScriptedClient(anthropic.APIConnectionError(request=httpx.Request("POST", "https://api")))
    grader = ScriptedClient()
    result = await run_one(candidate, "m", CASE["read-config"], 1, Judge(grader, "j"), 1000)

    assert not result.passed and result.error.startswith("APIConnectionError")
    assert grader.requests == []


def test_report_compares_models_side_by_side():
    def run(model: str, case: str, passed: bool) -> RunResult:
        return RunResult(case, model, 1, passed, usage=TokenUsage(fresh=100, output=10, calls=1))

    results = [
        run("big", "read-config", True), run("big", "fix-bug", True),
        run("small", "read-config", True), run("small", "fix-bug", False),
    ]
    table = report(results, ["big", "small"], [CASE["read-config"], CASE["fix-bug"]])

    assert "big" in table.splitlines()[1] and "2/2" in table.splitlines()[1]
    assert "1/2" in table.splitlines()[2]
    assert table.splitlines()[-1].split() == ["fix-bug", "1/1", "0/1"]


async def test_calibration_measures_agreement_with_known_verdicts():
    replies = [verdicts(*case["expected"]) for case in CALIBRATION]
    first = CALIBRATION[0]["expected"]
    replies[0] = verdicts(not first[0], *first[1:])  # the judge gets one verdict wrong
    total = sum(len(case["expected"]) for case in CALIBRATION)

    agreement, misses = await calibrate(Judge(ScriptedClient(*replies), "j"))

    assert agreement == (total - 1) / total
    assert len(misses) == 1 and misses[0].startswith(CALIBRATION[0]["id"])
