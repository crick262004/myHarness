"""Multi-model eval runner: the same tasks on several models, one comparison.

    uv run python -m evals.runner --models claude-opus-5,claude-haiku-5-5 --trials 2
    uv run python -m evals.runner --calibrate     # check the judge itself first

Every run gets a fresh temporary directory holding the case's files, and a
fresh Agent: no history, no memory, no MCP, file tools but no bash, and a
policy that allows edits inside that directory only. Runs go out in
parallel (capped by --concurrency). Results print as a table and are saved
as JSON under evals/results/, so runs can be compared over time.

Models are non-deterministic, so one run proves little: --trials repeats
each case and the table shows how often it passed (e.g. 2/3).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import anthropic
from dotenv import load_dotenv

from harness.agent import Agent, TokenUsage, TurnResult
from harness.llm import LLMError
from harness.main import build_system_prompt
from harness.permissions import DEFAULT_RULES, SECRET_PATHS, Policy, Rule, path_matches
from harness.tools import ToolRegistry, build_builtin_tools, walk_files

from .cases import CASES, Case
from .judge import Judge, JudgeError, calibrate

DEFAULT_JUDGE = "claude-opus-5"
COMMAND_TIMEOUT_SECONDS = 60
MAX_JUDGE_FILES = 20
RESULTS_DIR = Path(__file__).parent / "results"

# Inside a throwaway directory, edits are what the tasks ask for. The secret
# deny-list still applies: a matching deny always wins.
EVAL_RULES = (Rule("write_file", "allow"), Rule("edit_file", "allow"), *DEFAULT_RULES)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class RunResult:
    case: str
    model: str
    trial: int
    passed: bool
    checks: list[Check] = field(default_factory=list)
    answer: str = ""
    tools: list[str] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    judge_usage: TokenUsage = field(default_factory=TokenUsage)
    seconds: float = 0.0
    error: str | None = None  # the run or the judge broke: not a verdict on the model

    @property
    def failed_checks(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


def eval_registry(workdir: Path) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in build_builtin_tools(workdir):
        if tool.name != "bash":  # the one tool that could reach outside the directory
            registry.register(tool)
    return registry


async def run_command(command: str, workdir: Path) -> tuple[bool, str]:
    proc = await asyncio.create_subprocess_shell(
        command.replace("{python}", shlex.quote(sys.executable)),
        cwd=workdir,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), COMMAND_TIMEOUT_SECONDS)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return False, f"timed out after {COMMAND_TIMEOUT_SECONDS}s"
    return proc.returncode == 0, out.decode(errors="replace")[-500:]


async def code_checks(case: Case, turn: TurnResult, workdir: Path) -> list[Check]:
    """The exact, free checks: they run before (and regardless of) the judge."""
    answer = turn.text.lower()
    tools = turn.tool_names()
    checks = [Check(f"answer contains {s!r}", s.lower() in answer) for s in case.answer_contains]
    checks += [Check(f"answer omits {s!r}", s.lower() not in answer) for s in case.answer_excludes]
    checks += [Check(f"used {t}", t in tools) for t in case.tools_include]
    if case.max_tool_calls is not None:
        checks.append(Check(
            f"at most {case.max_tool_calls} tool calls", len(tools) <= case.max_tool_calls,
            f"made {len(tools)}",
        ))
    for path, needles in case.file_contains.items():
        file = workdir / path
        body = file.read_text() if file.is_file() else None
        checks += [
            Check(f"{path} contains {n!r}", body is not None and n in body,
                  "" if body is not None else "file missing")
            for n in needles
        ]
    if case.command:
        ok, output = await run_command(case.command, workdir)
        checks.append(Check("command passes", ok, output))
    return checks


def files_after(workdir: Path) -> dict[str, str]:
    """The directory's text files after the run, as evidence for the judge."""
    files: dict[str, str] = {}
    for path in walk_files(workdir, workdir, "*"):
        if len(files) == MAX_JUDGE_FILES:
            break
        if path_matches(path, SECRET_PATHS):
            continue  # the judge doesn't need secrets either
        try:
            files[str(path.relative_to(workdir))] = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
    return files


async def run_one(
    client: Any, model: str, case: Case, trial: int, judge: Judge | None, max_tokens: int
) -> RunResult:
    with tempfile.TemporaryDirectory(prefix=f"myharness-eval-{case.id}-") as tmp:
        workdir = Path(tmp).resolve()
        for rel, body in case.files.items():
            (workdir / rel).parent.mkdir(parents=True, exist_ok=True)
            (workdir / rel).write_text(body)
        agent = Agent(
            client, eval_registry(workdir), build_system_prompt(workdir, memory=False),
            model, max_tokens, policy=Policy(workdir, EVAL_RULES),
        )
        start = time.monotonic()
        try:
            turn = await agent.run_turn(case.prompt)
        except Exception as exc:  # noqa: BLE001 — record it; the other runs go on
            return RunResult(case.id, model, trial, False, usage=agent.usage,
                             seconds=time.monotonic() - start, error=f"{type(exc).__name__}: {exc}")
        seconds = time.monotonic() - start
        result = RunResult(case.id, model, trial, False, answer=turn.text,
                           tools=turn.tool_names(), usage=turn.usage, seconds=seconds)
        result.checks = await code_checks(case, turn, workdir)
        if case.rubric and judge is not None:
            try:
                verdicts, result.judge_usage = await judge.grade(
                    task=case.prompt, answer=turn.text, rubric=list(case.rubric),
                    tool_calls=turn.tool_calls, files=files_after(workdir),
                )
                result.checks += [Check(f"judge: {v.criterion}", v.passed, v.reasoning) for v in verdicts]
            except (JudgeError, LLMError, anthropic.APIError) as exc:
                result.error = f"judge failed: {type(exc).__name__}: {exc}"
        result.passed = result.error is None and all(c.passed for c in result.checks)
        return result


async def run_suite(
    client: Any,
    models: list[str],
    cases: list[Case],
    *,
    trials: int = 1,
    judge: Judge | None = None,
    concurrency: int = 4,
    max_tokens: int = 8_000,
    on_result: Callable[[RunResult], None] | None = None,
) -> list[RunResult]:
    """Every (model, case, trial), at most `concurrency` at a time."""
    limit = asyncio.Semaphore(concurrency)

    async def one(model: str, case: Case, trial: int) -> RunResult:
        async with limit:
            result = await run_one(client, model, case, trial, judge, max_tokens)
        if on_result is not None:
            on_result(result)
        return result

    return list(await asyncio.gather(
        *(one(m, c, t) for m in models for c in cases for t in range(1, trials + 1))
    ))


def report(results: list[RunResult], models: list[str], cases: list[Case]) -> str:
    """A per-model summary table, then a case-by-model pass grid."""
    lines = [
        f"{'model':<22} {'passed':>8} {'checks':>7} {'errors':>6} {'tools':>6} "
        f"{'prompt tok':>11} {'out tok':>8} {'secs':>6}"
    ]
    for model in models:
        runs = [r for r in results if r.model == model]
        n = len(runs) or 1
        checks = [c for r in runs for c in r.checks]
        lines.append(
            f"{model:<22} {sum(r.passed for r in runs):>4}/{len(runs):<3} "
            f"{sum(c.passed for c in checks) / (len(checks) or 1):>7.0%} "
            f"{sum(r.error is not None for r in runs):>6} "
            f"{sum(len(r.tools) for r in runs) / n:>6.1f} "
            f"{sum(r.usage.prompt for r in runs) // n:>11,} "
            f"{sum(r.usage.output for r in runs) // n:>8,} "
            f"{sum(r.seconds for r in runs) / n:>6.1f}"
        )
    width = max(len(m) for m in models)
    lines += ["", f"{'case':<18} " + " ".join(f"{m:>{width}}" for m in models)]
    for case in cases:
        cells = []
        for model in models:
            runs = [r for r in results if r.model == model and r.case == case.id]
            cells.append(f"{sum(r.passed for r in runs)}/{len(runs)}".rjust(width))
        lines.append(f"{case.id:<18} " + " ".join(cells))
    return "\n".join(lines)


def describe_failure(result: RunResult) -> str:
    if result.error:
        return result.error
    return "; ".join(f"{c.name} ({c.detail[:100]})" if c.detail else c.name for c in result.failed_checks)


def save(results: list[RunResult], settings: dict[str, Any], directory: Path = RESULTS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / time.strftime("%Y%m%d-%H%M%S.json")
    path.write_text(json.dumps({**settings, "results": [asdict(r) for r in results]}, indent=2))
    return path


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=os.environ.get("MYHARNESS_MODEL", "claude-opus-5"),
                        help="comma-separated model ids")
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help="model that grades rubrics")
    parser.add_argument("--no-judge", action="store_true", help="code checks only (cheaper)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8_000)
    parser.add_argument("--calibrate", action="store_true", help="grade the judge, then exit")
    args = parser.parse_args()

    load_dotenv()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cases = CASES
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in CASES if c.id in wanted]
        if unknown := wanted - {c.id for c in cases}:
            sys.exit(f"unknown case ids: {', '.join(sorted(unknown))}")

    async with anthropic.AsyncAnthropic(max_retries=4) as client:  # parallel runs meet rate limits
        judge = None if args.no_judge else Judge(client, args.judge)
        if args.calibrate:
            agreement, misses = await calibrate(Judge(client, args.judge))
            print(f"judge {args.judge}: {agreement:.0%} agreement with known verdicts")
            for miss in misses:
                print(f"  ✗ {miss}")
            return

        print(f"{len(models)} models × {len(cases)} cases × {args.trials} trials"
              f" · judge: {'none' if judge is None else args.judge}\n")

        def progress(r: RunResult) -> None:
            mark = "✓" if r.passed else "✗"
            why = "" if r.passed else f" · {describe_failure(r)}"
            print(f"  {mark} {r.model} · {r.case} #{r.trial} · {r.seconds:.1f}s{why}")

        results = await run_suite(
            client, models, cases, trials=args.trials, judge=judge,
            concurrency=args.concurrency, max_tokens=args.max_tokens, on_result=progress,
        )

    print("\n" + report(results, models, cases))
    settings = {"models": models, "judge": None if judge is None else args.judge, "trials": args.trials}
    print(f"\nsaved {save(results, settings)}")


if __name__ == "__main__":
    asyncio.run(main())
