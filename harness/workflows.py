"""A research workflow built on the graph engine.

    START → plan → research ─┬→ write → review ─(approved, or out of drafts)→ END
                 → memory  ──┘    ↑        │
                                  └────────┘ (changes requested)

  plan      splits the question into independent sub-questions (structured output)
  research  one read-only sub-agent per sub-question, all at the same time
  memory    looks up saved notes about the user, in parallel with research
  write     drafts the answer from the findings and notes; on a rewrite, also
            from the previous draft and the reviewer's feedback
  review    a model acting as judge: approve, or say exactly what to fix

Compared with letting one agent do all of it in its loop, the steps, their
order and the number of rewrites are fixed in code, so a run is predictable
and each node can be tested with a scripted model.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from .agent import Agent, TokenUsage
from .graph import END, START, Graph, State, Step
from .llm import ask_json, ask_text
from .memory import MemoryStore
from .subagents import run_subagent

log = logging.getLogger(__name__)

MAX_SUBQUESTIONS = 4
MAX_DRAFTS = 3  # the first draft plus up to two rewrites

PLANNER = """\
You plan research. Split the user's question into 1-4 sub-questions that \
can be researched independently and at the same time. Each one goes to a \
separate researcher who sees only that sub-question, so make each \
self-contained: name the files, systems or terms involved. If the question \
is simple, return it as the only sub-question."""

WRITER = """\
You write the final answer to the user's question from research findings. \
Use only what the findings support, and say plainly what they leave \
unknown. Keep exact paths, names and figures. Follow any notes about the \
user's preferences. Be direct and concise."""

REVIEWER = """\
You review a draft answer before it reaches the user. Approve it if it \
answers the question, every claim is backed by the findings, and nothing \
important in the findings is missing. Otherwise give specific, actionable \
feedback: what is wrong or missing, and how to fix it. Don't ask for more \
research; judge the draft against the findings given."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {"subquestions": {"type": "array", "items": {"type": "string"}}},
    "required": ["subquestions"],
    "additionalProperties": False,
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {"approved": {"type": "boolean"}, "feedback": {"type": "string"}},
    "required": ["approved", "feedback"],
    "additionalProperties": False,
}


def build_research_graph(
    agent: Agent,
    subagent_system: str,
    *,
    subagent_model: str | None = None,
    memory: MemoryStore | None = None,
) -> Graph:
    """The workflow, wired to `agent`: its client, model, tools, policy and
    token bill. Run it with {"question": ...}."""
    client, model = agent.client, agent.model

    async def plan(state: State) -> State:
        data, usage = await ask_json(
            client, model=model, system=PLANNER, prompt=state["question"], schema=PLAN_SCHEMA
        )
        agent.bill(TokenUsage.of(usage))
        subquestions = [q.strip() for q in data["subquestions"] if q.strip()]
        return {"subquestions": subquestions[:MAX_SUBQUESTIONS] or [state["question"]]}

    async def research(state: State) -> State:
        questions = state["subquestions"]
        reports = await asyncio.gather(
            *(
                run_subagent(agent, subagent_system, q, f"research {i}", subagent_model)
                for i, q in enumerate(questions, 1)
            ),
            return_exceptions=True,  # one failed researcher shouldn't sink the others
        )
        findings = []
        for question, report in zip(questions, reports):
            if isinstance(report, BaseException) and not isinstance(report, Exception):
                raise report  # Ctrl-C and cancellation still stop the run
            if isinstance(report, Exception):
                report = f"(research failed: {type(report).__name__}: {report})"
            findings.append({"question": question, "report": report})
        return {"findings": findings}

    async def recall(state: State) -> State:
        if memory is None:
            return {"notes": []}
        try:
            found = await memory.search(state["question"])
        except Exception as exc:  # noqa: BLE001 — as in the agent: memory never blocks a run
            log.warning("memory search failed: %s", exc)
            found = []
        return {"notes": [m.text for m in found]}

    async def write(state: State) -> State:
        parts = [f"Question: {state['question']}", _findings_text(state["findings"])]
        if state["notes"]:
            parts.append("Notes about the user:\n" + "\n".join(f"- {n}" for n in state["notes"]))
        review = state.get("review")
        if review and not review["approved"]:
            parts.append(f"Your previous draft:\n{state['draft']}")
            parts.append(f"Reviewer feedback to address:\n{review['feedback']}")
        draft, usage = await ask_text(client, model=model, system=WRITER, prompt="\n\n".join(parts))
        agent.bill(TokenUsage.of(usage))
        return {"draft": draft, "drafts": state.get("drafts", 0) + 1}

    async def review(state: State) -> State:
        prompt = "\n\n".join([
            f"Question: {state['question']}",
            _findings_text(state["findings"]),
            f"Draft answer:\n{state['draft']}",
        ])
        verdict, usage = await ask_json(
            client, model=model, system=REVIEWER, prompt=prompt, schema=REVIEW_SCHEMA
        )
        agent.bill(TokenUsage.of(usage))
        return {"review": verdict}

    def after_review(state: State) -> str:
        if state["review"]["approved"] or state["drafts"] >= MAX_DRAFTS:
            return END
        return "write"

    return (
        Graph()
        .add_node("plan", plan)
        .add_node("research", research)
        .add_node("memory", recall)
        .add_node("write", write)
        .add_node("review", review)
        .add_edge(START, "plan")
        .add_edge("plan", "research", "memory")  # parallel branches...
        .add_edge("research", "write")
        .add_edge("memory", "write")             # ...that join at write
        .add_edge("write", "review")
        .add_branch("review", after_review, ["write", END])
    )


def _findings_text(findings: list[dict[str, str]]) -> str:
    return "Research findings:\n\n" + "\n\n".join(
        f"## {f['question']}\n{f['report']}" for f in findings
    )


def _count(items: list, word: str) -> str:
    return f"{len(items)} {word}{'' if len(items) == 1 else 's'}"


def describe_step(step: Step) -> str:
    """One line per superstep for the console."""
    parts = []
    for node, update in step.updates.items():
        if node == "plan":
            parts.append(f"plan → {_count(update['subquestions'], 'sub-question')}")
        elif node == "research":
            parts.append(f"research → {_count(update['findings'], 'report')}")
        elif node == "memory":
            parts.append(f"memory → {_count(update['notes'], 'note')}")
        elif node == "write":
            parts.append(f"write → draft {update['drafts']}")
        elif node == "review":
            verdict = update["review"]
            parts.append("review → approved" if verdict["approved"]
                         else f"review → changes requested: {verdict['feedback'][:120]}")
        else:
            parts.append(node)
    return " ∥ ".join(parts)


async def run_research(
    agent: Agent,
    question: str,
    subagent_system: str,
    *,
    subagent_model: str | None = None,
    memory: MemoryStore | None = None,
    on_step: Callable[[Step], None] | None = None,
) -> State:
    """Run the workflow; the final state holds "draft" (the answer) and "review"."""
    graph = build_research_graph(
        agent, subagent_system, subagent_model=subagent_model, memory=memory
    )
    return await graph.run({"question": question}, on_step=on_step)
