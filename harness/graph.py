"""Graph orchestration: code decides the order of steps; models do the steps.

In the agent loop the model decides what happens next. That's flexible, but
hard to predict and hard to test. A graph fixes the structure in code: nodes
are steps (a model call, a sub-agent, plain Python), edges say what runs
next, and a conditional edge picks the next node by looking at the state.
That's how a "rewrite until the reviewer approves" loop is built. Models
make the judgment calls inside nodes; the control flow is ordinary code.

Execution runs in supersteps, as in Google's Pregel and LangGraph:
  1. every active node runs at the same time, each on its own copy of the
     state, and returns the keys it wants to change;
  2. those updates are merged into the state;
  3. the nodes their edges point to become the next active set. Two
     branches pointing at the same node join there: it runs once.
Two nodes writing the same key in one step is an error, unless the key has
a reducer (e.g. list concatenation) that says how to combine the writes.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

START = "START"
END = "END"

State = dict[str, Any]
Node = Callable[[State], Awaitable[State | None]]  # returns the keys it changes
Router = Callable[[State], str]                     # returns the next node's name
Reducer = Callable[[Any, Any], Any]                 # (current value, new write) -> merged


class GraphError(Exception):
    """The graph is malformed, or a run broke one of its rules."""


@dataclass
class Step:
    """One superstep, as reported to on_step."""

    index: int
    updates: dict[str, State]  # node -> what it changed, in run order


class Graph:
    def __init__(self, *, reducers: dict[str, Reducer] | None = None, max_steps: int = 25) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, list[str]] = {}
        self._branches: dict[str, tuple[Router, list[str]]] = {}
        self.reducers = reducers or {}
        # A loop whose exit condition never comes true would run forever
        # (and keep paying for model calls). This stops it.
        self.max_steps = max_steps

    def add_node(self, name: str, fn: Node) -> Graph:
        if name in (START, END) or name in self._nodes:
            raise GraphError(f"node name taken: {name}")
        self._nodes[name] = fn
        return self

    def add_edge(self, source: str, *targets: str) -> Graph:
        """After `source`, run all `targets` (several = parallel branches)."""
        self._check_free(source)
        self._edges[source] = list(targets)
        return self

    def add_branch(self, source: str, router: Router, targets: list[str]) -> Graph:
        """After `source`, run whichever of `targets` router(state) names."""
        self._check_free(source)
        self._branches[source] = (router, list(targets))
        return self

    def _check_free(self, source: str) -> None:
        if source in self._edges or source in self._branches:
            raise GraphError(f"{source} already has outgoing edges")

    def validate(self) -> None:
        """Catch wiring mistakes before anything runs (and costs money)."""
        if START not in self._edges:
            raise GraphError("nothing follows START")
        for source, targets in self._targets().items():
            if source != START and source not in self._nodes:
                raise GraphError(f"edge from unknown node: {source}")
            for target in targets:
                if target != END and target not in self._nodes:
                    raise GraphError(f"edge from {source} to unknown node: {target}")
        for name in self._nodes:
            if name not in self._edges and name not in self._branches:
                raise GraphError(f"{name} has no outgoing edge (point it at END to finish)")
        unreachable = set(self._nodes) - self._reachable()
        if unreachable:
            raise GraphError(f"unreachable nodes: {', '.join(sorted(unreachable))}")

    def _targets(self) -> dict[str, list[str]]:
        every = dict(self._edges)
        every.update({source: targets for source, (_, targets) in self._branches.items()})
        return every

    def _reachable(self) -> set[str]:
        seen: set[str] = set()
        todo = [START]
        targets = self._targets()
        while todo:
            for target in targets.get(todo.pop(), []):
                if target != END and target not in seen:
                    seen.add(target)
                    todo.append(target)
        return seen

    async def run(self, state: State, on_step: Callable[[Step], None] | None = None) -> State:
        """Run from START until no node is active. Returns the final state."""
        self.validate()
        state = dict(state)
        active = self._next(START, state)
        for index in range(self.max_steps):
            if not active:
                return state
            updates = await asyncio.gather(*(self._run_node(name, state) for name in active))
            state = self._merge(state, active, updates)
            if on_step is not None:
                on_step(Step(index, dict(zip(active, updates))))
            following: list[str] = []
            for name in active:
                for target in self._next(name, state):
                    if target != END and target not in following:
                        following.append(target)  # a join: runs once
            active = following
        if active:
            raise GraphError(f"stopped after {self.max_steps} steps: does a loop lack an exit?")
        return state

    async def _run_node(self, name: str, state: State) -> State:
        # Each node gets its own copy: parallel nodes can't see or disturb
        # each other's half-finished changes. Only returned updates count.
        update = await self._nodes[name](copy.deepcopy(state))
        if update is None:
            return {}
        if not isinstance(update, dict):
            raise GraphError(f"node {name} must return a dict of updates, not {type(update).__name__}")
        return update

    def _merge(self, state: State, names: list[str], updates: list[State]) -> State:
        merged = dict(state)
        written: dict[str, str] = {}  # key -> node that wrote it this step
        for name, update in zip(names, updates):
            for key, value in update.items():
                if key in self.reducers and key in merged:
                    merged[key] = self.reducers[key](merged[key], value)
                    continue
                if key in written:
                    raise GraphError(
                        f"{written[key]} and {name} both wrote {key!r} in one step; "
                        "give it a reducer to combine them"
                    )
                merged[key] = value
                written[key] = name
        return merged

    def _next(self, name: str, state: State) -> list[str]:
        if name in self._branches:
            router, targets = self._branches[name]
            choice = router(state)
            if choice not in targets:
                raise GraphError(f"router after {name} chose {choice!r}, not one of {targets}")
            return [choice]
        return self._edges.get(name, [])

    def to_mermaid(self) -> str:
        """The graph as a Mermaid diagram (dotted arrows are conditional)."""
        lines = ["graph TD"]
        for source, targets in self._edges.items():
            lines += [f"  {source} --> {target}" for target in targets]
        for source, (_, targets) in self._branches.items():
            lines += [f"  {source} -.-> {target}" for target in targets]
        return "\n".join(lines)
