import asyncio
import operator

import pytest

from harness.graph import END, START, Graph, GraphError, State, Step


def setter(**update):
    async def node(state: State) -> State:
        return dict(update)
    return node


async def test_state_flows_along_edges():
    async def double(state: State) -> State:
        return {"n": state["n"] * 2}

    graph = (
        Graph()
        .add_node("double", double)
        .add_node("label", setter(done=True))
        .add_edge(START, "double").add_edge("double", "label").add_edge("label", END)
    )
    assert await graph.run({"n": 3}) == {"n": 6, "done": True}


async def test_a_conditional_edge_loops_until_its_exit():
    async def bump(state: State) -> State:
        return {"n": state["n"] + 1}

    graph = (
        Graph()
        .add_node("bump", bump)
        .add_edge(START, "bump")
        .add_branch("bump", lambda s: END if s["n"] >= 3 else "bump", ["bump", END])
    )
    steps: list[Step] = []
    assert (await graph.run({"n": 0}, on_step=steps.append))["n"] == 3
    assert [list(s.updates) for s in steps] == [["bump"]] * 3


async def test_parallel_branches_run_together_and_join_once():
    running = peak = 0
    joins = []

    def slow(key):
        async def node(state: State) -> State:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.02)
            running -= 1
            return {key: True}
        return node

    async def join(state: State) -> State:
        joins.append(dict(state))
        return None

    graph = (
        Graph()
        .add_node("a", slow("a")).add_node("b", slow("b")).add_node("join", join)
        .add_edge(START, "a", "b")
        .add_edge("a", "join").add_edge("b", "join").add_edge("join", END)
    )
    await graph.run({})

    assert peak == 2
    assert joins == [{"a": True, "b": True}]  # ran once, after both, seeing both


async def test_two_writes_to_one_key_need_a_reducer():
    def fan(reducers):
        return (
            Graph(reducers=reducers)
            .add_node("a", setter(log=["a"])).add_node("b", setter(log=["b"]))
            .add_edge(START, "a", "b").add_edge("a", END).add_edge("b", END)
        )

    with pytest.raises(GraphError, match="both wrote 'log'"):
        await fan({}).run({})
    assert (await fan({"log": operator.add}).run({"log": ["start"]}))["log"] == ["start", "a", "b"]


async def test_a_runaway_loop_is_stopped():
    graph = Graph(max_steps=5).add_node("spin", setter()).add_edge(START, "spin")
    graph.add_branch("spin", lambda s: "spin", ["spin", END])
    with pytest.raises(GraphError, match="stopped after 5 steps"):
        await graph.run({})


async def test_nodes_change_state_only_through_their_return_value():
    async def meddle(state: State) -> State:
        state["items"].append("sneaky")  # edits its own copy
        return {"seen": len(state["items"])}

    graph = Graph().add_node("m", meddle).add_edge(START, "m").add_edge("m", END)
    assert await graph.run({"items": ["x"]}) == {"items": ["x"], "seen": 2}


@pytest.mark.parametrize("wire, problem", [
    (lambda g: g, "nothing follows START"),
    (lambda g: g.add_edge(START, "a").add_edge("a", "ghost"), "unknown node: ghost"),
    (lambda g: g.add_edge(START, "a"), "a has no outgoing edge"),
    (lambda g: g.add_edge(START, END).add_edge("a", END), "unreachable nodes: a"),
])
def test_wiring_mistakes_are_caught_before_running(wire, problem):
    graph = wire(Graph().add_node("a", setter()))
    with pytest.raises(GraphError, match=problem):
        graph.validate()


async def test_a_router_must_pick_a_declared_target():
    graph = Graph().add_node("a", setter()).add_edge(START, "a")
    graph.add_branch("a", lambda s: "elsewhere", [END])
    with pytest.raises(GraphError, match="chose 'elsewhere'"):
        await graph.run({})


def test_mermaid_diagram():
    graph = (
        Graph().add_node("a", setter()).add_edge(START, "a")
        .add_branch("a", lambda s: END, ["a", END])
    )
    assert graph.to_mermaid() == "graph TD\n  START --> a\n  a -.-> a\n  a -.-> END"
