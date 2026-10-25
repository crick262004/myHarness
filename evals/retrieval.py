"""Memory retrieval eval: is the auto-recall cutoff in the right place?

A labeled set: saved notes, and queries with the notes that should come back
(none, for off-topic questions). The notes go into a real MemoryStore with
the real embedding model; each query is searched once, then every setting
(k = most notes added, and the score cutoff) is scored on those results:

  precision  of the notes that would be added, how many were relevant
  recall     of the relevant notes, how many would be added
  off-topic  off-topic questions that would get any note at all

Notes are added to a message without anyone asking, so a wrong one costs
tokens and can mislead: the choice favours precision (F0.5 weighs it twice
as much as recall).

    uv run python -m evals.retrieval
    uv run python -m evals.retrieval --models BAAI/bge-small-en-v1.5,BAAI/bge-base-en-v1.5

The set is small and the settings were tuned on it, so treat the numbers as
a sanity check that catches a bad setting, not as a benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path

from harness.main import HOME
from harness.memory import AUTO_RECALL_K, TOP_K, LocalEmbedder, Memory, MemoryStore

NOTES = [
    "User prefers neovim as their code editor.",                                      # 1
    "User indents Python with tabs, never spaces.",                                   # 2
    "User's name is Arnav.",                                                          # 3
    "User works at Pure Storage on NVMe and Fibre Channel storage drivers.",          # 4
    "MyHarness tests run with `uv run pytest`.",                                      # 5
    "User wants very brief answers with no fluff.",                                   # 6
    "User's laptop is behind Zscaler; HTTPS tools need SSL_CERT_FILE set.",           # 7
    "User is preparing for job interviews about AI agents.",                          # 8
    "At work, deploys happen on Thursdays and never on Fridays.",                     # 9
    "User prefers dark mode in every app.",                                           # 10
    "User's main languages are C++ and Python.",                                      # 11
    "The MyHarness plan is in ~/.claude/plans/polymorphic-popping-naur.md.",          # 12
    "User is vegetarian.",                                                            # 13
    "User's manager is Priya.",                                                       # 14
    "User doesn't like being quizzed; explain concepts instead.",                     # 15
    "User's personal knowledge base is at ~/work-kb.",                                # 16
]

# (query, ids of the notes that should come back)
QUERIES: list[tuple[str, set[int]]] = [
    ("Which editor should I open this file in?", {1}),
    ("Format this Python function for me.", {2}),
    ("What's my name?", {3}),
    ("Explain how NVMe submission and completion queues work.", {4}),
    ("How do I run the MyHarness test suite?", {5}),
    ("Write me a detailed essay on Rust's borrow checker.", {6}),
    ("pip install fails with CERTIFICATE_VERIFY_FAILED.", {7}),
    ("curl to the API gives an SSL certificate error.", {7}),
    ("Help me get ready for my interview.", {8, 15}),
    ("Can we ship this release on Friday?", {9}),
    ("Pick a color scheme for my terminal.", {10}),
    ("Which language should I write this tool in?", {11}),
    ("Where is the MyHarness project plan?", {12}),
    ("Suggest something for dinner tonight.", {13}),
    ("Who do I report to?", {14}),
    ("Quiz me on Python decorators.", {15}),
    ("Where did I put my notes on Fibre Channel?", {4, 16}),
    # Off-topic: nothing saved is relevant, so nothing should be added.
    ("What's the capital of France?", set()),
    ("Convert 50 miles to kilometers.", set()),
    ("What year did the Berlin Wall fall?", set()),
    ("Tell me a joke about cats.", set()),
    ("How does photosynthesis work?", set()),
    ("Summarize the plot of Hamlet.", set()),
]

CUTOFFS = [round(0.45 + 0.025 * i, 3) for i in range(15)]  # 0.45 .. 0.80
KS = [1, 2, 3, TOP_K]  # most notes added per message


@dataclass
class Score:
    k: int
    cutoff: float
    precision: float
    recall: float
    off_topic: int  # off-topic queries that got at least one note

    @property
    def f05(self) -> float:
        p, r = self.precision, self.recall
        return 1.25 * p * r / (0.25 * p + r) if p + r else 0.0


def sweep(
    ranked: list[list[Memory]], relevant: list[set[int]], cutoffs: list[float], ks: list[int]
) -> list[Score]:
    """Score each (k, cutoff) on already-ranked search results (best first)."""
    total_relevant = sum(len(r) for r in relevant)
    scores = []
    for k in ks:
        for cutoff in cutoffs:
            added = hits = off_topic = 0
            for found, wanted in zip(ranked, relevant):
                kept = [m.id for m in found[:k] if m.score >= cutoff]
                added += len(kept)
                hits += len(set(kept) & wanted)
                off_topic += not wanted and bool(kept)
            scores.append(Score(
                k, cutoff,
                precision=hits / added if added else 1.0,  # added nothing: nothing wrong
                recall=hits / total_relevant if total_relevant else 1.0,
                off_topic=off_topic,
            ))
    return scores


def best(scores: list[Score]) -> Score:
    """Highest F0.5; on a tie, the stricter setting (higher cutoff, fewer notes)."""
    return max(scores, key=lambda s: (round(s.f05, 4), s.cutoff, -s.k))


async def rank_all(store: MemoryStore) -> list[list[Memory]]:
    for note in NOTES:
        await store.add(note)
    assert store.count() == len(NOTES), "two notes were close enough to count as duplicates"
    return [await store.search(query, k=TOP_K, min_score=-1) for query, _ in QUERIES]


async def evaluate(model: str | None) -> None:
    embedder = LocalEmbedder(cache_dir=HOME / "models", **({"model": model} if model else {}))
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "memory.db", embedder)
        ranked = await rank_all(store)
        store.close()
    scores = sweep(ranked, [wanted for _, wanted in QUERIES], CUTOFFS, KS)
    chosen = best(scores)
    off_topic_total = sum(not wanted for _, wanted in QUERIES)
    print(f"{embedder.name} · {len(NOTES)} notes · {len(QUERIES)} queries\n")
    print(f"{'k':>2} {'cutoff':>7} {'precision':>10} {'recall':>7} {'F0.5':>6} {'off-topic':>10}")
    for s in scores:
        current = s.k == AUTO_RECALL_K and s.cutoff == embedder.min_score
        if s.k != chosen.k and not current:
            continue  # the full table only for the best k, plus the setting in use
        mark = " ←" if s is chosen else ""
        mark += "  (current)" if current else ""
        print(f"{s.k:>2} {s.cutoff:>7.3f} {s.precision:>10.0%} {s.recall:>7.0%} {s.f05:>6.2f} "
              f"{s.off_topic:>6}/{off_topic_total}{mark}")
    print("\nbest per k: " + " · ".join(
        f"k={k}: {(b := best([s for s in scores if s.k == k])).cutoff} (F0.5 {b.f05:.2f})" for k in KS
    ))
    print(f"best overall: k={chosen.k}, cutoff {chosen.cutoff}, "
          f"precision {chosen.precision:.0%}, recall {chosen.recall:.0%}\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", help="comma-separated fastembed model names (default: the one in use)")
    args = parser.parse_args()
    for model in (args.models.split(",") if args.models else [None]):
        await evaluate(model)


if __name__ == "__main__":
    asyncio.run(main())
