from evals.fakes import WordEmbedder
from evals.retrieval import NOTES, QUERIES, Score, best, rank_all, sweep
from harness.memory import Memory, MemoryStore


def found(*pairs: tuple[int, float]) -> list[Memory]:
    return [Memory(id_, f"note {id_}", 0.0, score) for id_, score in pairs]


def test_sweep_counts_precision_recall_and_off_topic_hits():
    ranked = [
        found((1, 0.9), (2, 0.7)),  # wants {1}
        found((3, 0.6)),            # wants {3}
        found((4, 0.65)),           # off-topic: wants nothing
    ]
    relevant = [{1}, {3}, set()]
    by_setting = {(s.k, s.cutoff): s for s in sweep(ranked, relevant, [0.62, 0.8], [1, 2])}

    loose = by_setting[(2, 0.62)]  # adds 1, 2 and 4: one right, two wrong; misses 3
    assert (loose.precision, loose.recall, loose.off_topic) == (1 / 3, 1 / 2, 1)
    tight = by_setting[(1, 0.8)]   # adds only 1
    assert (tight.precision, tight.recall, tight.off_topic) == (1.0, 1 / 2, 0)


def test_adding_nothing_is_not_counted_as_imprecise():
    [score] = sweep([found((1, 0.4))], [set()], [0.5], [1])
    assert score.precision == 1.0 and score.off_topic == 0


def test_best_prefers_the_stricter_setting_on_a_tie():
    same = dict(precision=1.0, recall=0.5, off_topic=0)
    assert best([Score(2, 0.6, **same), Score(1, 0.7, **same), Score(1, 0.6, **same)]).cutoff == 0.7


async def test_the_labeled_set_is_consistent(tmp_path):
    assert all(i in range(1, len(NOTES) + 1) for _, wanted in QUERIES for i in wanted)
    store = MemoryStore(tmp_path / "m", WordEmbedder())
    ranked = await rank_all(store)  # every note stored: none counted as a duplicate
    assert len(ranked) == len(QUERIES)
