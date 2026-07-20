from research.sonas.genome import Genome
from research.sonas.metrics import is_feasible, measure, repair

SLIM = Genome(p2=False, p5=True, d2=1, d3=1, d4=1, d5=1, head_depth=1, widths=0,
              block="C3Ghost", down="SCDown", fusion="pan", end2end=True, psa=False)
FAT = Genome(p2=True, p5=True, d2=3, d3=3, d4=3, d5=3, head_depth=2, widths=1,
             block="C3k2", down="Conv", fusion="ctx_p3", end2end=True, psa=True)


def test_measure_returns_positive_numbers():
    m = measure(SLIM)
    assert m["params"] > 100_000 and m["gflops"] > 0.1


def test_measure_raises_when_flops_unavailable(monkeypatch):
    import research.sonas.metrics as mod
    import pytest

    monkeypatch.setattr(mod, "get_flops", lambda *a, **k: 0.0)
    with pytest.raises(RuntimeError):
        measure(SLIM)


def test_slim_is_feasible_fat_may_not_be():
    assert is_feasible(measure(SLIM))


def test_repair_returns_feasible_or_none():
    r = repair(FAT)
    assert r is None or is_feasible(measure(r))


def test_repair_keeps_feasible_genome_unchanged():
    assert repair(SLIM) == SLIM
