import random

from research.sonas.genome import Genome
from research.sonas.surrogate import fit_surrogate


def synthetic_records(n):
    rng = random.Random(0)
    out = []
    for _ in range(n):
        g = Genome.random(rng)
        v = g.to_vector()
        out.append({"key": "k", "genome": g.to_dict(), "status": "ok",
                    "ap_small": 0.1 + 0.03 * v[0] - 0.004 * v[5], "gflops": 4.0 + sum(v[2:6]),
                    "map5095": 0.3, "params": 1})
    return out


def test_needs_min_records():
    assert fit_surrogate(synthetic_records(10)) is None


def test_screen_prefers_predicted_good_candidates():
    s = fit_surrogate(synthetic_records(120))
    rng = random.Random(1)
    genomes = [Genome.random(rng) for _ in range(40)]
    kept = s.screen([g.to_vector() for g in genomes], keep=10)
    assert len(kept) == 10
    kept_p2 = sum(genomes[i].p2 for i in kept)
    rest_p2 = sum(g.p2 for g in genomes) - kept_p2
    assert kept_p2 / 10 >= rest_p2 / 30  # p2 raises ap_small in the synthetic world
