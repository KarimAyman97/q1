"""Gradient-boosted surrogate over the genome space (spec §4.2 'light surrogate')."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from research.sonas.genome import Genome


class Surrogate:
    def __init__(self, ap_model, fl_model):
        self.ap_model = ap_model
        self.fl_model = fl_model

    def screen(self, vectors, keep: int) -> list:
        X = np.asarray(vectors, dtype=float)
        ap = self.ap_model.predict(X)
        fl = self.fl_model.predict(X)
        # Non-dominated sorting on predictions; fill by rank, break ties by predicted ap_small.
        order = sorted(range(len(X)), key=lambda i: (_dom_rank(i, ap, fl), -ap[i]))
        return order[:keep]


def _dom_rank(i, ap, fl) -> int:
    return sum(1 for j in range(len(ap)) if ap[j] >= ap[i] and fl[j] <= fl[i] and (ap[j] > ap[i] or fl[j] < fl[i]))


def fit_surrogate(records, min_records: int = 40):
    ok = [r for r in records if r.get("status") == "ok"]
    if len(ok) < min_records:
        return None
    X = np.array([Genome.from_dict(r["genome"]).to_vector() for r in ok], dtype=float)
    ap = GradientBoostingRegressor(random_state=0).fit(X, [r["ap_small"] for r in ok])
    fl = GradientBoostingRegressor(random_state=0).fit(X, [r["gflops"] for r in ok])
    return Surrogate(ap, fl)
