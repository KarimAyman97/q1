"""NSGA-II over the SONAS genome space with journal-backed caching (spec §4.2)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

from research.sonas.genome import GENE_CARDINALITIES, Genome
from research.sonas.metrics import repair as shrink_repair
from research.sonas.proxy import append_eval, load_journal, lookup


@lru_cache(maxsize=None)
def _repair_cached(g: Genome):
    return shrink_repair(g)


class ShrinkRepair(Repair):
    """Population-level pymoo Repair so every genome pymoo tracks (res.X) is the one actually evaluated."""

    def _do(self, problem, X, **kwargs):
        rows = []
        for row in np.asarray(X):
            g = Genome.from_vector(np.rint(row).astype(int))
            r = _repair_cached(g)
            rows.append((r if r is not None else g).to_vector())  # unrepairable stays raw; penalized in _evaluate
        return np.array(rows)


def _settings_mismatch(rec: dict, settings: dict) -> bool:
    """True if `rec` was journaled under different values for any field `settings` cares about."""
    return any(k in rec and rec[k] != v for k, v in settings.items())


class SonasProblem(ElementwiseProblem):
    def __init__(self, evaluate, baseline_map5095: float, journal, settings: dict | None = None):
        super().__init__(n_var=len(GENE_CARDINALITIES), n_obj=2, n_ieq_constr=1,
                         xl=np.zeros(len(GENE_CARDINALITIES)),
                         xu=np.array(GENE_CARDINALITIES) - 1, vtype=int)
        self.evaluate_genome = evaluate
        self.baseline = baseline_map5095
        self.journal = journal
        self.settings = settings

    def _evaluate(self, x, out, *args, **kwargs):
        g = Genome.from_vector(np.rint(x).astype(int))
        # population is pre-repaired by ShrinkRepair, so this is a cache hit that also re-derives the
        # feasibility signal (None) for genomes ShrinkRepair couldn't fix and left raw.
        repaired = _repair_cached(g)
        if repaired is None:  # provably can't fit the budget — never spend a training run on it
            out["F"] = [1.0, 1e3]  # dominated by everything real
            out["G"] = [1.0]
            return
        g = repaired
        records = load_journal(self.journal)
        if self.settings:  # a record journaled under other settings (epochs, imgsz, seed...) isn't a cache hit here
            records = [r for r in records if not _settings_mismatch(r, self.settings)]
        rec = lookup(records, g)
        if rec is None:
            rec = self.evaluate_genome(g)
            append_eval(self.journal, rec)
        if rec["status"] != "ok":
            out["F"] = [1.0, 1e3]
            out["G"] = [1.0]
            return
        out["F"] = [-rec["ap_small"], rec["gflops"]]
        # mAP flows through this codebase as a 0-1 fraction (pycocotools), so the spec's
        # "-1.0 percentage point" tolerance is 0.01 here. Feasible when G <= 0.
        out["G"] = [(self.baseline - 0.01) - rec["map5095"]]


def run_search(evaluate, *, baseline_map5095: float, journal, pop_size: int = 20, n_gen: int = 8, seed: int = 0,
               settings: dict | None = None):
    if not 0 < baseline_map5095 < 1:
        raise ValueError(f"baseline_map5095 must be a 0-1 fraction, got {baseline_map5095!r}")
    if settings:
        for rec in load_journal(journal):
            if _settings_mismatch(rec, settings):
                raise ValueError(f"journal {journal} has records journaled under different settings than {settings}: "
                                  f"{rec}")
    problem = SonasProblem(evaluate, baseline_map5095, journal, settings=settings)
    algo = NSGA2(pop_size=pop_size, sampling=IntegerRandomSampling(),
                 crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
                 mutation=PM(prob=0.3, eta=20, vtype=float, repair=RoundingRepair()),
                 repair=ShrinkRepair(),
                 eliminate_duplicates=True)
    return minimize(problem, algo, ("n_gen", n_gen), seed=seed, verbose=False, save_history=True)
