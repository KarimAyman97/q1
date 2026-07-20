import json

import numpy as np
import pytest

from research.sonas.genome import Genome
from research.sonas.proxy import load_journal, lookup
from research.sonas.search import run_search

CALLS = []


def analytic_evaluate(g: Genome) -> dict:
    """Cheap stand-in: slim models 'see small objects' better; no training."""
    CALLS.append(g)
    v = g.to_vector()
    ap_small = 0.10 + 0.02 * int(g.p2) - 0.005 * (g.d5 - 1)
    return {"key": "-".join(map(str, v)), "genome": g.to_dict(), "params": 1_000_000,
            "gflops": 5.0 + sum(v[2:6]), "status": "ok", "ap_small": ap_small,
            "map5095": ap_small + 0.15}


def test_search_runs_and_returns_front(tmp_path):
    CALLS.clear()
    res = run_search(analytic_evaluate, baseline_map5095=0.20, journal=tmp_path / "evals.ndjson",
                     pop_size=8, n_gen=3, seed=0)
    assert res.F is not None and len(res.F) >= 1
    assert len(load_journal(tmp_path / "evals.ndjson")) == len(CALLS)


def test_search_resumes_from_journal_cache(tmp_path):
    CALLS.clear()
    journal = tmp_path / "evals.ndjson"
    run_search(analytic_evaluate, baseline_map5095=0.20, journal=journal, pop_size=8, n_gen=2, seed=0)
    n_first = len(CALLS)
    assert n_first > 0
    run_search(analytic_evaluate, baseline_map5095=0.20, journal=journal, pop_size=8, n_gen=2, seed=0)
    # same seed -> identical deterministic trajectory -> every genome cache-hits
    assert len(CALLS) == n_first


def test_res_x_rows_are_repaired_genomes_with_journal_records(tmp_path):
    """res.X must carry REPAIRED genomes: every row decodes to a genome the journal actually evaluated."""
    CALLS.clear()
    journal = tmp_path / "evals.ndjson"
    res = run_search(analytic_evaluate, baseline_map5095=0.20, journal=journal, pop_size=8, n_gen=3, seed=0)
    records = load_journal(journal)
    assert all(lookup(records, Genome.from_vector(np.rint(x).astype(int))) is not None
               for x in np.atleast_2d(res.X))


def test_invalid_baseline_map5095_raises(tmp_path):
    with pytest.raises(ValueError):
        run_search(analytic_evaluate, baseline_map5095=20.5, journal=tmp_path / "evals.ndjson",
                   pop_size=4, n_gen=1, seed=0)


def test_settings_mismatch_raises(tmp_path):
    journal = tmp_path / "evals.ndjson"
    journal.write_text(json.dumps({"key": "x", "genome": {}, "status": "ok", "epochs": 15}) + "\n")
    with pytest.raises(ValueError):
        run_search(analytic_evaluate, baseline_map5095=0.20, journal=journal, pop_size=4, n_gen=1,
                   settings={"epochs": 25})
