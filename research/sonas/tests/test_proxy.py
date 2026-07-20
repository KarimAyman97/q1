import json
import random

import pytest

from research.sonas.genome import Genome
from research.sonas.proxy import append_eval, load_journal, lookup, run_proxy


def fake_trainer(model_yaml, data, epochs, imgsz, device, seed, project, gt_json):
    return {"ap_small": 0.11, "map5095": 0.22}  # stands in for train+val+COCOeval


def test_run_proxy_journals_metrics(tmp_path):
    g = Genome.random(random.Random(0))
    rec = run_proxy(g, data="VisDrone.yaml", gt_json=tmp_path / "gt.json",
                    project=tmp_path, trainer=fake_trainer)
    assert rec["status"] == "ok" and rec["ap_small"] == 0.11
    assert rec["params"] > 0 and rec["gflops"] > 0
    journal = tmp_path / "evals.ndjson"
    append_eval(journal, rec)
    records = load_journal(journal)
    assert lookup(records, g)["map5095"] == 0.22


def test_lookup_misses_unknown_genome(tmp_path):
    g1, g2 = Genome.random(random.Random(1)), Genome.random(random.Random(2))
    assert g1 != g2
    rec = run_proxy(g1, data="d", gt_json=tmp_path / "gt.json", project=tmp_path, trainer=fake_trainer)
    journal = tmp_path / "evals.ndjson"
    append_eval(journal, rec)
    assert lookup(load_journal(journal), g2) is None


def test_trainer_failure_is_journaled_not_raised(tmp_path):
    def boom(*a, **k):
        raise RuntimeError("CUDA OOM")

    g = Genome.random(random.Random(3))
    rec = run_proxy(g, data="d", gt_json=tmp_path / "gt.json", project=tmp_path, trainer=boom)
    assert rec["status"] == "error" and "CUDA OOM" in rec["error"]


def test_load_journal_drops_truncated_final_line(tmp_path):
    journal = tmp_path / "evals.ndjson"
    good = [json.dumps({"key": "a", "status": "ok"}), json.dumps({"key": "b", "status": "ok"})]
    journal.write_text("\n".join(good) + '\n{"key": "c", "status": "ok"')  # torn tail, no closing brace
    records = load_journal(journal)
    assert len(records) == 2
    assert [r["key"] for r in records] == ["a", "b"]


def test_load_journal_raises_on_malformed_interior_line(tmp_path):
    journal = tmp_path / "evals.ndjson"
    journal.write_text('{"key": "a", "status": "ok"\n{"key": "b", "status": "ok"}\n')  # first line is broken
    with pytest.raises(json.JSONDecodeError):
        load_journal(journal)


def test_lookup_skips_error_record_until_ok_record_appended(tmp_path):
    def boom(*a, **k):
        raise RuntimeError("boom")

    g = Genome.random(random.Random(4))
    journal = tmp_path / "evals.ndjson"
    err_rec = run_proxy(g, data="d", gt_json=tmp_path / "gt.json", project=tmp_path, trainer=boom)
    append_eval(journal, err_rec)
    assert lookup(load_journal(journal), g) is None  # an error record must not permanently poison the cache

    ok_rec = run_proxy(g, data="d", gt_json=tmp_path / "gt.json", project=tmp_path, trainer=fake_trainer)
    append_eval(journal, ok_rec)
    found = lookup(load_journal(journal), g)
    assert found is not None and found["status"] == "ok"
