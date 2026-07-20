"""Proxy evaluation: 15-epoch VisDrone run -> objectives, journaled to NDJSON (spec §4.2)."""

from __future__ import annotations

import json
import pathlib

from research.sonas.coco_eval import evaluate_coco
from research.sonas.genome import Genome
from research.sonas.metrics import measure
from research.sonas.yamlgen import write_model_yaml


def _key(g: Genome) -> str:
    return "-".join(str(x) for x in g.to_vector())


def _real_trainer(model_yaml, data, epochs, imgsz, device, seed, project, gt_json):
    """Train + val + COCOeval on GPU. Only exercised on Colab; tests inject a fake."""
    from ultralytics import YOLO

    model = YOLO(str(model_yaml))
    model.train(data=data, epochs=epochs, imgsz=imgsz, device=device, seed=seed, deterministic=True,
                project=str(project), name="train", exist_ok=True, val=True, plots=False)
    metrics = model.val(data=data, imgsz=imgsz, device=device, save_json=True,
                        project=str(project), name="val", exist_ok=True, plots=False)
    pred_json = pathlib.Path(metrics.save_dir) / "predictions.json"
    coco = evaluate_coco(gt_json, pred_json)
    return {"ap_small": coco["ap_small"], "map5095": coco["map5095"]}


def run_proxy(g: Genome, *, data, gt_json, project, epochs: int = 15, imgsz: int = 640,
              device=0, seed: int = 0, trainer=None) -> dict:
    project = pathlib.Path(project)
    project.mkdir(parents=True, exist_ok=True)
    trainer = trainer or _real_trainer
    m = measure(g)
    rec = {"key": _key(g), "genome": g.to_dict(), "params": m["params"], "gflops": m["gflops"],
           "epochs": epochs, "imgsz": imgsz, "seed": seed}
    model_yaml = write_model_yaml(g, project / f"{rec['key']}.yaml", nc=10)
    try:
        out = trainer(model_yaml, data, epochs, imgsz, device, seed, project, gt_json)
        rec.update(status="ok", ap_small=float(out["ap_small"]), map5095=float(out["map5095"]))
    except Exception as e:  # a failed candidate must not kill a 3-day search
        rec.update(status="error", error=str(e), ap_small=0.0, map5095=0.0)
    return rec


def append_eval(journal, rec: dict) -> None:
    journal = pathlib.Path(journal)
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def load_journal(journal) -> list:
    journal = pathlib.Path(journal)
    if not journal.exists():
        return []
    lines = [ln for ln in journal.read_text().splitlines() if ln.strip()]
    records = []
    for i, ln in enumerate(lines):
        try:
            records.append(json.loads(ln))
        except json.JSONDecodeError:
            if i == len(lines) - 1:  # torn tail from a hard kill — drop it, the genome just re-evaluates
                print(f"load_journal: dropping truncated final line in {journal}")
                break
            raise
    return records


def lookup(records: list, g: Genome):
    """First `status == "ok"` record for `g`; an errored genome is not a cache hit and re-evaluates on resume."""
    key = _key(g)
    for rec in records:
        if rec["key"] == key and rec["status"] == "ok":
            return rec
    return None
