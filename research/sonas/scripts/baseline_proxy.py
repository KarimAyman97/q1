"""Proxy baseline (spec §4.2 constraint): stock yolo26n, 15 epochs, same eval path as candidates."""

from __future__ import annotations

import argparse
import json
import pathlib

from research.sonas.coco_eval import evaluate_coco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--device", default=0)
    args = ap.parse_args()
    work = pathlib.Path(args.workdir)
    from ultralytics import YOLO

    model = YOLO("yolo26n.yaml")
    model.train(data="VisDrone.yaml", epochs=15, imgsz=640, device=args.device, seed=0, deterministic=True,
                project=str(work / "baseline"), name="train", exist_ok=True, val=True, plots=False)
    metrics = model.val(data="VisDrone.yaml", imgsz=640, device=args.device, save_json=True,
                        project=str(work / "baseline"), name="val", exist_ok=True, plots=False)
    coco = evaluate_coco(work / "gt.json", pathlib.Path(metrics.save_dir) / "predictions.json")
    (work / "baseline_proxy.json").write_text(json.dumps(coco, indent=2))
    print("Baseline proxy:", coco, "-> pass baseline_map5095 =", coco["map5095"], "to run_search")


if __name__ == "__main__":
    main()
