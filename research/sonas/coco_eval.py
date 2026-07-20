"""VisDrone (YOLO-format labels) -> COCO json, and pycocotools evaluation with AP_small (spec §5)."""

from __future__ import annotations

import contextlib
import io
import json
import pathlib

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def gt_to_coco(images_dir, labels_dir, out_json, class_names) -> pathlib.Path:
    images_dir, labels_dir, out_json = pathlib.Path(images_dir), pathlib.Path(labels_dir), pathlib.Path(out_json)
    images, annotations = [], []
    ann_id = 1
    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        w, h = Image.open(img_path).size
        stem = img_path.stem
        image_id = int(stem) if stem.isnumeric() else stem  # match ultralytics pred_to_json
        images.append({"id": image_id, "file_name": img_path.name, "width": w, "height": h})
        label = labels_dir / f"{stem}.txt"
        if not label.exists():
            continue
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, xc, yc, bw, bh = int(parts[0]), *(float(v) for v in parts[1:5])
            bw_px, bh_px = bw * w, bh * h
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cls + 1,  # 1-based to match ultralytics pred_to_json class_map (val.py:90)
                "bbox": [xc * w - bw_px / 2, yc * h - bh_px / 2, bw_px, bh_px],
                "area": bw_px * bh_px,
                "iscrowd": 0,
            })
            ann_id += 1
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)],
    }))
    return out_json


def evaluate_coco(gt_json, pred_json) -> dict:
    preds = json.loads(pathlib.Path(pred_json).read_text())
    with contextlib.redirect_stdout(io.StringIO()):  # silence pycocotools chatter
        coco_gt = COCO(str(gt_json))
        if not preds:
            return {"map5095": 0.0, "map50": 0.0, "ap_small": 0.0, "ap_medium": 0.0, "ap_large": 0.0}
        coco_dt = coco_gt.loadRes(preds)
        e = COCOeval(coco_gt, coco_dt, "bbox")
        # pycocotools' default 100 truncates dense VisDrone scenes; 300 matches ultralytics val max_det.
        # 100 must stay IN the list too: summarize()'s overall AP (stats[0], our map5095) hardcodes an
        # internal lookup for a maxDets==100 entry, so dropping it would silently turn map5095 into -1.
        e.params.maxDets = [1, 100, 300]
        e.evaluate()
        e.accumulate()
        e.summarize()
    s = e.stats  # [AP, AP50, AP75, APs, APm, APl, ...]
    return {
        "map5095": float(s[0]),
        "map50": float(s[1]),
        "ap_small": float(s[3]),
        "ap_medium": float(s[4]),
        "ap_large": float(s[5]),
    }
