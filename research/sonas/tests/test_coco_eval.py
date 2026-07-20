import json

from PIL import Image

from research.sonas.coco_eval import evaluate_coco, gt_to_coco


def make_dataset(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for stem, boxes in [("img_a", [(0, 0.5, 0.5, 0.02, 0.02)]), ("img_b", [(1, 0.25, 0.25, 0.4, 0.4)])]:
        Image.new("RGB", (640, 640)).save(tmp_path / "images" / f"{stem}.jpg")
        lines = [" ".join(str(v) for v in b) for b in boxes]
        (tmp_path / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    return tmp_path


def test_gt_to_coco_areas_and_ids(tmp_path):
    root = make_dataset(tmp_path)
    p = gt_to_coco(root / "images", root / "labels", tmp_path / "gt.json", ["ped", "car"])
    gt = json.loads(p.read_text())
    assert {im["id"] for im in gt["images"]} == {"img_a", "img_b"}
    small = next(a for a in gt["annotations"] if a["image_id"] == "img_a")
    assert small["area"] < 32 * 32  # 0.02*640 = 12.8 px box -> COCO 'small'
    assert {c["id"] for c in gt["categories"]} == {1, 2}


def test_perfect_predictions_score_ap_small_1(tmp_path):
    root = make_dataset(tmp_path)
    gt_path = gt_to_coco(root / "images", root / "labels", tmp_path / "gt.json", ["ped", "car"])
    gt = json.loads(gt_path.read_text())
    preds = [
        {"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 0.9}
        for a in gt["annotations"]
    ]
    (tmp_path / "pred.json").write_text(json.dumps(preds))
    r = evaluate_coco(gt_path, tmp_path / "pred.json")
    assert r["ap_small"] > 0.99 and r["map5095"] > 0.99


def test_empty_predictions_score_zero(tmp_path):
    root = make_dataset(tmp_path)
    gt_path = gt_to_coco(root / "images", root / "labels", tmp_path / "gt.json", ["ped", "car"])
    (tmp_path / "pred.json").write_text(json.dumps([]))
    r = evaluate_coco(gt_path, tmp_path / "pred.json")
    assert r["map5095"] == 0.0
