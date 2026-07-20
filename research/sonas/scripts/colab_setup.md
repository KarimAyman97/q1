# SONAS Colab setup (run top to bottom once per session)

```python
# Cell 1 — Drive for persistence (checkpoints, journals survive disconnects)
from google.colab import drive
drive.mount("/content/drive")
WORK = "/content/drive/MyDrive/sonas"
```

```python
# Cell 2 — install the sonas-paper branch + research deps
!git clone --branch sonas-paper --depth 1 <YOUR-FORK-URL> /content/ultralytics
%cd /content/ultralytics
!pip install -e . -q
!pip install -r research/sonas/requirements.txt -q
```

After the first Colab session: `!pip freeze > requirements-colab.lock` and pin the branch clone above to a
commit SHA (`--branch sonas-paper` -> a fixed SHA) once the branch has commits, so later sessions reproduce
the exact environment instead of drifting with new pushes.

```python
# Cell 3 — dataset (auto-downloads via VisDrone.yaml on first use) + one-time COCO GT
from ultralytics.utils import DATASETS_DIR
from research.sonas.coco_eval import gt_to_coco
from ultralytics.data.utils import check_det_dataset
data = check_det_dataset("VisDrone.yaml")  # triggers download; converter moves files to VisDrone/{images,labels}/val and DELETES the VisDrone2019-DET-* dirs
val_images = str(data["val"])                       # .../VisDrone/images/val
val_labels = val_images.replace("/images/", "/labels/")
gt_to_coco(val_images, val_labels, f"{WORK}/gt.json", list(data["names"].values()))  # names is a dict; converter wants a list
```

```python
# Cell 4 — kill-tests, in spec order. Only the kt3 stages are resume-safe; kt2_smoke and baseline_proxy
# are NOT — a disconnect mid-stage repeats that stage from scratch (~1 GPU-hour each). Rerun the cell
# after any disconnect.
!python -m research.sonas.scripts.kt2_smoke --workdir $WORK
!python -m research.sonas.scripts.baseline_proxy --workdir $WORK
!python -m research.sonas.scripts.kt3_proxy_trust --workdir $WORK --stage short
!python -m research.sonas.scripts.kt3_proxy_trust --workdir $WORK --stage long   # ~40-50 A100-hours total
!python -m research.sonas.scripts.kt3_proxy_trust --workdir $WORK --stage verdict
```
