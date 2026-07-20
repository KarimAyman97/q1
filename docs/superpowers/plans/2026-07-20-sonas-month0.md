# SONAS Month-0/1 Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the SONAS search infrastructure — genome codec, YAML generator, constraint metrics, VisDrone COCO evaluation, proxy-run harness, NSGA-II driver, surrogate, and the kill-test scripts — so the architecture search can start the moment the proxy-trust kill-test passes.

**Architecture:** A self-contained `research/sonas/` package on branch `sonas-paper` (worktree `/home/karim/Documents/Master/Q1/ultralytics-sonas`). A 13-gene integer genome decodes deterministically into an ultralytics model-YAML dict; `DetectionModel` builds it; `get_flops` + param count gate feasibility; a proxy harness trains 15-epoch VisDrone runs and journals every evaluation to NDJSON; a pymoo NSGA-II problem consumes the journal (cache + resume); an optional GBT surrogate pre-screens offspring. Long GPU work runs on Colab via scripts; everything else is locally unit-tested.

**Tech Stack:** Python ≥3.9 (research code only — the repo's 3.8 floor applies to the shipped package, not to `research/`), PyTorch + this ultralytics checkout (editable install), pymoo ≥0.6, pycocotools, scikit-learn, scipy, thop.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-20-sonas-design.md` — this plan implements its §4 (method), §4.3 kill-tests 1–3, and §9 (engineering).
- All new files live under `research/sonas/` — **zero modifications to `ultralytics/`** in this plan (reuse only).
- Branch `sonas-paper` in the `ultralytics-sonas` worktree; never commit to `main`; never push to `main`.
- Hard constraints from spec §4.1: params ≤ 5,000,000; GFLOPs ≤ 12.0 (at 640).
- Objectives from spec §4.2: maximize AP_small, minimize GFLOPs; constraint mAP50-95 ≥ (proxy baseline − 1.0), proxy-to-proxy.
- Proxy = 15-epoch VisDrone at 640, fixed seed. Proxy-trust gate: Kendall τ ≥ 0.6 (spec §4.3.3).
- Style: ruff, line length 120 (repo `pyproject.toml`). No license headers by hand (the Actions bot owns those; research files may simply omit them).
- Run tests from the worktree root: `cd /home/karim/Documents/Master/Q1/ultralytics-sonas`.
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 0: Step-zero novelty check (research action — BLOCKING, no code)

**Files:**
- Create: `docs/superpowers/specs/2026-07-20-sonas-novelty-check.md`

**Interfaces:** Produces a written OPEN/OCCUPIED verdict that gates all following tasks.

- [ ] **Step 1: Run the searches.** Web-search each of these queries on Google Scholar and arXiv, restricted to 2022–2026; skim titles/abstracts of the top ~20 hits each:
  1. `neural architecture search small object detection`
  2. `NAS tiny object detection aerial`
  3. `evolutionary architecture search drone detection VisDrone`
  4. `multi-objective NAS object detection FPN`
- [ ] **Step 2: Classify hits.** For each relevant hit record: title, year, venue, whether its search space/objectives are small-object-specific, datasets, and whether it does multi-objective search. A paper counts as an *occupant* only if it combines (a) architecture search with (b) small-object-specific objectives or space and (c) aerial/small-object benchmarks.
- [ ] **Step 3: Write the verdict file** `docs/superpowers/specs/2026-07-20-sonas-novelty-check.md` with the hit table and one of: **OPEN** (proceed), **PARTIAL** (proceed, cite and differentiate — list the required differentiators), **OCCUPIED** (STOP — return to design).
- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-07-20-sonas-novelty-check.md
git commit -m "Record SONAS step-zero novelty check verdict

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 1: Package scaffold + Genome codec

**Files:**
- Create: `research/sonas/__init__.py` (empty), `research/sonas/requirements.txt`, `research/sonas/genome.py`
- Test: `research/sonas/tests/__init__.py` (empty), `research/sonas/tests/test_genome.py`

**Interfaces:**
- Produces: `Genome` (frozen dataclass), `GENE_NAMES: list[str]`, `GENE_CARDINALITIES: list[int]`, `Genome.from_vector(v) -> Genome`, `genome.to_vector() -> list[int]`, `Genome.random(rng) -> Genome`, `genome.to_dict() -> dict`, `Genome.from_dict(d) -> Genome`, and constants `WIDTH_PROFILES`, `BLOCKS = ["C3k2", "C2f", "C3Ghost"]`, `DOWNS = ["Conv", "SCDown"]`, `FUSIONS = ["pan", "ctx_p3", "p4skip"]`. Every later task consumes `Genome` and the vector encoding.

- [ ] **Step 1: Write requirements file** `research/sonas/requirements.txt`:

```
pymoo>=0.6.1
pycocotools>=2.0.7
scikit-learn>=1.3
scipy>=1.10
thop>=0.1.1
```

Install: `uv pip install -e . && uv pip install -r research/sonas/requirements.txt`

- [ ] **Step 2: Write the failing test** `research/sonas/tests/test_genome.py`:

```python
import random

import pytest

from research.sonas.genome import BLOCKS, DOWNS, FUSIONS, GENE_CARDINALITIES, GENE_NAMES, Genome


def test_gene_space_shape():
    assert len(GENE_NAMES) == len(GENE_CARDINALITIES) == 13


def test_vector_roundtrip():
    rng = random.Random(0)
    for _ in range(50):
        g = Genome.random(rng)
        assert Genome.from_vector(g.to_vector()) == g


def test_vector_bounds():
    rng = random.Random(1)
    for _ in range(50):
        v = Genome.random(rng).to_vector()
        assert all(0 <= x < c for x, c in zip(v, GENE_CARDINALITIES))


def test_dict_roundtrip_json_safe():
    import json

    g = Genome.random(random.Random(2))
    d = json.loads(json.dumps(g.to_dict()))
    assert Genome.from_dict(d) == g


def test_validation_rejects_bad_depth():
    g = Genome.random(random.Random(3))
    with pytest.raises(ValueError):
        Genome.from_dict({**g.to_dict(), "d3": 9})


def test_categorical_values_are_registry_members():
    g = Genome.random(random.Random(4))
    assert g.block in BLOCKS and g.down in DOWNS and g.fusion in FUSIONS
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest research/sonas/tests/test_genome.py -v`
Expected: FAIL / collection error with `ModuleNotFoundError: No module named 'research.sonas.genome'`

- [ ] **Step 4: Implement** `research/sonas/genome.py`:

```python
"""SONAS genome: 13 discrete genes encoding a YOLO-family architecture (spec §4.1)."""

from __future__ import annotations

import random
from dataclasses import dataclass, fields

# Width profiles distribute the channel budget across level groups (P2, P3, P4, P5).
WIDTH_PROFILES = {
    0: (1.0, 1.0, 1.0, 1.0),  # uniform
    1: (1.5, 1.25, 1.0, 0.75),  # shallow_heavy: capacity at high resolution
    2: (0.75, 1.0, 1.25, 1.5),  # deep_heavy: capacity at low resolution
}
BLOCKS = ["C3k2", "C2f", "C3Ghost"]
DOWNS = ["Conv", "SCDown"]
FUSIONS = ["pan", "ctx_p3", "p4skip"]

GENE_NAMES = ["p2", "p5", "d2", "d3", "d4", "d5", "head_depth", "widths", "block", "down", "fusion", "end2end", "psa"]
GENE_CARDINALITIES = [2, 2, 3, 3, 3, 3, 2, 3, 3, 2, 3, 2, 2]


@dataclass(frozen=True)
class Genome:
    p2: bool  # add P2/4 head level
    p5: bool  # keep P5/32 head level
    d2: int  # backbone stage repeats, 1..3
    d3: int
    d4: int
    d5: int
    head_depth: int  # head block repeats, 1..2
    widths: int  # WIDTH_PROFILES key, 0..2
    block: str  # backbone block, member of BLOCKS
    down: str  # P4/P5 downsample module, member of DOWNS
    fusion: str  # member of FUSIONS
    end2end: bool  # NMS-free dual-branch head vs standard NMS head
    psa: bool  # C2PSA attention block after SPPF

    def __post_init__(self):
        for name in ("d2", "d3", "d4", "d5"):
            if getattr(self, name) not in (1, 2, 3):
                raise ValueError(f"{name} must be in 1..3, got {getattr(self, name)}")
        if self.head_depth not in (1, 2):
            raise ValueError(f"head_depth must be 1 or 2, got {self.head_depth}")
        if self.widths not in WIDTH_PROFILES:
            raise ValueError(f"widths must be in {sorted(WIDTH_PROFILES)}, got {self.widths}")
        if self.block not in BLOCKS or self.down not in DOWNS or self.fusion not in FUSIONS:
            raise ValueError(f"bad categorical gene: block={self.block} down={self.down} fusion={self.fusion}")

    def to_vector(self) -> list:
        return [
            int(self.p2),
            int(self.p5),
            self.d2 - 1,
            self.d3 - 1,
            self.d4 - 1,
            self.d5 - 1,
            self.head_depth - 1,
            self.widths,
            BLOCKS.index(self.block),
            DOWNS.index(self.down),
            FUSIONS.index(self.fusion),
            int(self.end2end),
            int(self.psa),
        ]

    @classmethod
    def from_vector(cls, v) -> "Genome":
        v = [int(x) for x in v]
        if len(v) != len(GENE_CARDINALITIES):
            raise ValueError(f"expected {len(GENE_CARDINALITIES)} genes, got {len(v)}")
        for x, c, name in zip(v, GENE_CARDINALITIES, GENE_NAMES):
            if not 0 <= x < c:
                raise ValueError(f"gene {name}={x} out of range 0..{c - 1}")
        return cls(
            p2=bool(v[0]),
            p5=bool(v[1]),
            d2=v[2] + 1,
            d3=v[3] + 1,
            d4=v[4] + 1,
            d5=v[5] + 1,
            head_depth=v[6] + 1,
            widths=v[7],
            block=BLOCKS[v[8]],
            down=DOWNS[v[9]],
            fusion=FUSIONS[v[10]],
            end2end=bool(v[11]),
            psa=bool(v[12]),
        )

    @classmethod
    def random(cls, rng: random.Random) -> "Genome":
        return cls.from_vector([rng.randrange(c) for c in GENE_CARDINALITIES])

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        return cls(**{f.name: d[f.name] for f in fields(cls)})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest research/sonas/tests/test_genome.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add research/sonas/
git commit -m "Add SONAS genome codec (13 discrete genes, vector/dict roundtrip)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: YAML generator (genome → ultralytics model dict)

**Files:**
- Create: `research/sonas/yamlgen.py`
- Test: `research/sonas/tests/test_yamlgen.py`

**Interfaces:**
- Consumes: `Genome`, `WIDTH_PROFILES` from Task 1.
- Produces: `build_model_dict(genome: Genome, nc: int = 10) -> dict` (an in-memory ultralytics model config with keys `nc`, `end2end`, `reg_max`, `scales`, `backbone`, `head`) and `write_model_yaml(genome: Genome, path, nc: int = 10) -> pathlib.Path`. Tasks 3, 5 consume both.

Row format is the verified ultralytics convention `[from, repeats, module, args]` (see `ultralytics/cfg/models/26/yolo26.yaml` and `yolo26-p2.yaml`). Channels are baked absolute (n-scale base × width profile, divisible by 8) and `scales: {n: [1.0, 1.0, 1024]}` disables rescaling, so the genome fully controls the architecture.

- [ ] **Step 1: Write the failing test** `research/sonas/tests/test_yamlgen.py`:

```python
import random

from research.sonas.genome import Genome
from research.sonas.yamlgen import build_model_dict, write_model_yaml

STOCKLIKE = Genome(
    p2=False, p5=True, d2=2, d3=2, d4=2, d5=2, head_depth=2, widths=0,
    block="C3k2", down="Conv", fusion="pan", end2end=True, psa=True,
)


def detect_row(d):
    return d["head"][-1]


def test_stocklike_topology_matches_yolo26_shape():
    d = build_model_dict(STOCKLIKE, nc=10)
    assert d["nc"] == 10 and d["end2end"] is True and d["reg_max"] == 1
    assert d["scales"] == {"n": [1.0, 1.0, 1024]}
    assert len(d["backbone"]) == 11  # stem..C2PSA, same as stock yolo26
    assert detect_row(d)[2] == "Detect" and len(detect_row(d)[0]) == 3  # P3,P4,P5


def test_p2_adds_fourth_detect_level():
    d = build_model_dict(Genome.from_dict({**STOCKLIKE.to_dict(), "p2": True}), nc=10)
    assert len(detect_row(d)[0]) == 4


def test_no_p5_drops_to_two_levels():
    d = build_model_dict(Genome.from_dict({**STOCKLIKE.to_dict(), "p5": False}), nc=10)
    assert len(detect_row(d)[0]) == 2


def test_detect_sources_reference_existing_rows():
    rng = random.Random(0)
    for _ in range(30):
        g = Genome.random(rng)
        d = build_model_dict(g, nc=10)
        n_rows = len(d["backbone"]) + len(d["head"])
        for src in detect_row(d)[0]:
            assert 0 <= src < n_rows - 1


def test_every_random_genome_builds_a_torch_model():
    from ultralytics.nn.tasks import DetectionModel

    rng = random.Random(1)
    for _ in range(8):
        g = Genome.random(rng)
        model = DetectionModel(cfg=build_model_dict(g, nc=10), ch=3, nc=10, verbose=False)
        n_levels = 2 + int(g.p2) + int(g.p5)
        assert model.model[-1].nl == n_levels


def test_write_model_yaml_roundtrip(tmp_path):
    import yaml

    p = write_model_yaml(STOCKLIKE, tmp_path / "g.yaml", nc=10)
    loaded = yaml.safe_load(p.read_text())
    assert loaded["backbone"] == build_model_dict(STOCKLIKE, nc=10)["backbone"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest research/sonas/tests/test_yamlgen.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.sonas.yamlgen'`

- [ ] **Step 3: Implement** `research/sonas/yamlgen.py`:

```python
"""Decode a Genome into an ultralytics model-config dict (spec §4.1).

Row format: [from, repeats, module, args] — the convention used by ultralytics/cfg/models/26/yolo26.yaml.
Channels are absolute (scales fixed at [1.0, 1.0, 1024]) so the genome fully controls width.
"""

from __future__ import annotations

import pathlib

import yaml

from research.sonas.genome import WIDTH_PROFILES, Genome

# n-scale base channels per level (stock yolo26 channels × 0.25 width), before the profile multiplier.
BASE = {"stem": 16, "p2": 32, "c2": 64, "p3": 64, "c3": 128, "p4": 128, "c4": 128, "p5": 256, "c5": 256}


def _div8(x: float) -> int:
    return max(8, int(round(x / 8)) * 8)


def build_model_dict(g: Genome, nc: int = 10) -> dict:
    w2, w3, w4, w5 = WIDTH_PROFILES[g.widths]
    ch = {
        "stem": _div8(BASE["stem"]),
        "p2": _div8(BASE["p2"] * w2),
        "c2": _div8(BASE["c2"] * w2),
        "p3": _div8(BASE["p3"] * w3),
        "c3": _div8(BASE["c3"] * w3),
        "p4": _div8(BASE["p4"] * w4),
        "c4": _div8(BASE["c4"] * w4),
        "p5": _div8(BASE["p5"] * w5),
        "c5": _div8(BASE["c5"] * w5),
    }
    backbone, head = [], []
    idx = -1

    def add(rows, row):
        nonlocal idx
        idx += 1
        rows.append(row)
        return idx

    # Backbone: indices tracked by name for head wiring.
    add(backbone, [-1, 1, "Conv", [ch["stem"], 3, 2]])  # P1/2
    add(backbone, [-1, 1, "Conv", [ch["p2"], 3, 2]])  # P2/4
    b_c2 = add(backbone, [-1, g.d2, g.block, [ch["c2"], True]])
    add(backbone, [-1, 1, "Conv", [ch["p3"], 3, 2]])  # P3/8
    b_c3 = add(backbone, [-1, g.d3, g.block, [ch["c3"], True]])
    add(backbone, [-1, 1, g.down, [ch["p4"], 3, 2]])  # P4/16
    b_c4 = add(backbone, [-1, g.d4, g.block, [ch["c4"], True]])
    add(backbone, [-1, 1, g.down, [ch["p5"], 3, 2]])  # P5/32
    add(backbone, [-1, g.d5, g.block, [ch["c5"], True]])
    top = add(backbone, [-1, 1, "SPPF", [ch["c5"], 5, 3, True]])
    if g.psa:
        top = add(backbone, [-1, 2, "C2PSA", [ch["c5"]]])

    # Head: top-down (FPN) then bottom-up (PAN), P2/P5 branches optional.
    add(head, [-1, 1, "nn.Upsample", [None, 2, "nearest"]])
    add(head, [[-1, b_c4], 1, "Concat", [1]])
    h4 = add(head, [-1, g.head_depth, "C3k2", [ch["c4"], True]])

    add(head, [-1, 1, "nn.Upsample", [None, 2, "nearest"]])
    p3_cat = [-1, b_c3]
    if g.fusion == "ctx_p3":  # extra global-context path: top features upsampled x4 straight into P3
        ctx = add(head, [top, 1, "nn.Upsample", [None, 4, "nearest"]])
        p3_cat = [-2, b_c3, ctx]  # -2: the x2-upsample two rows back (the ctx row is -1)
    add(head, [p3_cat, 1, "Concat", [1]])
    h3 = add(head, [-1, g.head_depth, "C3k2", [ch["c3"], True]])

    detect_from = []
    smallest = h3
    if g.p2:
        add(head, [-1, 1, "nn.Upsample", [None, 2, "nearest"]])
        add(head, [[-1, b_c2], 1, "Concat", [1]])
        h2 = add(head, [-1, g.head_depth, "C3k2", [ch["c2"], True]])
        detect_from.append(h2)
        add(head, [-1, 1, "Conv", [ch["c2"], 3, 2]])  # back down to P3
        add(head, [[-1, h3], 1, "Concat", [1]])
        smallest = add(head, [-1, g.head_depth, "C3k2", [ch["c3"], True]])
    detect_from.append(smallest)

    add(head, [-1, 1, "Conv", [ch["c3"], 3, 2]])  # P3 -> P4 (smallest is always the previous row)
    p4_cat = [-1, h4]
    if g.fusion == "p4skip":  # extra raw-backbone skip at P4
        p4_cat = [-1, h4, b_c4]
    add(head, [p4_cat, 1, "Concat", [1]])
    h4b = add(head, [-1, g.head_depth, "C3k2", [ch["c4"], True]])
    detect_from.append(h4b)

    if g.p5:
        add(head, [-1, 1, "Conv", [ch["c4"], 3, 2]])  # P4 -> P5
        add(head, [[-1, top], 1, "Concat", [1]])
        h5 = add(head, [-1, 1, "C3k2", [ch["c5"], True]])
        detect_from.append(h5)

    add(head, [detect_from, 1, "Detect", ["nc"]])  # the string "nc" resolves inside parse_model, as in stock YAMLs
    return {
        "nc": nc,
        "end2end": bool(g.end2end),
        "reg_max": 1,
        "scales": {"n": [1.0, 1.0, 1024]},
        "backbone": backbone,
        "head": head,
    }


def write_model_yaml(g: Genome, path, nc: int = 10) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(build_model_dict(g, nc=nc), sort_keys=False))
    return path
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest research/sonas/tests/test_yamlgen.py -v`
Expected: 6 passed. If `test_every_random_genome_builds_a_torch_model` fails on a specific genome, print `g.to_dict()` from the failure, decode the row indices by hand against the generated dict, and fix the index bookkeeping in `build_model_dict` — the Detect-source test narrows where. One known subtlety already handled in the code above: the `ctx_p3` concat uses `-2` for the ×2-upsample because the ctx upsample row sits between it and the concat.

- [ ] **Step 5: Commit**

```bash
git add research/sonas/yamlgen.py research/sonas/tests/test_yamlgen.py
git commit -m "Add SONAS genome-to-YAML generator with topology tests

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Constraint metrics + repair

**Files:**
- Create: `research/sonas/metrics.py`
- Test: `research/sonas/tests/test_metrics.py`

**Interfaces:**
- Consumes: `Genome` (Task 1), `build_model_dict` (Task 2), `ultralytics.nn.tasks.DetectionModel`, `ultralytics.utils.torch_utils.get_flops`.
- Produces: `measure(genome, nc=10, imgsz=640) -> dict` with keys `params: int`, `gflops: float`; `is_feasible(m: dict, max_params=5_000_000, max_gflops=12.0) -> bool`; `repair(genome, nc=10, imgsz=640, max_params=5_000_000, max_gflops=12.0) -> Genome | None` (None = unrepairable). Tasks 5, 6 consume all three.

- [ ] **Step 1: Write the failing test** `research/sonas/tests/test_metrics.py`:

```python
from research.sonas.genome import Genome
from research.sonas.metrics import is_feasible, measure, repair

SLIM = Genome(p2=False, p5=True, d2=1, d3=1, d4=1, d5=1, head_depth=1, widths=0,
              block="C3Ghost", down="SCDown", fusion="pan", end2end=True, psa=False)
FAT = Genome(p2=True, p5=True, d2=3, d3=3, d4=3, d5=3, head_depth=2, widths=1,
             block="C3k2", down="Conv", fusion="ctx_p3", end2end=True, psa=True)


def test_measure_returns_positive_numbers():
    m = measure(SLIM)
    assert m["params"] > 100_000 and m["gflops"] > 0.1


def test_measure_raises_when_flops_unavailable(monkeypatch):
    import research.sonas.metrics as mod
    import pytest

    monkeypatch.setattr(mod, "get_flops", lambda *a, **k: 0.0)
    with pytest.raises(RuntimeError):
        measure(SLIM)


def test_slim_is_feasible_fat_may_not_be():
    assert is_feasible(measure(SLIM))


def test_repair_returns_feasible_or_none():
    r = repair(FAT)
    assert r is None or is_feasible(measure(r))


def test_repair_keeps_feasible_genome_unchanged():
    assert repair(SLIM) == SLIM
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest research/sonas/tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.sonas.metrics'`

- [ ] **Step 3: Implement** `research/sonas/metrics.py`:

```python
"""Feasibility metrics and deterministic shrink-repair (spec §4.1 hard constraints)."""

from __future__ import annotations

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops

from research.sonas.genome import Genome
from research.sonas.yamlgen import build_model_dict

MAX_PARAMS = 5_000_000
MAX_GFLOPS = 12.0


def measure(g: Genome, nc: int = 10, imgsz: int = 640) -> dict:
    model = DetectionModel(cfg=build_model_dict(g, nc=nc), ch=3, nc=nc, verbose=False)
    gflops = get_flops(model, imgsz)
    if not gflops:
        raise RuntimeError("get_flops returned 0.0 — is thop installed? A silent 0 would fake feasibility.")
    return {"params": sum(p.numel() for p in model.parameters()), "gflops": round(gflops, 2)}


def is_feasible(m: dict, max_params: int = MAX_PARAMS, max_gflops: float = MAX_GFLOPS) -> bool:
    return m["params"] <= max_params and m["gflops"] <= max_gflops


def _shrink_once(g: Genome) -> Genome | None:
    """One deterministic shrink step; None when nothing is left to shrink."""
    d = g.to_dict()
    if d["psa"]:
        d["psa"] = False
    elif d["fusion"] != "pan":
        d["fusion"] = "pan"
    elif d["d5"] > 1:
        d["d5"] -= 1
    elif d["d4"] > 1:
        d["d4"] -= 1
    elif d["d3"] > 1:
        d["d3"] -= 1
    elif d["d2"] > 1:
        d["d2"] -= 1
    elif d["head_depth"] > 1:
        d["head_depth"] = 1
    elif d["widths"] != 0:
        d["widths"] = 0
    elif d["block"] != "C3Ghost":
        d["block"] = "C3Ghost"
    elif d["down"] != "SCDown":
        d["down"] = "SCDown"
    else:
        return None
    return Genome.from_dict(d)


def repair(g: Genome, nc: int = 10, imgsz: int = 640,
           max_params: int = MAX_PARAMS, max_gflops: float = MAX_GFLOPS) -> Genome | None:
    for _ in range(20):
        if is_feasible(measure(g, nc=nc, imgsz=imgsz), max_params, max_gflops):
            return g
        nxt = _shrink_once(g)
        if nxt is None:
            return None
        g = nxt
    return None
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest research/sonas/tests/test_metrics.py -v`
Expected: 5 passed (model builds on CPU; ~seconds per build)

- [ ] **Step 5: Commit**

```bash
git add research/sonas/metrics.py research/sonas/tests/test_metrics.py
git commit -m "Add SONAS feasibility metrics and deterministic shrink-repair

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: VisDrone→COCO ground truth + AP_small evaluator

**Files:**
- Create: `research/sonas/coco_eval.py`
- Test: `research/sonas/tests/test_coco_eval.py`

**Interfaces:**
- Consumes: ultralytics YOLO-format labels layout (`<root>/images/…`, `<root>/labels/…` as produced by `VisDrone.yaml` auto-download); ultralytics `pred_to_json` output format (COCO-style `[{"image_id", "category_id", "bbox": [x,y,w,h], "score"}]`, image_id = filename stem, category_id = 0-based class index).
- Produces: `gt_to_coco(images_dir, labels_dir, out_json, class_names) -> pathlib.Path` and `evaluate_coco(gt_json, pred_json) -> dict` with keys `map5095, map50, ap_small, ap_medium, ap_large`. Task 5 consumes `evaluate_coco`; the Colab setup builds the GT json once.

**Convention decisions (must match ultralytics `pred_to_json` in `ultralytics/models/yolo/detect/val.py:413`):** image ids are filename stems (kept as strings when non-numeric — pycocotools accepts any hashable id), category ids are **1-based** (`cls + 1`) because for non-COCO/LVIS datasets `pred_to_json` maps classes through `self.class_map = list(range(1, nc + 1))` (val.py:90) — VisDrone is non-COCO, so a 0-based GT would silently zero every AP. bbox is COCO `[x, y, w, h]` in pixels, `area = w * h`. *(Corrected 2026-07-20 during execution — the original 0-based assertion was wrong; caught by Task 4's review.)*

- [ ] **Step 1: Write the failing test** `research/sonas/tests/test_coco_eval.py`:

```python
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
    assert {c["id"] for c in gt["categories"]} == {0, 1}


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest research/sonas/tests/test_coco_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.sonas.coco_eval'`

- [ ] **Step 3: Implement** `research/sonas/coco_eval.py`:

```python
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
                "category_id": cls,
                "bbox": [xc * w - bw_px / 2, yc * h - bh_px / 2, bw_px, bh_px],
                "area": bw_px * bh_px,
                "iscrowd": 0,
            })
            ann_id += 1
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for i, n in enumerate(class_names)],
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
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest research/sonas/tests/test_coco_eval.py -v`
Expected: 3 passed. If `loadRes` complains about string image ids, the ids in GT and predictions disagree — print both id sets; the fix is always making the GT follow `pred_to_json`'s stem convention, never the reverse.

- [ ] **Step 5: Commit**

```bash
git add research/sonas/coco_eval.py research/sonas/tests/test_coco_eval.py
git commit -m "Add VisDrone-to-COCO converter and AP_small evaluator

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Proxy-run harness with NDJSON journal

**Files:**
- Create: `research/sonas/proxy.py`
- Test: `research/sonas/tests/test_proxy.py`

**Interfaces:**
- Consumes: `Genome`, `write_model_yaml`, `measure`, `evaluate_coco`.
- Produces: `run_proxy(genome, *, data, gt_json, project, epochs=15, imgsz=640, device=0, seed=0, trainer=None) -> dict` (record with keys `key`, `genome`, `params`, `gflops`, `ap_small`, `map5095`, `status`), `append_eval(journal, record)`, `load_journal(journal) -> list[dict]`, `lookup(records, genome) -> dict | None`. Task 6 consumes the journal API; `trainer` is injectable for tests.
- The journal is the single source of truth: every evaluation (including failures, `status="error"`) is appended; the search resumes from it; the surrogate trains on it.

- [ ] **Step 1: Write the failing test** `research/sonas/tests/test_proxy.py`:

```python
import random

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest research/sonas/tests/test_proxy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.sonas.proxy'`

- [ ] **Step 3: Implement** `research/sonas/proxy.py`:

```python
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
    return [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]


def lookup(records: list, g: Genome):
    key = _key(g)
    for rec in records:
        if rec["key"] == key:
            return rec
    return None
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest research/sonas/tests/test_proxy.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add research/sonas/proxy.py research/sonas/tests/test_proxy.py
git commit -m "Add SONAS proxy-run harness with NDJSON evaluation journal

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: NSGA-II driver (pymoo) with journal cache/resume

**Files:**
- Create: `research/sonas/search.py`
- Test: `research/sonas/tests/test_search.py`

**Interfaces:**
- Consumes: `Genome`, `GENE_CARDINALITIES`, `repair`, journal API from Task 5.
- Produces: `run_search(evaluate, *, baseline_map5095, journal, pop_size=20, n_gen=8, seed=0) -> pymoo Result` where `evaluate: Callable[[Genome], dict]` returns a proxy record. Objectives: `f1 = -ap_small`, `f2 = gflops`; constraint `g1 = (baseline_map5095 - 0.01) - map5095 <= 0` (proxy-to-proxy, spec §4.2; mAP is a 0–1 fraction, so one percentage point = 0.01). Already-journaled genomes are returned from cache, which is what makes a killed Colab session resumable.

- [ ] **Step 1: Write the failing test** `research/sonas/tests/test_search.py`:

```python
from research.sonas.genome import Genome
from research.sonas.proxy import load_journal
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest research/sonas/tests/test_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'research.sonas.search'`

- [ ] **Step 3: Implement** `research/sonas/search.py`:

```python
"""NSGA-II over the SONAS genome space with journal-backed caching (spec §4.2)."""

from __future__ import annotations

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

from research.sonas.genome import GENE_CARDINALITIES, Genome
from research.sonas.metrics import repair as shrink_repair
from research.sonas.proxy import append_eval, load_journal, lookup


class SonasProblem(ElementwiseProblem):
    def __init__(self, evaluate, baseline_map5095: float, journal):
        super().__init__(n_var=len(GENE_CARDINALITIES), n_obj=2, n_ieq_constr=1,
                         xl=np.zeros(len(GENE_CARDINALITIES)),
                         xu=np.array(GENE_CARDINALITIES) - 1, vtype=int)
        self.evaluate_genome = evaluate
        self.baseline = baseline_map5095
        self.journal = journal

    def _evaluate(self, x, out, *args, **kwargs):
        g = Genome.from_vector(np.rint(x).astype(int))
        repaired = shrink_repair(g)
        if repaired is None:  # provably can't fit the budget — never spend a training run on it
            out["F"] = [1.0, 1e3]  # dominated by everything real
            out["G"] = [1.0]
            return
        g = repaired
        rec = lookup(load_journal(self.journal), g)
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


def run_search(evaluate, *, baseline_map5095: float, journal, pop_size: int = 20, n_gen: int = 8, seed: int = 0):
    problem = SonasProblem(evaluate, baseline_map5095, journal)
    algo = NSGA2(pop_size=pop_size, sampling=IntegerRandomSampling(),
                 crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
                 mutation=PM(prob=0.3, eta=20, vtype=float, repair=RoundingRepair()),
                 eliminate_duplicates=True)
    return minimize(problem, algo, ("n_gen", n_gen), seed=seed, verbose=False)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest research/sonas/tests/test_search.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add research/sonas/search.py research/sonas/tests/test_search.py
git commit -m "Add NSGA-II search driver with journal cache and constraint handling

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: GBT surrogate pre-screen

**Files:**
- Create: `research/sonas/surrogate.py`
- Test: `research/sonas/tests/test_surrogate.py`

**Interfaces:**
- Consumes: journal records (Task 5 format).
- Produces: `fit_surrogate(records, min_records=40) -> Surrogate | None` and `Surrogate.screen(vectors, keep) -> list[int]` (indices of the `keep` most promising candidates by predicted non-dominated rank). The search integrates it in month 2 by pre-filtering each generation's offspring; wiring into `run_search` is deliberately deferred until the plain search has produced its first real journal (YAGNI until then).

- [ ] **Step 1: Write the failing test** `research/sonas/tests/test_surrogate.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest research/sonas/tests/test_surrogate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement** `research/sonas/surrogate.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest research/sonas/tests/test_surrogate.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add research/sonas/surrogate.py research/sonas/tests/test_surrogate.py
git commit -m "Add GBT surrogate with predicted-rank offspring screening

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Colab pipeline + kill-test scripts (KT2 infra smoke, KT3 proxy trust)

**Files:**
- Create: `research/sonas/scripts/__init__.py` (empty — makes the scripts importable for the local test), `research/sonas/scripts/colab_setup.md`, `research/sonas/scripts/kt2_smoke.py`, `research/sonas/scripts/kt3_proxy_trust.py`, `research/sonas/scripts/baseline_proxy.py`
- Test: `research/sonas/tests/test_kendall.py`

**Prerequisite step (one-time, before the Colab doc is usable):** Colab needs to `git clone` the branch, and `sonas-paper` currently exists only in the local worktree. Create a **private** GitHub repository (e.g., `karim/sonas-ultralytics` — private because this is unpublished research), then:

```bash
git remote add research git@github.com:<karim-account>/sonas-ultralytics.git
git push -u research sonas-paper
```

Then substitute that clone URL for `<YOUR-FORK-URL>` in `colab_setup.md` Cell 2 (with a read token or `gh auth` in Colab for private access). Never push this branch to the `origin` ultralytics remote.

**Interfaces:**
- Consumes: everything above.
- Produces: the runnable Colab workflow and `kendall_verdict(records_short, records_long, threshold=0.6) -> dict` (keys `tau`, `p`, `passed`) inside `kt3_proxy_trust.py` — the spec §4.3.3 gate, unit-tested locally.

- [ ] **Step 1: Write the failing test** `research/sonas/tests/test_kendall.py`:

```python
from research.sonas.scripts.kt3_proxy_trust import kendall_verdict


def rec(key, ap):
    return {"key": key, "ap_small": ap, "status": "ok"}


def test_perfect_agreement_passes():
    short = [rec(str(i), 0.1 + i / 100) for i in range(10)]
    long = [rec(str(i), 0.2 + i / 50) for i in range(10)]
    v = kendall_verdict(short, long)
    assert v["passed"] and v["tau"] > 0.99


def test_shuffled_disagreement_fails():
    short = [rec(str(i), 0.1 + i / 100) for i in range(10)]
    long = [rec(str(i), 0.2 + ((i * 7) % 10) / 50) for i in range(10)]
    assert not kendall_verdict(short, long)["passed"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest research/sonas/tests/test_kendall.py -v`
Expected: FAIL with `ModuleNotFoundError` (add `research/sonas/scripts/__init__.py`, empty, so the import works)

- [ ] **Step 3: Implement the three scripts.**

`research/sonas/scripts/kt3_proxy_trust.py`:

```python
"""KT3 (spec §4.3.3): do 15-epoch rankings agree with 120-epoch rankings? Gate: Kendall tau >= 0.6.

Colab usage (after colab_setup.md):
    python -m research.sonas.scripts.kt3_proxy_trust --workdir /content/drive/MyDrive/sonas --stage short
    python -m research.sonas.scripts.kt3_proxy_trust --workdir ... --stage long
    python -m research.sonas.scripts.kt3_proxy_trust --workdir ... --stage verdict
"""

from __future__ import annotations

import argparse
import pathlib
import random

from scipy.stats import kendalltau

from research.sonas.genome import Genome
from research.sonas.metrics import repair
from research.sonas.proxy import append_eval, load_journal, run_proxy

PANEL_SEED = 42
PANEL_SIZE = 10
TAU_GATE = 0.6


def panel() -> list:
    rng = random.Random(PANEL_SEED)
    out = []
    while len(out) < PANEL_SIZE:
        g = repair(Genome.random(rng))
        if g is not None and g not in out:
            out.append(g)
    return out


def kendall_verdict(records_short, records_long, threshold: float = TAU_GATE) -> dict:
    by_key_s = {r["key"]: r["ap_small"] for r in records_short if r["status"] == "ok"}
    by_key_l = {r["key"]: r["ap_small"] for r in records_long if r["status"] == "ok"}
    keys = sorted(set(by_key_s) & set(by_key_l))
    tau, p = kendalltau([by_key_s[k] for k in keys], [by_key_l[k] for k in keys])
    return {"tau": float(tau), "p": float(p), "n": len(keys), "passed": bool(tau >= threshold)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--stage", choices=["short", "long", "verdict"], required=True)
    ap.add_argument("--device", default=0)
    args = ap.parse_args()
    work = pathlib.Path(args.workdir)
    if args.stage == "verdict":
        v = kendall_verdict(load_journal(work / "kt3_short.ndjson"), load_journal(work / "kt3_long.ndjson"))
        print(v)
        print("PROXY TRUSTED — start the search" if v["passed"] else "PROXY NOT TRUSTED — lengthen proxy (spec §8)")
        return
    epochs = 15 if args.stage == "short" else 120
    journal = work / f"kt3_{args.stage}.ndjson"
    done = {r["key"] for r in load_journal(journal)}
    for g in panel():
        if "-".join(map(str, g.to_vector())) in done:
            continue  # resume-safe after Colab disconnects
        rec = run_proxy(g, data="VisDrone.yaml", gt_json=work / "gt.json", project=work / f"kt3_{args.stage}",
                        epochs=epochs, device=args.device)
        append_eval(journal, rec)
        print(rec["key"], rec["status"], rec.get("ap_small"))


if __name__ == "__main__":
    main()
```

`research/sonas/scripts/kt2_smoke.py`:

```python
"""KT2 (spec §4.3.2): 8 random repaired genomes must each survive a 3-epoch training without crashing."""

from __future__ import annotations

import argparse
import pathlib
import random

from research.sonas.genome import Genome
from research.sonas.metrics import measure, repair
from research.sonas.proxy import run_proxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--device", default=0)
    args = ap.parse_args()
    rng = random.Random(7)
    failures = ran = 0
    for i in range(8):
        g = repair(Genome.random(rng))
        if g is None:
            print(f"[{i}] unrepairable draw, skipped")
            continue
        ran += 1
        rec = run_proxy(g, data="VisDrone.yaml", gt_json=pathlib.Path(args.workdir) / "gt.json",
                        project=pathlib.Path(args.workdir) / "kt2", epochs=3, device=args.device)
        m = measure(g)
        print(f"[{i}] {rec['key']} status={rec['status']} params={m['params']:,} gflops={m['gflops']}")
        failures += rec["status"] != "ok"
    passed = failures == 0 and ran > 0  # zero runs must not read as success
    print(f"KT2 {'PASSED' if passed else 'FAILED'} ({ran} ran, {failures} failed, {8 - ran} unrepairable)")


if __name__ == "__main__":
    main()
```

`research/sonas/scripts/baseline_proxy.py`:

```python
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
```

`research/sonas/scripts/colab_setup.md` — paste these cells into a fresh Colab (A100, background execution on):

````markdown
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

```python
# Cell 3 — dataset (auto-downloads via VisDrone.yaml on first use) + one-time COCO GT
from ultralytics.utils import DATASETS_DIR
from research.sonas.coco_eval import gt_to_coco
from ultralytics.data.utils import check_det_dataset
data = check_det_dataset("VisDrone.yaml")  # triggers download; converter moves files to VisDrone/{images,labels}/val and DELETES the VisDrone2019-DET-* dirs
val_images = str(data["val"])                       # .../VisDrone/images/val
val_labels = val_images.replace("/images/", "/labels/")
gt_to_coco(val_images, val_labels, f"{WORK}/gt.json", data["names"])
```

```python
# Cell 4 — kill-tests, in spec order (each is resume-safe; rerun the cell after any disconnect)
!python -m research.sonas.scripts.kt2_smoke --workdir $WORK
!python -m research.sonas.scripts.baseline_proxy --workdir $WORK
!python -m research.sonas.scripts.kt3_proxy_trust --workdir $WORK --stage short
!python -m research.sonas.scripts.kt3_proxy_trust --workdir $WORK --stage long   # ~40-50 A100-hours total
!python -m research.sonas.scripts.kt3_proxy_trust --workdir $WORK --stage verdict
```
````

- [ ] **Step 4: Run the local tests**

Run: `python -m pytest research/sonas/tests/test_kendall.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the full local suite once**

Run: `python -m pytest research/sonas/tests/ -v`
Expected: all tests pass (Tasks 1–8)

- [ ] **Step 6: Commit**

```bash
git add research/sonas/scripts/ research/sonas/tests/test_kendall.py
git commit -m "Add Colab pipeline and KT2/KT3 kill-test scripts with resume safety

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Execution notes

- **Conscious v1 deviation from spec §4.1:** the "early stride schedule" dimension is encoded as the downsample *operator* gene (`Conv` vs `SCDown`) rather than downsample *placement*; placement variants change every downstream index and are deferred to a v2 search-space extension if month-2 results suggest resolution scheduling matters. Record this in the paper's search-space section.
- **Order is strict:** Task 0 gates everything; Tasks 1→8 build on each other; KT2/KT3 then run on Colab (human-in-the-loop, GPU-days) before any search.
- **What this plan does NOT include (deliberately):** the actual search campaign (month 2–3, run on Colab with `run_search` + real `run_proxy` once KT3 passes), surrogate wiring into `run_search` (after the first real journal exists), full trainings, transfer datasets, and the density-binned AP analysis (month 7–8 per spec §5 — AP_small via COCOeval is sufficient for the search objectives).
- **Verification before "done":** every task's tests pass locally; `python -m pytest research/sonas/tests/ -v` green; KT2 prints `KT2 PASSED` on Colab; KT3 verdict decides whether month-2 search starts or spec §8's fallback ladder engages.

---

## Post-review amendments (final whole-branch review, 2026-07-20)

Applied after all 9 tasks passed task-level review; suite went 30 → 37 tests. Details in `.superpowers/sdd/final-review-fixes-report.md`.

1. **(Critical)** `search.py`: shrink-repair relocated into a pymoo `Repair` operator (`ShrinkRepair`) with `lru_cache`-memoized repair — the population now carries repaired genomes, so `res.X` ≡ evaluated architectures; `save_history=True` enables the generation-4 check. Winner extraction should still read the journal's Pareto set, not `res.X`.
2. `proxy.load_journal` tolerates a truncated final NDJSON line (Colab hard-kill mid-append); interior corruption still raises.
3. `proxy.lookup` returns ok-status records only — transient failures retry on resume instead of permanently poisoning the cache; kt3's done-set likewise.
4. `run_search(settings=...)` validates journal records against the run's proxy settings (guards the spec §8 lengthen-proxy fallback against stale-cache mixing).
5. `kendall_verdict` gains `min_n=PANEL_SIZE` — τ on a silently shrunken panel no longer passes.
6. `run_search` rejects a non-fraction `baseline_map5095` (percent/fraction mixup would silently degenerate the search).
7. `evaluate_coco` sets `maxDets=[1,100,300]` (NOT [1,10,300]: pycocotools `summarize()` hardcodes a `maxDets==100` lookup for mAP; 300 matches ultralytics val `max_det` and fixes dense-scene AP_small truncation). Month-4 paper tables need a custom summarize (map5095@100 vs ap_small@300 mix is fine proxy-to-proxy only).
8. `requirements.txt` pinned to exact versions (incl. `ultralytics-thop==2.0.20`); Colab doc notes `pip freeze` lock + commit-SHA pinning once the branch has commits.
Minors: Cell 3 passes `list(data["names"].values())`; kt3 imports `_key` from proxy; kt2/baseline resume-safety claim corrected.
