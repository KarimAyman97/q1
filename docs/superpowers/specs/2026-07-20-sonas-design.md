# SONAS — Small-Object-Aware Multi-Objective Neural Architecture Search for Efficient Aerial Object Detection

**Design spec, 2026-07-20.** Author: Karim Ayman (advisor: Prof. Elshafey). Assistant-drafted, approved by Karim after a multi-round brainstorming session and an adversarially-verified literature sweep (2026-07-20).

---

## 1. Goal and one-sentence story

Every recent lightweight aerial small-object detector is designed **by hand**. SONAS replaces hand design with an **evolutionary multi-objective search whose objectives are built for small objects** — and delivers a Pareto front of architectures that match or beat hand-designed state of the art on small-object accuracy at equal or lower cost, validated across multiple datasets, plus an analysis of the design principles the search discovered.

> Search finds detector architectures that see small objects better while costing less — proven on several datasets — designed by optimization, not by hand.

## 2. Constraints (fixed)

| Constraint | Value |
| --- | --- |
| Team | Solo (Karim), advisor review |
| Timeline | 9–12 months to journal submission (~mid-2027) |
| Compute | Google Colab Pro+ (A100, background execution); no cluster |
| Hardware | **No edge device will be available — the paper must make no on-device claims** |
| Primary dataset | VisDrone-DET (config ships in this repo: `ultralytics/cfg/datasets/VisDrone.yaml`) |
| Codebase | This ultralytics checkout; research code on branch `sonas-paper` (never `main`) |
| Target venue class | JCR-Q1 applied venues (Scientific Reports tier and up: ESWA, EAAI, Neurocomputing, Pattern Recognition as stretch) |

**No-hardware precedent (verified in the 2026-07-20 sweep):** BPD-YOLO (Scientific Reports, 2025) was accepted with desktop-GPU-only evidence — no Jetson, no energy measurements. SONAS follows that evidence bundle and reports T4/A100 latency as *descriptive* information only, never as an edge claim.

## 3. Literature position (from the 2026-07-20 deep-research sweep)

### 3.1 The gap

All current lightweight VisDrone/aerial detectors found by the sweep are **hand-designed, with zero search**:

| Competitor | Venue, year | VisDrone result | Search? | Edge HW? |
| --- | --- | --- | --- | --- |
| EBAD-YOLO (YOLOv10s + LAMP prune) | J. Real-Time Image Proc., 12/2025 | 35.9 mAP@50 | none | FPS only |
| LMW-YOLO | Scientific Reports, 2026 | 37.2 mAP@50 @ ~2.6M params | none | none (CPU FPS) |
| BPD-YOLO | Scientific Reports, 2025 | +2.8 mAP50 over v8n+P2 | none | none |
| LEAF-YOLO | Intell. Sys. w/ Appl., 2025 | 48.3 AP50 (val) @ 4.28M | none | FPS only |
| LAF-YOLOv10 | arXiv 2602.13378, 2026 | mAP not captured in sweep notes — pull from paper | none | measured, single design |

Hardware-aware/multi-objective search exists but not here: HAMP (PMC11330115) does multi-objective evolutionary pruning **on classification only** (CIFAR-10, AlexNet/MobileNet) and uses plain NSGA-II as a *baseline* — so SONAS must not present bare NSGA-II as its novelty; the novelty is the **small-object-specific search space and objectives**, surrogate assistance, and the transfer/principles study.

*Caveat: competitor numbers above were extracted with quotes from primary sources but several lost their formal 3-vote verification to a session cap — re-verify every number against the papers when writing the related-work section.*

### 3.2 Step-zero novelty check (blocking)

Before any implementation: one focused literature search on **"NAS / architecture search for small-object or tiny-object detection, 2022–2026"** (the sweep covered adjacent clusters, not this exact phrase). If a direct occupant exists, redesign the differentiators before proceeding. Re-run a quick check at month 6 before writing.

### 3.3 Adjacent occupied territory (do not claim)

- Muon-family optimizer studies for detection: occupied (MiMuon, arXiv 2605.19619, benchmarks five optimizers on YOLO26m). Do not include an optimizer contribution.
- Small-object label assignment: occupied upstream by YOLO26's STAL + ProgLoss (arXiv 2509.25164, 2606.03748). SONAS searches *architecture*, not assignment.
- Hardware-in-the-loop / measured-energy objectives: requires a device; excluded by constraint.

## 4. Method

### 4.1 Search space (~15–18 discrete genes)

Every dimension encodes a testable small-object hypothesis:

| Dimension | Genes | Small-object hypothesis |
| --- | --- | --- |
| Head levels | P2 on/off, P5 on/off | high-res head helps tiny objects; coarse P5 head may be dead weight in aerial scenes |
| Early stride schedule | downsample placement in backbone | preserving early resolution keeps tiny-object features alive |
| Channel budget | per-level width split under a total budget | capacity should concentrate in shallow, high-resolution levels |
| Fusion topology | 3–4 discrete variants (PAN / BiFPN-lite / extra high-res skip) | how small-object features acquire context |
| Blocks per stage | choices from the repo module zoo (C2f, C3k2, ghost/partial conv, attention-lite) | cheap blocks where resolution is high, capable blocks where it pays |
| Depth/width scales | per-stage multipliers | where depth actually helps |
| Head type | end-to-end NMS-free vs standard NMS head | reported bonus finding either way |

- **Decoding:** genome → generated model YAML → `parse_model` (`ultralytics/nn/tasks.py`) builds it. No new layer code in v1 — the space is combinations of existing modules (delete/replace/reconfigure, not invent).
- **Hard constraints:** params ≤ ~5M, GFLOPs ≤ ~12 (nano/small class, where the published bar sits). Infeasible genomes are repaired or rejected at decode time.

### 4.2 Objectives and search algorithm

- **Objective 1:** maximize AP_small (COCO definition; density-binned AP tracked alongside) from the proxy run on VisDrone-val.
- **Objective 2:** minimize GFLOPs (params reported; capped by constraint).
- **Constraint:** overall mAP50-95 ≥ (stock same-budget baseline **under the same 15-epoch proxy** − 1.0 pt) — proxy compared to proxy, never proxy to full training — so the search may not sacrifice large objects to farm small ones.
- **Algorithm:** NSGA-II (pymoo), population 20, 8–10 generations (~160–200 evaluations), plus a light surrogate: a predictor fit on already-evaluated genomes (default: gradient-boosted trees over the genome encoding; exact choice pinned in the implementation plan) pre-screens offspring — positioned against the surrogate-assisted MOEA lineage, and the answer to "plain NSGA-II is a published baseline".
- **Proxy evaluation:** 15-epoch VisDrone training at 640 on A100 (~25 min each), fixed seed, fixed augmentation recipe.

### 4.3 Kill-tests, strictly in order (cheap failure before expensive failure)

1. **Week 0 — novelty check** (§3.2). Blocking.
2. **Weeks 1–2 — infra correctness:** genome→YAML→model roundtrip unit tests pass; a batch of random genomes each trains 3 epochs without crashing; decoded params/GFLOPs match `get_flops` ground truth.
3. **Weeks 2–3 — proxy trust:** 10 spread configs trained at 15 and 120 epochs; require Kendall τ ≥ 0.6 between rankings. Below that: lengthen proxy (25–30 epochs), train on a stratified VisDrone subset, or fall back to a weight-sharing supernet. Do not start the search on an untrusted proxy.
4. **Mid-search check (generation 4):** the current front must dominate stock YOLOv8n and YOLO26n on the proxy. If not, the search space is wrong — fix it then, not at month 6.

## 5. Evaluation protocol

- **Datasets:** search on VisDrone-DET; full-train 3–5 Pareto-knee architectures (~120+ epochs, n- and s-scale); transfer-retrain winners on UAVDT and TinyPerson (AI-TOD if time allows).
- **Baselines:** stock YOLOv8n/s and YOLO26n/s trained under identical recipes; published numbers for EBAD-YOLO, LMW-YOLO, BPD-YOLO, LEAF-YOLO (re-verified); search-method ablations: random search, single-objective (accuracy-only), and no-AP_small-objective variants at matched evaluation budget.
- **Metrics:** mAP50, mAP50-95, AP_small, density-binned AP, params, GFLOPs; T4/A100 latency reported descriptively (explicitly *not* an edge claim).
- **Key ablations:**
  1. *Objective ablation* — with vs without the AP_small objective at equal budget: proves C1 (the small-object objective changes what is found).
  2. *Search-space ablation* — freeze each dimension group; which dimensions matter.
  3. *Proxy fidelity* — report the τ study honestly.
  4. *Principle validation* — extract 2–3 recurring design principles from the winners, apply them by hand to a stock model, show the gain persists. This is the chapter that lifts the paper above "we ran NAS."

## 6. Compute budget (Colab Pro+, A100 background execution)

| Item | Estimate |
| --- | --- |
| Search (≈170 proxy runs × ~25 min) | ~70 h |
| Proxy-trust study (10 × full + 10 × proxy) | ~55 h |
| Final full trainings (3–5 archs × 2 scales) | ~50 h |
| Transfer datasets (2–3 × 3 archs) | ~30 h |
| Baselines + ablations | ~60 h |
| **Total** | **~265–300 A100-hours over ~6 months** |

Discipline: checkpoints to Drive every epoch, `resume=True` on every long run, fixed seeds, results logged to CSV committed on the branch.

## 7. Timeline (12 months from 2026-07-20)

| Months | Work |
| --- | --- |
| M0 | Novelty check; genome codec + YAML generator + unit tests |
| M1 | Colab pipeline; stock baselines; proxy-trust kill-test |
| M2–3 | Search runs (+ surrogate), generation-4 check |
| M4–5 | Full-train winners; VisDrone tables; objective/space ablations |
| M6 | Transfer datasets; novelty re-check |
| M7–8 | Design-principle analysis + validation; figures |
| M9–10 | Writing; advisor review; submission |
| M11–12 | Buffer for reviews/revisions |

## 8. Risks and fallbacks

| Risk | Mitigation / fallback |
| --- | --- |
| Proxy ranking unreliable | Longer proxy → subset training → weight-sharing supernet (in that order) |
| Search finds nothing beating hand design | Space contains known-good ingredients (P2 head, ghost blocks), so the front lower-bounds informed hand design; objective-ablation finding + principles + transfer still carry the paper if margins are thin |
| "Crowded genre" reviews | The three lifts no competitor has: search, principle extraction, multi-dataset transfer |
| Scooped | Step-zero check now; re-check month 6; the specific bundle (small-object objectives + principles + transfer) is hard to fully occupy |
| Colab instability | Background execution, Drive checkpoints, resume discipline; Kaggle 30 h/week as overflow |

## 9. Engineering plan (this repo)

- All work on branch `sonas-paper` in the `ultralytics-sonas` worktree; never `main` (repo rule). Research code is not intended for upstream PR.
- New `research/sonas/` package: genome codec, YAML generator, pymoo driver, proxy-run harness, analysis notebooks. Reuses `parse_model`, the module zoo (`ultralytics/nn/modules/`), `VisDrone.yaml` auto-download, the standard val pipeline, and `get_flops` — no duplication of existing utilities.
- Tests: unit tests for codec roundtrip and constraint repair; smoke ladder before any long spend (1-batch CPU → 3-epoch Colab → full run).

## 10. Explicitly out of scope

Edge-device measurements of any kind; optimizer contributions (Muon/MuSGD); label-assignment changes; distillation of any form; new hand-designed layer modules (v1 searches combinations of existing ones); tiled/sliced inference.
