"""Month-2 SONAS search campaign: NSGA-II with real proxy training (spec §4.2).

Self-gated: refuses to start unless baseline_proxy.json exists AND the KT3 verdict passes
(tau >= 0.6 over the full panel). Resume-safe: every evaluation journals to
<workdir>/search.ndjson; rerunning after a disconnect continues where it stopped.

Colab usage:
    python -m research.sonas.scripts.campaign --workdir /content/drive/MyDrive/sonas
"""

from __future__ import annotations

import argparse
import json
import pathlib

from research.sonas.genome import Genome
from research.sonas.proxy import load_journal, run_proxy
from research.sonas.scripts.kt3_proxy_trust import kendall_verdict
from research.sonas.search import run_search

PROXY = {"epochs": 15, "imgsz": 640, "seed": 0}  # must match KT3-short and baseline_proxy


def pareto_front(records: list, baseline: float) -> list:
    """Feasible, non-dominated records (maximize ap_small, minimize gflops), sorted by gflops."""
    feas = [r for r in records if r["status"] == "ok" and r["map5095"] >= baseline - 0.01]
    front = [
        r
        for r in feas
        if not any(
            o["ap_small"] >= r["ap_small"]
            and o["gflops"] <= r["gflops"]
            and (o["ap_small"] > r["ap_small"] or o["gflops"] < r["gflops"])
            for o in feas
        )
    ]
    return sorted(front, key=lambda r: r["gflops"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--device", default=0)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--gens", type=int, default=8)
    args = ap.parse_args()
    work = pathlib.Path(args.workdir)

    # Gate 1: the baseline measuring stick must exist.
    bl_path = work / "baseline_proxy.json"
    if not bl_path.exists():
        raise SystemExit("GATE FAILED: baseline_proxy.json not found — run baseline_proxy first.")
    baseline = json.loads(bl_path.read_text())["map5095"]

    # Gate 2: the proxy must be proven trustworthy (KT3, tau >= 0.6 over the full panel).
    v = kendall_verdict(load_journal(work / "kt3_short.ndjson"), load_journal(work / "kt3_long.ndjson"))
    if not v["passed"]:
        raise SystemExit(f"GATE FAILED: KT3 verdict {v} — never search on an untrusted proxy (spec §4.3.3).")
    print(f"Gates OK: tau={v['tau']:.3f} (n={v['n']}), baseline map5095={baseline:.4f}")
    print(f"Searching: pop={args.pop} x gens={args.gens} (~{args.pop * (args.gens + 1)} evaluations max, cache-aware)")

    def evaluate(g: Genome) -> dict:
        return run_proxy(
            g,
            data="VisDrone.yaml",
            gt_json=work / "gt.json",
            project=work / "search",
            epochs=PROXY["epochs"],
            imgsz=PROXY["imgsz"],
            seed=PROXY["seed"],
            device=args.device,
        )

    run_search(
        evaluate,
        baseline_map5095=baseline,
        journal=work / "search.ndjson",
        pop_size=args.pop,
        n_gen=args.gens,
        seed=0,
        settings=PROXY,
    )

    # Report the front from the JOURNAL (the auditable source of truth), never from res.X alone.
    records = load_journal(work / "search.ndjson")
    front = pareto_front(records, baseline)
    out = work / "pareto_front.json"
    out.write_text(json.dumps(front, indent=2))
    n_ok = sum(r["status"] == "ok" for r in records)
    print(f"\n=== PARETO FRONT: {len(front)} models ({n_ok} evaluated ok, {len(records)} total) -> {out} ===")
    for r in front:
        print(
            f"  gflops={r['gflops']:6.2f} params={r['params']:>9,} "
            f"ap_small={r['ap_small']:.4f} map5095={r['map5095']:.4f} key={r['key']}"
        )
    print("\nSend pareto_front.json (or this printout) to Claude for the winner-selection step.")


if __name__ == "__main__":
    main()
