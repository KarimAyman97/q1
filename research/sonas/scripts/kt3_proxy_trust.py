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
from research.sonas.proxy import _key, append_eval, load_journal, run_proxy

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


def kendall_verdict(records_short, records_long, threshold: float = TAU_GATE, min_n: int = PANEL_SIZE) -> dict:
    by_key_s = {r["key"]: r["ap_small"] for r in records_short if r["status"] == "ok"}
    by_key_l = {r["key"]: r["ap_small"] for r in records_long if r["status"] == "ok"}
    keys = sorted(set(by_key_s) & set(by_key_l))
    tau, p = kendalltau([by_key_s[k] for k in keys], [by_key_l[k] for k in keys])
    n = len(keys)
    return {"tau": float(tau), "p": float(p), "n": n, "passed": bool(tau >= threshold and n >= min_n)}


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
    done = {r["key"] for r in load_journal(journal) if r["status"] == "ok"}
    for g in panel():
        if _key(g) in done:
            continue  # resume-safe after Colab disconnects
        rec = run_proxy(g, data="VisDrone.yaml", gt_json=work / "gt.json", project=work / f"kt3_{args.stage}",
                        epochs=epochs, device=args.device)
        append_eval(journal, rec)
        print(rec["key"], rec["status"], rec.get("ap_small"))


if __name__ == "__main__":
    main()
