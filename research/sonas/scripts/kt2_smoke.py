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
