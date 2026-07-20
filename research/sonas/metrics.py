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
