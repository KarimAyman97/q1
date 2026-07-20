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


def build_model_dict(genome: Genome, nc: int = 10) -> dict:
    w2, w3, w4, w5 = WIDTH_PROFILES[genome.widths]
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
        # Head-only channels: stock yolo26's P2/P3 head blocks run at half the backbone c2/c3
        # width (e.g. yolo26-p2.yaml row 19 C3k2[128,...] and yolo26.yaml row 16 C3k2[256,...]
        # at 0.25 width = 32/64), unlike the P4/P5 head blocks which reuse c4/c5 directly.
        "h2": _div8(32 * w2),
        "h3": _div8(64 * w3),
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
    b_c2 = add(backbone, [-1, genome.d2, genome.block, [ch["c2"], True]])
    add(backbone, [-1, 1, "Conv", [ch["p3"], 3, 2]])  # P3/8
    b_c3 = add(backbone, [-1, genome.d3, genome.block, [ch["c3"], True]])
    add(backbone, [-1, 1, genome.down, [ch["p4"], 3, 2]])  # P4/16
    b_c4 = add(backbone, [-1, genome.d4, genome.block, [ch["c4"], True]])
    add(backbone, [-1, 1, genome.down, [ch["p5"], 3, 2]])  # P5/32
    add(backbone, [-1, genome.d5, genome.block, [ch["c5"], True]])
    top = add(backbone, [-1, 1, "SPPF", [ch["c5"], 5, 3, True]])
    if genome.psa:
        top = add(backbone, [-1, 2, "C2PSA", [ch["c5"]]])

    # Head: top-down (FPN) then bottom-up (PAN), P2/P5 branches optional.
    add(head, [-1, 1, "nn.Upsample", [None, 2, "nearest"]])
    add(head, [[-1, b_c4], 1, "Concat", [1]])
    h4 = add(head, [-1, genome.head_depth, "C3k2", [ch["c4"], True]])

    add(head, [-1, 1, "nn.Upsample", [None, 2, "nearest"]])
    p3_cat = [-1, b_c3]
    if genome.fusion == "ctx_p3":  # extra global-context path: top features upsampled x4 straight into P3
        ctx = add(head, [top, 1, "nn.Upsample", [None, 4, "nearest"]])
        p3_cat = [-2, b_c3, ctx]  # -2: the x2-upsample two rows back (the ctx row is -1)
    add(head, [p3_cat, 1, "Concat", [1]])
    h3 = add(head, [-1, genome.head_depth, "C3k2", [ch["h3"], True]])

    detect_from = []
    smallest = h3
    if genome.p2:
        add(head, [-1, 1, "nn.Upsample", [None, 2, "nearest"]])
        add(head, [[-1, b_c2], 1, "Concat", [1]])
        h2 = add(head, [-1, genome.head_depth, "C3k2", [ch["h2"], True]])
        detect_from.append(h2)
        add(head, [-1, 1, "Conv", [ch["h2"], 3, 2]])  # back down to P3
        add(head, [[-1, h3], 1, "Concat", [1]])
        smallest = add(head, [-1, genome.head_depth, "C3k2", [ch["h3"], True]])
    detect_from.append(smallest)

    add(head, [-1, 1, "Conv", [ch["h3"], 3, 2]])  # P3 -> P4 (smallest is always the previous row)
    p4_cat = [-1, h4]
    if genome.fusion == "p4skip":  # extra raw-backbone skip at P4
        p4_cat = [-1, h4, b_c4]
    add(head, [p4_cat, 1, "Concat", [1]])
    h4b = add(head, [-1, genome.head_depth, "C3k2", [ch["c4"], True]])
    detect_from.append(h4b)

    if genome.p5:
        add(head, [-1, 1, "Conv", [ch["c4"], 3, 2]])  # P4 -> P5
        add(head, [[-1, top], 1, "Concat", [1]])
        h5 = add(head, [-1, 1, "C3k2", [ch["c5"], True]])
        detect_from.append(h5)

    add(head, [detect_from, 1, "Detect", ["nc"]])  # the string "nc" resolves inside parse_model, as in stock YAMLs
    return {
        "nc": nc,
        "end2end": bool(genome.end2end),
        "reg_max": 1,
        "scales": {"n": [1.0, 1.0, 1024]},
        "scale": "n",
        "backbone": backbone,
        "head": head,
    }


def write_model_yaml(genome: Genome, path, nc: int = 10) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(build_model_dict(genome, nc=nc), sort_keys=False))
    return path
