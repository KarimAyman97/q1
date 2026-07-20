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
        for name in ("p2", "p5", "end2end", "psa"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}")
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
