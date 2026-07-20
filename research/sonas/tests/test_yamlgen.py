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
