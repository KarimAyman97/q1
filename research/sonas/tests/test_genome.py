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


def test_validation_rejects_non_bool_flag():
    g = Genome.random(random.Random(5))
    with pytest.raises(ValueError):
        Genome.from_dict({**g.to_dict(), "p2": 5})
