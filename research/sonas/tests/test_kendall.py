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


def test_below_panel_size_fails_even_with_perfect_agreement():
    short = [rec(str(i), 0.1 + i / 100) for i in range(5)]
    long = [rec(str(i), 0.2 + i / 50) for i in range(5)]
    v = kendall_verdict(short, long)  # default min_n == PANEL_SIZE (10) > n == 5
    assert v["tau"] > 0.99 and not v["passed"]
