from src.advisor.gap import compute_gap
from src.advisor.profile import Target
from src.advisor.snapshot import CurrentPicture, Position


def test_class_deltas():
    cur = CurrentPicture(mode="rebalance", base=100.0, cash=10.0, positions=[
        Position("AAPL", "equity", "Technology", 90.0),
    ])
    target = Target(weights={"equity": 0.7, "cash": 0.3})
    gap = compute_gap(cur, target)
    assert round(gap.class_delta["equity"], 2) == -20.0   # have 90, want 70 -> sell 20
    assert round(gap.class_delta["cash"], 2) == 20.0      # have 10, want 30


def test_position_over_cap_flagged():
    cur = CurrentPicture(mode="rebalance", base=100.0, cash=0.0, positions=[
        Position("OKLO", "equity", "Energy", 22.0),
        Position("INGR", "equity", "Consumer", 78.0),
    ])
    target = Target(weights={"equity": 1.0}, max_position_pct=0.15)
    gap = compute_gap(cur, target)
    assert round(gap.position_overage["OKLO"], 2) == 7.0  # 22 - 15% of 100


def test_sector_over_cap_flagged():
    cur = CurrentPicture(mode="rebalance", base=100.0, cash=0.0, positions=[
        Position("CNQ.TO", "equity", "Energy", 20.0),
        Position("WCP.TO", "equity", "Energy", 14.0),
        Position("INGR", "equity", "Consumer", 66.0),
    ])
    target = Target(weights={"equity": 1.0}, max_position_pct=1.0, max_sector_pct=0.30)
    gap = compute_gap(cur, target)
    assert round(gap.sector_overage["Energy"], 2) == 4.0  # 34 - 30% of 100


def test_unknown_sector_not_flagged():
    # "Unknown" is missing data, not a real sector — it must not trigger a trim.
    cur = CurrentPicture(mode="rebalance", base=100.0, cash=0.0, positions=[
        Position("A", "equity", "Unknown", 50.0),
        Position("B", "equity", "Unknown", 50.0),
    ])
    target = Target(weights={"equity": 1.0}, max_position_pct=1.0, max_sector_pct=0.30)
    gap = compute_gap(cur, target)
    assert "Unknown" not in gap.sector_overage
