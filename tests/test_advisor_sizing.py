from src.advisor.candidates import Candidate
from src.advisor.gap import Gap
from src.advisor.profile import Target
from src.advisor.sizing import size_positions
from src.advisor.snapshot import CurrentPicture, Position


def _target(**kw):
    kw.setdefault("weights", {"equity": 1.0})
    return Target(**kw)


def test_trims_only_over_cap_positions():
    cur = CurrentPicture(mode="rebalance", base=100.0, cash=0.0, positions=[
        Position("OKLO", "equity", "Energy", 22.0),   # over 15 cap
        Position("INGR", "equity", "Consumer", 12.0), # under cap -> left alone
    ])
    gap = Gap(class_delta={"equity": 0.0}, position_overage={"OKLO": 7.0}, sector_overage={})
    actions = size_positions(gap, candidates=[], current=cur, target=_target(max_position_pct=0.15))
    trims = {a.symbol: a for a in actions if a.side == "trim"}
    assert "OKLO" in trims and round(trims["OKLO"].amount, 2) == 7.0
    assert "INGR" not in trims  # conviction bet under cap is respected


def test_buys_fill_equity_delta_under_caps():
    cur = CurrentPicture(mode="deploy", base=100.0, cash=100.0, positions=[])
    gap = Gap(class_delta={"equity": 100.0}, position_overage={}, sector_overage={})
    cands = [Candidate("INGR", strategy_count=4, sector="Consumer"),
             Candidate("TRV", strategy_count=4, sector="Financials")]
    actions = size_positions(gap, candidates=cands, current=cur, target=_target(max_position_pct=0.15))
    buys = [a for a in actions if a.side == "buy"]
    assert all(a.amount <= 15.0 + 1e-6 for a in buys)      # per-position cap = 15% of 100
    assert sum(a.amount for a in buys) <= 100.0 + 1e-6     # within cash/delta


def test_non_equity_delta_becomes_dollar_target_action():
    cur = CurrentPicture(mode="deploy", base=100.0, cash=100.0, positions=[])
    gap = Gap(class_delta={"fixed_income": 20.0}, position_overage={}, sector_overage={})
    actions = size_positions(gap, candidates=[], current=cur, target=_target(weights={"fixed_income": 1.0}))
    fi = [a for a in actions if a.asset_class == "fixed_income"]
    assert fi and fi[0].side == "buy" and round(fi[0].amount, 2) == 20.0
    assert fi[0].symbol == ""  # dollar target, not a specific security


def test_unallocated_when_no_candidates():
    cur = CurrentPicture(mode="deploy", base=100.0, cash=100.0, positions=[])
    gap = Gap(class_delta={"equity": 100.0}, position_overage={}, sector_overage={})
    actions = size_positions(gap, candidates=[], current=cur, target=_target(max_position_pct=0.15))
    assert any(a.side == "unallocated" and a.asset_class == "equity" for a in actions)
