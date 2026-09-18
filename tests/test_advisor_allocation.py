from src.advisor.allocation import suggest_allocation
from src.advisor.profile import InvestorProfile


def test_weights_sum_to_one_and_validate():
    for risk in ("conservative", "moderate", "aggressive"):
        t = suggest_allocation(InvestorProfile(risk_tolerance=risk))
        t.validate()  # raises if broken
        assert abs(sum(t.weights.values()) - 1.0) < 1e-9


def test_aggressive_has_more_equity_than_conservative():
    agg = suggest_allocation(InvestorProfile(risk_tolerance="aggressive"))
    con = suggest_allocation(InvestorProfile(risk_tolerance="conservative"))
    assert agg.weights["equity"] > con.weights["equity"]


def test_longer_horizon_lifts_equity():
    short = suggest_allocation(InvestorProfile(risk_tolerance="moderate", horizon_years=2))
    long = suggest_allocation(InvestorProfile(risk_tolerance="moderate", horizon_years=25))
    assert long.weights["equity"] >= short.weights["equity"]


def test_exclude_crypto_constraint():
    t = suggest_allocation(InvestorProfile(risk_tolerance="aggressive",
                                           constraints={"exclude_crypto": True}))
    assert t.weights.get("crypto", 0.0) == 0.0


def test_suggested_target_is_unlocked():
    assert suggest_allocation(InvestorProfile()).locked is False
