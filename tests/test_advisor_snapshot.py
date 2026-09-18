from src.advisor.profile import InvestorProfile
from src.advisor.snapshot import get_current


class _FakeSvc:
    def __init__(self, payload):
        self._payload = payload

    def get_portfolio_snapshot(self):
        return self._payload


def test_deploy_mode_when_empty():
    svc = _FakeSvc({"total_cash": 0.0, "accounts": []})
    profile = InvestorProfile(investable_cash=25000.0)
    cur = get_current(profile, portfolio_service=svc)
    assert cur.mode == "deploy"
    assert cur.positions == []
    assert cur.base == 25000.0
    assert cur.cash == 25000.0


def test_rebalance_mode_with_holdings():
    svc = _FakeSvc({
        "total_cash": 1000.0,
        "accounts": [{
            "positions": [
                {"symbol": "AAPL", "quantity": 10, "market_value_base": 2000.0},
                {"symbol": "BTC-USD", "quantity": 0.1, "market_value_base": 3000.0},
            ]
        }],
    })
    profile = InvestorProfile(investable_cash=500.0)
    cur = get_current(profile, portfolio_service=svc, sector_lookup={"AAPL": "Technology"})
    assert cur.mode == "rebalance"
    assert cur.base == 2000.0 + 3000.0 + 1000.0 + 500.0  # holdings + snapshot cash + new cash
    syms = {p.symbol: p for p in cur.positions}
    assert syms["AAPL"].asset_class == "equity"
    assert syms["AAPL"].sector == "Technology"
    assert syms["BTC-USD"].asset_class == "crypto"
    assert cur.cash == 1000.0 + 500.0
