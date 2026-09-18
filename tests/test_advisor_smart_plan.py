from src.advisor.plan import generate_plan
from src.advisor.profile import InvestorProfile, Target


def _setup(monkeypatch, tmp_path):
    monkeypatch.setattr("src.advisor.plan._plans_dir", lambda: str(tmp_path))
    profile = InvestorProfile(owner_id="default", risk_tolerance="moderate", investable_cash=1000.0)
    target = Target(weights={"equity": 1.0}, max_position_pct=0.5, locked=True)

    class _Repo:
        def load(self, owner_id): return (profile, target)

    class _Svc:
        def get_portfolio_snapshot(self): return {"total_cash": 0.0, "accounts": []}

    class _Board:
        stocks = [type("S", (), {"ticker": "INGR", "name": "Ingredion", "strategy_count": 4, "sector": "Consumer"})(),
                  type("S", (), {"ticker": "TRV", "name": "Travelers", "strategy_count": 4, "sector": "Financials"})()]

    return _Repo(), _Svc(), lambda: _Board()


def test_smart_uses_llm_order_and_sets_engine(monkeypatch, tmp_path):
    repo, svc, board = _setup(monkeypatch, tmp_path)
    # LLM prefers TRV first, and provides a rationale.
    select_fn = lambda cands, profile, budget, cap: (
        sorted(cands, key=lambda c: 0 if c.ticker == "TRV" else 1), "AI says quality.")
    plan = generate_plan(owner_id="default", repo=repo, portfolio_service=svc,
                         board_provider=board, watchlist_provider=lambda: [],
                         smart=True, select_fn=select_fn)
    assert plan.engine == "ai"
    assert plan.rationale == "AI says quality."
    buys = [a for a in plan.actions if a.side == "buy" and a.symbol]
    assert buys[0].symbol == "TRV"  # LLM order respected


def test_smart_falls_back_when_selector_returns_none(monkeypatch, tmp_path):
    repo, svc, board = _setup(monkeypatch, tmp_path)
    plan = generate_plan(owner_id="default", repo=repo, portfolio_service=svc,
                         board_provider=board, watchlist_provider=lambda: [],
                         smart=True, select_fn=lambda *a, **k: None)  # LLM unavailable
    assert plan.engine == "deterministic"
    buys = [a for a in plan.actions if a.side == "buy" and a.symbol]
    assert buys[0].symbol == "INGR"  # deterministic conviction order (board order)
    assert "your locked target" in plan.rationale  # templated rationale


def test_deterministic_by_default(monkeypatch, tmp_path):
    repo, svc, board = _setup(monkeypatch, tmp_path)
    plan = generate_plan(owner_id="default", repo=repo, portfolio_service=svc,
                         board_provider=board, watchlist_provider=lambda: [])
    assert plan.engine == "deterministic"
