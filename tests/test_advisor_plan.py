import pytest

from src.advisor.plan import Plan, render_markdown, generate_plan
from src.advisor.profile import InvestorProfile, Target
from src.advisor.sizing import Action


def test_render_contains_disclaimer_and_actions():
    plan = Plan(mode="deploy", base=1000.0,
                target={"equity": 0.7, "cash": 0.3},
                actions=[Action("buy", "equity", "INGR", 150.0, "fills equity target")],
                rationale="Deploying cash into your locked target.")
    md = render_markdown(plan)
    assert "Not financial advice" in md
    assert "does not account for taxes" in md.lower()
    assert "INGR" in md
    assert "## " in md  # has a heading


def test_generate_plan_deterministic_end_to_end(tmp_path, monkeypatch):
    # Fake repo returns a locked target + profile; fake services avoid network/LLM.
    profile = InvestorProfile(owner_id="default", risk_tolerance="moderate", investable_cash=1000.0)
    target = Target(weights={"equity": 1.0}, max_position_pct=0.5, locked=True)

    class _Repo:
        def load(self, owner_id): return (profile, target)

    class _Svc:
        def get_portfolio_snapshot(self): return {"total_cash": 0.0, "accounts": []}

    class _Board:
        stocks = []

    monkeypatch.setattr("src.advisor.plan._plans_dir", lambda: str(tmp_path))
    plan = generate_plan(
        owner_id="default",
        repo=_Repo(),
        portfolio_service=_Svc(),
        board_provider=lambda: _Board(),
        watchlist_provider=lambda: [],
    )
    assert plan.mode == "deploy"
    assert plan.base == 1000.0
    # No candidates -> unallocated note, but a valid plan still returns.
    assert any(a.side == "unallocated" for a in plan.actions)
    # It saved a markdown file.
    saved = list(tmp_path.glob("plan_*.md"))
    assert len(saved) == 1


def test_generate_plan_requires_locked_target():
    profile = InvestorProfile()
    unlocked = Target(weights={"equity": 1.0}, locked=False)

    class _Repo:
        def load(self, owner_id): return (profile, unlocked)

    with pytest.raises(ValueError, match="lock"):
        generate_plan(owner_id="default", repo=_Repo(),
                      portfolio_service=type("S", (), {"get_portfolio_snapshot": lambda self: {"accounts": []}})(),
                      board_provider=lambda: type("B", (), {"stocks": []})(),
                      watchlist_provider=lambda: [])


def test_generate_plan_missing_profile_raises():
    class _Repo:
        def load(self, owner_id): return None
    with pytest.raises(LookupError):
        generate_plan(owner_id="default", repo=_Repo())
