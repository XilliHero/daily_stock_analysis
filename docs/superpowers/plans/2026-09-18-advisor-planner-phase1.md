# Advisor / Whole-Portfolio Planner — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic, zero-LLM whole-portfolio planner: an investor profile, a suggested→locked target allocation, and a plan (buys/trims) that reaches the target — with a "conviction + guardrails" sizing rule, working in both rebalance and deploy-from-cash modes, surfaced on a new dashboard Plan page.

**Architecture:** A new `src/advisor/` package of small, single-purpose, deterministic modules (profile, classify, allocation, snapshot, candidates, gap, sizing, plan, repository). A FastAPI router exposes profile CRUD + plan generation. A React Plan page drives it. No LLM in Phase 1 — the AdvisorAgent (Phase 2) will wrap these same functions as tools and fall back to them.

**Tech Stack:** Python 3.12 (framework python at `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`, no venv), pytest, SQLAlchemy (SQLite `data/stock_analysis.db`), FastAPI, React + Vite + TypeScript (`apps/dsa-web`), vitest.

**Design spec:** `docs/superpowers/specs/2026-07-18-advisor-planner-design.md`

**Run tests:** `python3 -m pytest tests/test_advisor_*.py -q` · frontend: `cd apps/dsa-web && npx vitest run`

---

## Conventions (read once)

- Single-user v1: everything is keyed by `owner_id`, defaulting to the constant `DEFAULT_OWNER_ID = "default"` (mirrors how `portfolio` endpoints leave `owner_id` optional). Multi-user is out of scope.
- Money is in the profile's `base_currency` (default `"USD"`). Deploy-from-cash (the path that runs today, since the portfolio is empty) needs no FX. Rebalance mode reads `market_value_base` from the snapshot and treats it as base-currency; cross-currency normalization is a **known deferred item** (documented alongside taxes), acceptable because the live portfolio is empty.
- Asset classes (the fixed vocabulary): `"equity"`, `"fixed_income"`, `"cash"`, `"crypto"`, `"commodity"`.
- Every module is import-light and side-effect-free except `repository.py` (DB), `snapshot.py`/`candidates.py` (call existing services, but accept injected fakes for tests), and `plan.py` (writes `output/plans/`).

## File Structure

**Create:**
- `src/advisor/__init__.py` — package marker + `DEFAULT_OWNER_ID`, `ASSET_CLASSES`.
- `src/advisor/profile.py` — `InvestorProfile`, `Target` dataclasses + validation.
- `src/advisor/classify.py` — `classify_asset(ticker)`, `sector_of(ticker, lookup)`.
- `src/advisor/allocation.py` — `suggest_allocation(profile) -> Target`.
- `src/advisor/snapshot.py` — `Position`, `CurrentPicture`, `get_current(profile, portfolio_service=None, sector_lookup=None)`.
- `src/advisor/candidates.py` — `Candidate`, `fetch_candidates(limit, ...)`.
- `src/advisor/gap.py` — `Gap`, `compute_gap(current, target)`.
- `src/advisor/sizing.py` — `Action`, `size_positions(gap, candidates, current, target)`.
- `src/advisor/plan.py` — `Plan`, `render_markdown(plan)`, `generate_plan(owner_id, ...)`.
- `src/advisor/repository.py` — `InvestorProfileRepository` (load/save profile+target).
- `api/v1/schemas/advisor.py` — pydantic request/response models.
- `api/v1/endpoints/advisor.py` — profile GET/PUT, plan POST/GET.
- `apps/dsa-web/src/api/advisor.ts` — API client.
- `apps/dsa-web/src/pages/PlanPage.tsx` — the Plan page.
- `apps/dsa-web/src/pages/__tests__/PlanPage.test.tsx` — component test.
- Tests: `tests/test_advisor_profile.py`, `_classify.py`, `_allocation.py`, `_snapshot.py`, `_candidates.py`, `_gap.py`, `_sizing.py`, `_plan.py`, `_repository.py`, `_api.py`.

**Modify:**
- `src/storage.py` — add `InvestorProfile` SQLAlchemy model (auto-created by `Base.metadata.create_all`).
- `api/v1/router.py` — import + `include_router(advisor.router, prefix="/advisor")`.
- `apps/dsa-web/src/App.tsx` — import `PlanPage`, add `<Route path="/plan" ...>`.
- The dashboard nav component (search `to="/portfolio"` under `apps/dsa-web/src/components`) — add a "Plan" link next to Portfolio.

---

## Task 1: Profile & Target data models

**Files:**
- Create: `src/advisor/__init__.py`, `src/advisor/profile.py`
- Test: `tests/test_advisor_profile.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_advisor_profile.py
import pytest
from src.advisor.profile import InvestorProfile, Target, ASSET_CLASSES, DEFAULT_OWNER_ID


def test_profile_defaults():
    p = InvestorProfile()
    assert p.owner_id == DEFAULT_OWNER_ID
    assert p.risk_tolerance == "moderate"
    assert p.horizon_years == 10
    assert p.base_currency == "USD"
    p.validate()  # defaults are valid


def test_profile_rejects_bad_risk():
    with pytest.raises(ValueError):
        InvestorProfile(risk_tolerance="yolo").validate()


def test_target_weights_must_sum_to_one():
    Target(weights={"equity": 0.7, "fixed_income": 0.2, "cash": 0.1}).validate()
    with pytest.raises(ValueError):
        Target(weights={"equity": 0.7, "cash": 0.1}).validate()  # sums to 0.8


def test_target_rejects_unknown_class_and_bad_caps():
    with pytest.raises(ValueError):
        Target(weights={"stonks": 1.0}).validate()
    with pytest.raises(ValueError):
        Target(weights={"equity": 1.0}, max_position_pct=1.5).validate()


def test_target_known_classes_subset():
    assert set(ASSET_CLASSES) == {"equity", "fixed_income", "cash", "crypto", "commodity"}
```

- [ ] **Step 2: Run it, expect failure**

Run: `python3 -m pytest tests/test_advisor_profile.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.advisor'`.

- [ ] **Step 3: Implement**

```python
# src/advisor/__init__.py
# -*- coding: utf-8 -*-
"""Advisor package — deterministic whole-portfolio planner (Phase 1)."""

DEFAULT_OWNER_ID = "default"
ASSET_CLASSES = ("equity", "fixed_income", "cash", "crypto", "commodity")
```

```python
# src/advisor/profile.py
# -*- coding: utf-8 -*-
"""Investor profile + target allocation — pure data with validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.advisor import ASSET_CLASSES, DEFAULT_OWNER_ID

RISK_LEVELS = ("conservative", "moderate", "aggressive")
_WEIGHT_TOLERANCE = 1e-6


@dataclass
class Target:
    """Target allocation the planner aims for. Weights sum to 1.0."""

    weights: Dict[str, float] = field(default_factory=dict)
    max_position_pct: float = 0.15  # guardrail: max % of portfolio in one stock
    max_sector_pct: float = 0.30    # guardrail: max % in one equity sector
    locked: bool = False

    def validate(self) -> None:
        for cls in self.weights:
            if cls not in ASSET_CLASSES:
                raise ValueError(f"unknown asset class: {cls}")
        for cls, w in self.weights.items():
            if w < 0:
                raise ValueError(f"negative weight for {cls}: {w}")
        total = sum(self.weights.values())
        if abs(total - 1.0) > _WEIGHT_TOLERANCE:
            raise ValueError(f"weights must sum to 1.0, got {total:.6f}")
        for name, cap in (("max_position_pct", self.max_position_pct),
                          ("max_sector_pct", self.max_sector_pct)):
            if not (0 < cap <= 1.0):
                raise ValueError(f"{name} must be in (0, 1], got {cap}")


@dataclass
class InvestorProfile:
    owner_id: str = DEFAULT_OWNER_ID
    risk_tolerance: str = "moderate"
    horizon_years: int = 10
    goals: List[str] = field(default_factory=list)
    constraints: Dict[str, bool] = field(default_factory=dict)  # e.g. {"exclude_crypto": True}
    investable_cash: float = 0.0
    base_currency: str = "USD"

    def validate(self) -> None:
        if self.risk_tolerance not in RISK_LEVELS:
            raise ValueError(f"risk_tolerance must be one of {RISK_LEVELS}")
        if self.horizon_years <= 0:
            raise ValueError("horizon_years must be positive")
        if self.investable_cash < 0:
            raise ValueError("investable_cash must be >= 0")
```

- [ ] **Step 4: Run it, expect pass**

Run: `python3 -m pytest tests/test_advisor_profile.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/advisor/__init__.py src/advisor/profile.py tests/test_advisor_profile.py
git commit -m "feat(advisor): investor profile + target allocation models"
```

---

## Task 2: Storage table + profile repository

**Files:**
- Modify: `src/storage.py` (add `InvestorProfile` model — note the name clash below)
- Create: `src/advisor/repository.py`
- Test: `tests/test_advisor_repository.py`

> **Naming note:** the dataclass in `profile.py` is `InvestorProfile`. To avoid confusion, name the **SQLAlchemy model** `InvestorProfileRecord` in `storage.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_advisor_repository.py
import os
import tempfile

from src.advisor.profile import InvestorProfile, Target
from src.advisor.repository import InvestorProfileRepository


def _fresh_repo():
    # Isolated in-memory-ish DB per test via a temp file + a fresh DatabaseManager.
    from src.storage import DatabaseManager
    DatabaseManager._instance = None  # reset singleton
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    dbm = DatabaseManager(db_url=f"sqlite:///{tmp.name}")
    return InvestorProfileRepository(db_manager=dbm), tmp.name


def test_load_returns_none_when_absent():
    repo, path = _fresh_repo()
    try:
        assert repo.load("nobody") is None
    finally:
        os.unlink(path)


def test_save_then_load_roundtrip():
    repo, path = _fresh_repo()
    try:
        profile = InvestorProfile(owner_id="default", risk_tolerance="aggressive",
                                  horizon_years=20, goals=["growth"], investable_cash=25000.0)
        target = Target(weights={"equity": 0.8, "cash": 0.2}, max_position_pct=0.2, locked=True)
        repo.save(profile, target)
        loaded_profile, loaded_target = repo.load("default")
        assert loaded_profile.risk_tolerance == "aggressive"
        assert loaded_profile.investable_cash == 25000.0
        assert loaded_target.weights == {"equity": 0.8, "cash": 0.2}
        assert loaded_target.locked is True
    finally:
        os.unlink(path)


def test_save_is_upsert():
    repo, path = _fresh_repo()
    try:
        repo.save(InvestorProfile(owner_id="default", horizon_years=5), Target(weights={"cash": 1.0}))
        repo.save(InvestorProfile(owner_id="default", horizon_years=30), Target(weights={"cash": 1.0}))
        loaded_profile, _ = repo.load("default")
        assert loaded_profile.horizon_years == 30  # updated, not duplicated
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run it, expect failure**

Run: `python3 -m pytest tests/test_advisor_repository.py -q`
Expected: FAIL — `ImportError`/`AttributeError` (`InvestorProfileRepository` / `InvestorProfileRecord` missing).

- [ ] **Step 3a: Add the model to `src/storage.py`**

Insert after the `PortfolioFxRate` model (near line 597), following the existing column style:

```python
class InvestorProfileRecord(Base):
    """Advisor investor profile + locked target allocation (one row per owner)."""

    __tablename__ = 'investor_profiles'

    owner_id = Column(String(64), primary_key=True)
    profile_json = Column(Text, nullable=False)
    target_json = Column(Text, nullable=False)
    locked = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
```

(`Column`, `String`, `Text`, `Boolean`, `DateTime`, `datetime` are already imported in `storage.py`.)

- [ ] **Step 3b: Implement the repository**

```python
# src/advisor/repository.py
# -*- coding: utf-8 -*-
"""Persistence for the investor profile + locked target (one row per owner)."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional, Tuple

from src.advisor.profile import InvestorProfile, Target
from src.storage import DatabaseManager, InvestorProfileRecord


class InvestorProfileRepository:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager.get_instance()

    def load(self, owner_id: str) -> Optional[Tuple[InvestorProfile, Target]]:
        with self.db.session_scope() as session:
            row = session.get(InvestorProfileRecord, owner_id)
            if row is None:
                return None
            profile = InvestorProfile(**json.loads(row.profile_json))
            target = Target(**json.loads(row.target_json))
            return profile, target

    def save(self, profile: InvestorProfile, target: Target) -> None:
        with self.db.session_scope() as session:
            row = session.get(InvestorProfileRecord, profile.owner_id)
            payload = dict(
                profile_json=json.dumps(asdict(profile)),
                target_json=json.dumps(asdict(target)),
                locked=bool(target.locked),
            )
            if row is None:
                session.add(InvestorProfileRecord(owner_id=profile.owner_id, **payload))
            else:
                row.profile_json = payload["profile_json"]
                row.target_json = payload["target_json"]
                row.locked = payload["locked"]
```

- [ ] **Step 4: Run it, expect pass**

Run: `python3 -m pytest tests/test_advisor_repository.py -q` → PASS.
(`DatabaseManager.session_scope()` at `src/storage.py:846` commits on clean exit and rolls back on exception — confirmed — so `save()` persists.)

- [ ] **Step 5: Commit**

```bash
git add src/storage.py src/advisor/repository.py tests/test_advisor_repository.py
git commit -m "feat(advisor): investor_profiles table + profile repository (upsert)"
```

---

## Task 3: Asset-class + sector classifier

**Files:**
- Create: `src/advisor/classify.py`
- Test: `tests/test_advisor_classify.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_advisor_classify.py
from src.advisor.classify import classify_asset, sector_of


def test_classify_asset():
    assert classify_asset("AAPL") == "equity"
    assert classify_asset("cnq.to") == "equity"       # Canadian equity, case-insensitive
    assert classify_asset("BTC-USD") == "crypto"
    assert classify_asset("BND") == "fixed_income"
    assert classify_asset("CORN") == "commodity"
    assert classify_asset("CASH") == "cash"


def test_sector_lookup_with_default():
    lookup = {"CNQ.TO": "Energy", "AAPL": "Technology"}
    assert sector_of("CNQ.TO", lookup) == "Energy"
    assert sector_of("AAPL", lookup) == "Technology"
    assert sector_of("ZZZZ", lookup) == "Unknown"     # unknown -> Unknown
    assert sector_of("AAPL", None) == "Unknown"        # no lookup -> Unknown
```

- [ ] **Step 2: Run it, expect failure** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/advisor/classify.py
# -*- coding: utf-8 -*-
"""Deterministic ticker → asset class, and sector lookup with a safe default.

Known non-equity tickers are enumerated; everything else defaults to equity.
The lists are intentionally small and easy to extend.
"""

from __future__ import annotations

from typing import Dict, Optional

_CASH = {"CASH", "USD", "$CASH"}
_BOND_ETFS = {"BND", "AGG", "TLT", "IEF", "LQD", "BNDX", "SHY", "TIP", "VGSH", "GOVT"}
_COMMODITY_ETFS = {"CORN", "WEAT", "SOYB", "GLD", "SLV", "USO", "DBC", "UNG", "PDBC"}


def classify_asset(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    if t in _CASH:
        return "cash"
    if t in _BOND_ETFS:
        return "fixed_income"
    if t in _COMMODITY_ETFS:
        return "commodity"
    if t.endswith("-USD"):
        return "crypto"
    return "equity"


def sector_of(ticker: str, lookup: Optional[Dict[str, str]]) -> str:
    if not lookup:
        return "Unknown"
    return lookup.get((ticker or "").strip().upper(), "Unknown")
```

- [ ] **Step 4: Run it, expect pass.**

- [ ] **Step 5: Commit**

```bash
git add src/advisor/classify.py tests/test_advisor_classify.py
git commit -m "feat(advisor): asset-class classifier + sector lookup"
```

---

## Task 4: Suggested allocation from profile

**Files:**
- Create: `src/advisor/allocation.py`
- Test: `tests/test_advisor_allocation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_advisor_allocation.py
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
```

- [ ] **Step 2: Run it, expect failure.**

- [ ] **Step 3: Implement**

```python
# src/advisor/allocation.py
# -*- coding: utf-8 -*-
"""Deterministic profile -> suggested Target allocation (rule table).

This is a transparent heuristic, not investment advice: base equity by risk,
tilt by horizon, small crypto sleeve only for aggressive investors who allow it,
remainder split fixed-income / cash. Caps widen with risk appetite.
"""

from __future__ import annotations

from src.advisor.profile import InvestorProfile, Target

_BASE_EQUITY = {"conservative": 0.45, "moderate": 0.65, "aggressive": 0.80}
_CAPS = {  # (max_position_pct, max_sector_pct)
    "conservative": (0.08, 0.25),
    "moderate": (0.15, 0.30),
    "aggressive": (0.25, 0.40),
}


def _horizon_tilt(years: int) -> float:
    if years <= 3:
        return -0.10
    if years >= 15:
        return 0.10
    return 0.0


def suggest_allocation(profile: InvestorProfile) -> Target:
    equity = _BASE_EQUITY[profile.risk_tolerance] + _horizon_tilt(profile.horizon_years)
    equity = max(0.30, min(0.90, equity))

    crypto = 0.0
    if profile.risk_tolerance == "aggressive" and not profile.constraints.get("exclude_crypto"):
        crypto = 0.05

    remainder = 1.0 - equity - crypto
    # More fixed income for conservative, more cash buffer for short horizons.
    fixed_share = {"conservative": 0.70, "moderate": 0.60, "aggressive": 0.50}[profile.risk_tolerance]
    fixed_income = round(remainder * fixed_share, 4)
    cash = round(remainder - fixed_income, 4)

    weights = {"equity": round(equity, 4), "fixed_income": fixed_income, "cash": cash}
    if crypto:
        weights["crypto"] = crypto
    # Normalize away rounding drift onto the largest bucket.
    drift = round(1.0 - sum(weights.values()), 6)
    weights["equity"] = round(weights["equity"] + drift, 6)

    max_pos, max_sec = _CAPS[profile.risk_tolerance]
    return Target(weights=weights, max_position_pct=max_pos, max_sector_pct=max_sec, locked=False)
```

- [ ] **Step 4: Run it, expect pass.**

- [ ] **Step 5: Commit**

```bash
git add src/advisor/allocation.py tests/test_advisor_allocation.py
git commit -m "feat(advisor): deterministic suggested allocation from profile"
```

---

## Task 5: Current-picture / snapshot adapter (rebalance vs deploy)

**Files:**
- Create: `src/advisor/snapshot.py`
- Test: `tests/test_advisor_snapshot.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_advisor_snapshot.py
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
```

- [ ] **Step 2: Run it, expect failure.**

- [ ] **Step 3: Implement**

```python
# src/advisor/snapshot.py
# -*- coding: utf-8 -*-
"""Current portfolio picture — the single place that decides rebalance vs deploy.

Money formula (both modes): base = holdings_value + snapshot_cash + investable_cash.
Deploy-from-cash is the degenerate case where holdings and snapshot cash are 0.
Reads existing PortfolioService.get_portfolio_snapshot(); a fake is injectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.advisor.classify import classify_asset, sector_of


@dataclass
class Position:
    symbol: str
    asset_class: str
    sector: str
    value: float


@dataclass
class CurrentPicture:
    mode: str                      # "rebalance" | "deploy"
    base: float                    # total investable base in profile currency
    cash: float                    # snapshot cash + new investable cash
    positions: List[Position] = field(default_factory=list)


def get_current(profile, portfolio_service=None,
                sector_lookup: Optional[Dict[str, str]] = None) -> CurrentPicture:
    svc = portfolio_service
    if svc is None:
        from src.services.portfolio_service import PortfolioService
        svc = PortfolioService()

    snap = svc.get_portfolio_snapshot() or {}
    snapshot_cash = float(snap.get("total_cash", 0.0) or 0.0)

    positions: List[Position] = []
    holdings_value = 0.0
    for account in snap.get("accounts", []) or []:
        for row in account.get("positions", []) or []:
            symbol = str(row.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            value = float(row.get("market_value_base", 0.0) or 0.0)
            positions.append(Position(
                symbol=symbol,
                asset_class=classify_asset(symbol),
                sector=sector_of(symbol, sector_lookup),
                value=value,
            ))
            holdings_value += value

    new_cash = float(profile.investable_cash or 0.0)
    base = holdings_value + snapshot_cash + new_cash
    mode = "rebalance" if positions else "deploy"
    return CurrentPicture(mode=mode, base=base, cash=snapshot_cash + new_cash, positions=positions)
```

- [ ] **Step 4: Run it, expect pass.**

- [ ] **Step 5: Commit**

```bash
git add src/advisor/snapshot.py tests/test_advisor_snapshot.py
git commit -m "feat(advisor): current-picture adapter (rebalance vs deploy, one money formula)"
```

---

## Task 6: Equity buy candidates

**Files:**
- Create: `src/advisor/candidates.py`
- Test: `tests/test_advisor_candidates.py`

The real sources are `build_overlap_board()` (`src/scanner/overlap.py`), deep-dive results, and the watchlist. To stay deterministic/offline, `fetch_candidates` accepts injected providers.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_advisor_candidates.py
from src.advisor.candidates import Candidate, fetch_candidates


class _Stock:
    def __init__(self, ticker, name, strategy_count, sector="Unknown"):
        self.ticker = ticker
        self.name = name
        self.strategy_count = strategy_count
        self.sector = sector


class _Board:
    def __init__(self, stocks):
        self.stocks = stocks


def test_merges_and_ranks_by_conviction():
    board = _Board([_Stock("CNQ.TO", "Cdn Natural", 3, "Energy"),
                    _Stock("INGR", "Ingredion", 4, "Consumer")])
    cands = fetch_candidates(limit=10, board_provider=lambda: board,
                             watchlist_provider=lambda: ["OKLO"])
    tickers = [c.ticker for c in cands]
    assert tickers[0] == "INGR"          # 4/4 ranks above 3/4
    assert "CNQ.TO" in tickers
    assert "OKLO" in tickers             # watchlist adds names not on the board


def test_dedup_and_limit():
    board = _Board([_Stock("INGR", "Ingredion", 4)])
    cands = fetch_candidates(limit=1, board_provider=lambda: board,
                             watchlist_provider=lambda: ["INGR", "OKLO"])
    assert len(cands) == 1
    assert cands[0].ticker == "INGR"     # dedup: INGR not repeated; limit respected


def test_candidate_carries_verdict_when_present():
    board = _Board([_Stock("INGR", "Ingredion", 4)])
    verdicts = {"INGR": {"verdict": "Buy", "sentiment": "Bullish"}}
    cands = fetch_candidates(limit=5, board_provider=lambda: board,
                             watchlist_provider=lambda: [], verdict_provider=lambda t: verdicts.get(t))
    assert cands[0].verdict == "Buy"
```

- [ ] **Step 2: Run it, expect failure.**

- [ ] **Step 3: Implement**

```python
# src/advisor/candidates.py
# -*- coding: utf-8 -*-
"""Equity buy candidates, merged from the overlap board + watchlist + deep-dive.

All external sources are injectable so the merge/rank logic is testable offline.
Ranking: higher strategy_count first (conviction), then board order, watchlist last.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from src.advisor.classify import classify_asset


@dataclass
class Candidate:
    ticker: str
    name: str = ""
    strategy_count: int = 0
    sector: str = "Unknown"
    verdict: str = ""
    sentiment: str = ""


def _default_board():
    from src.scanner.overlap import build_overlap_board
    return build_overlap_board()


def _default_watchlist():
    from src.scanner.watchlist import get_watchlist_tickers  # parses STOCK_LIST -> list[str]
    return get_watchlist_tickers()


def fetch_candidates(limit: int = 5,
                     board_provider: Optional[Callable[[], object]] = None,
                     watchlist_provider: Optional[Callable[[], List[str]]] = None,
                     verdict_provider: Optional[Callable[[str], Optional[dict]]] = None) -> List[Candidate]:
    if limit <= 0:
        return []
    board = (board_provider or _default_board)()
    stocks = list(getattr(board, "stocks", []) or [])
    # Sort a copy by conviction desc, preserving original order within a tier.
    ranked = sorted(enumerate(stocks), key=lambda p: (-int(getattr(p[1], "strategy_count", 0)), p[0]))

    out: List[Candidate] = []
    seen = set()

    def _add(ticker, name="", strategy_count=0, sector="Unknown"):
        key = ticker.strip().upper()
        if not key or key in seen or classify_asset(key) != "equity":
            return
        verdict = sentiment = ""
        if verdict_provider:
            v = verdict_provider(key) or {}
            verdict, sentiment = v.get("verdict", ""), v.get("sentiment", "")
        out.append(Candidate(ticker=key, name=name, strategy_count=strategy_count,
                             sector=sector, verdict=verdict, sentiment=sentiment))
        seen.add(key)

    for _, s in ranked:
        _add(getattr(s, "ticker", ""), getattr(s, "name", ""),
             int(getattr(s, "strategy_count", 0)), getattr(s, "sector", "Unknown"))

    for sym in (watchlist_provider or _default_watchlist)() or []:
        _add(sym)

    return out[:limit]
```

> Both default helpers are confirmed present: `build_overlap_board` (`src/scanner/overlap.py`) and `get_watchlist_tickers` (`src/scanner/watchlist.py:60`, parses `STOCK_LIST`). The tests inject fakes, so these only matter for production defaults.

- [ ] **Step 4: Run it, expect pass.**

- [ ] **Step 5: Commit**

```bash
git add src/advisor/candidates.py tests/test_advisor_candidates.py
git commit -m "feat(advisor): equity candidate merge + conviction ranking"
```

---

## Task 7: Gap engine

**Files:**
- Create: `src/advisor/gap.py`
- Test: `tests/test_advisor_gap.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_advisor_gap.py
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
```

- [ ] **Step 2: Run it, expect failure.**

- [ ] **Step 3: Implement**

```python
# src/advisor/gap.py
# -*- coding: utf-8 -*-
"""Deterministic gap: current vs target by asset class, plus cap overages.

class_delta[c] > 0 means "buy this class"; < 0 means "trim". Overages drive the
conviction+guardrails rule: only positions/sectors OVER their cap get trimmed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Gap:
    class_delta: Dict[str, float] = field(default_factory=dict)
    position_overage: Dict[str, float] = field(default_factory=dict)   # symbol -> $ over cap
    sector_overage: Dict[str, float] = field(default_factory=dict)     # sector -> $ over cap


def compute_gap(current, target) -> Gap:
    base = current.base or 0.0

    current_by_class: Dict[str, float] = defaultdict(float)
    current_by_class["cash"] += current.cash
    for p in current.positions:
        if p.asset_class == "cash":
            continue  # cash already counted via current.cash
        current_by_class[p.asset_class] += p.value

    classes = set(current_by_class) | set(target.weights)
    class_delta = {c: round(target.weights.get(c, 0.0) * base - current_by_class.get(c, 0.0), 6)
                   for c in classes}

    pos_cap = target.max_position_pct * base
    position_overage = {p.symbol: round(p.value - pos_cap, 6)
                        for p in current.positions if p.value - pos_cap > 1e-9}

    sector_totals: Dict[str, float] = defaultdict(float)
    for p in current.positions:
        if p.asset_class == "equity":
            sector_totals[p.sector] += p.value
    sec_cap = target.max_sector_pct * base
    sector_overage = {s: round(v - sec_cap, 6) for s, v in sector_totals.items() if v - sec_cap > 1e-9}

    return Gap(class_delta=class_delta, position_overage=position_overage, sector_overage=sector_overage)
```

- [ ] **Step 4: Run it, expect pass.**

- [ ] **Step 5: Commit**

```bash
git add src/advisor/gap.py tests/test_advisor_gap.py
git commit -m "feat(advisor): gap engine (class deltas + position/sector cap overages)"
```

---

## Task 8: Position sizing (conviction + guardrails)

**Files:**
- Create: `src/advisor/sizing.py`
- Test: `tests/test_advisor_sizing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_advisor_sizing.py
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
```

- [ ] **Step 2: Run it, expect failure.**

- [ ] **Step 3: Implement**

```python
# src/advisor/sizing.py
# -*- coding: utf-8 -*-
"""Turn a Gap into concrete buy/trim actions under the guardrail caps.

Rules:
- Trim only positions/sectors that are OVER cap (conviction + guardrails).
- Fill positive equity delta from ranked candidates, each new position <= per-
  position cap and its sector <= sector cap, spending no more than available.
- Non-equity positive deltas become a single dollar-target buy (symbol="").
- If equity delta can't be filled (no candidates), emit an 'unallocated' note.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List


@dataclass
class Action:
    side: str            # "buy" | "trim" | "unallocated"
    asset_class: str
    symbol: str = ""     # "" for a class-level dollar target
    amount: float = 0.0
    reason: str = ""


def size_positions(gap, candidates, current, target) -> List[Action]:
    base = current.base or 0.0
    pos_cap = target.max_position_pct * base
    sec_cap = target.max_sector_pct * base
    actions: List[Action] = []

    # 1) Trims — over-cap positions down to the cap.
    for symbol, over in sorted(gap.position_overage.items()):
        if over > 1e-9:
            actions.append(Action("trim", "equity", symbol, round(over, 2),
                                   f"{symbol} exceeds the {target.max_position_pct:.0%} per-position cap"))

    # 2) Buys per asset class from positive deltas.
    sector_running = defaultdict(float)
    for p in current.positions:
        if p.asset_class == "equity":
            sector_running[p.sector] += p.value

    for asset_class, delta in gap.class_delta.items():
        if delta <= 1e-6:
            continue
        if asset_class != "equity":
            actions.append(Action("buy", asset_class, "", round(delta, 2),
                                   f"increase {asset_class.replace('_', ' ')} by ${delta:,.0f} "
                                   f"(e.g. via a broad {asset_class.replace('_', ' ')} ETF)"))
            continue

        remaining = delta
        equity_cands = [c for c in candidates if c.strategy_count is not None]
        for c in equity_cands:
            if remaining <= 1e-6:
                break
            headroom_sector = sec_cap - sector_running[c.sector]
            room = min(pos_cap, headroom_sector, remaining)
            if room <= 1e-6:
                continue
            amount = round(room, 2)
            reason = f"fills equity target; {c.strategy_count}/4 strategies"
            if c.verdict:
                reason += f"; deep-dive: {c.verdict}"
            actions.append(Action("buy", "equity", c.ticker, amount, reason))
            sector_running[c.sector] += amount
            remaining -= amount

        if remaining > 1.0:  # meaningfully unfilled
            actions.append(Action("unallocated", "equity", "", round(remaining, 2),
                                   "no qualifying candidates under caps; consider a broad-market ETF"))

    return actions
```

- [ ] **Step 4: Run it, expect pass.**

- [ ] **Step 5: Commit**

```bash
git add src/advisor/sizing.py tests/test_advisor_sizing.py
git commit -m "feat(advisor): position sizing with conviction+guardrails caps"
```

---

## Task 9: Plan model, renderer, and generate_plan orchestration

**Files:**
- Create: `src/advisor/plan.py`
- Test: `tests/test_advisor_plan.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_advisor_plan.py
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

    import pytest
    with pytest.raises(ValueError, match="lock"):
        generate_plan(owner_id="default", repo=_Repo(),
                      portfolio_service=type("S", (), {"get_portfolio_snapshot": lambda self: {"accounts": []}})(),
                      board_provider=lambda: type("B", (), {"stocks": []})(),
                      watchlist_provider=lambda: [])


def test_generate_plan_missing_profile_raises():
    class _Repo:
        def load(self, owner_id): return None
    import pytest
    with pytest.raises(LookupError):
        generate_plan(owner_id="default", repo=_Repo())
```

- [ ] **Step 2: Run it, expect failure.**

- [ ] **Step 3: Implement**

```python
# src/advisor/plan.py
# -*- coding: utf-8 -*-
"""Assemble a deterministic plan and render it. No LLM in Phase 1.

generate_plan() is the single entry point the API calls. Every collaborator is
injectable so it is fully testable offline. A plan always includes the compliance
disclaimer and the deferred-scope notes (taxes, non-equity picks).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

from src.advisor import DEFAULT_OWNER_ID
from src.advisor.candidates import fetch_candidates
from src.advisor.gap import compute_gap
from src.advisor.sizing import Action, size_positions
from src.advisor.snapshot import get_current

DISCLAIMER = ("_Not financial advice. This is an educational, automated plan and "
              "does not account for taxes or your account type (e.g. TFSA/RRSP vs "
              "taxable). You decide and act; nothing here is executed for you._")


@dataclass
class Plan:
    mode: str
    base: float
    target: Dict[str, float]
    actions: List[Action] = field(default_factory=list)
    rationale: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def _plans_dir() -> str:
    return os.path.join("output", "plans")


def _default_repo():
    from src.advisor.repository import InvestorProfileRepository
    return InvestorProfileRepository()


def generate_plan(owner_id: str = DEFAULT_OWNER_ID,
                  repo=None,
                  portfolio_service=None,
                  board_provider: Optional[Callable] = None,
                  watchlist_provider: Optional[Callable] = None,
                  verdict_provider: Optional[Callable] = None,
                  candidate_limit: int = 8,
                  save: bool = True) -> Plan:
    repo = repo or _default_repo()
    loaded = repo.load(owner_id)
    if loaded is None:
        raise LookupError(f"no investor profile for owner_id={owner_id!r}; create one first")
    profile, target = loaded
    if not target.locked:
        raise ValueError("target allocation must be locked before generating a plan")

    current = get_current(profile, portfolio_service=portfolio_service)
    gap = compute_gap(current, target)
    candidates = fetch_candidates(limit=candidate_limit, board_provider=board_provider,
                                  watchlist_provider=watchlist_provider, verdict_provider=verdict_provider)
    actions = size_positions(gap, candidates, current, target)

    rationale = _rationale(current, target, actions)
    plan = Plan(mode=current.mode, base=round(current.base, 2), target=dict(target.weights),
                actions=actions, rationale=rationale)

    if save:
        _save_markdown(plan)
    return plan


def _rationale(current, target, actions) -> str:
    verb = "Deploying your cash into" if current.mode == "deploy" else "Rebalancing toward"
    n_buy = sum(1 for a in actions if a.side == "buy")
    n_trim = sum(1 for a in actions if a.side == "trim")
    return (f"{verb} your locked target on a ${current.base:,.0f} base: "
            f"{n_buy} buy action(s), {n_trim} trim(s). Concentrated positions under "
            f"their caps are left in place.")


def _fmt(amount: float) -> str:
    return f"${amount:,.0f}"


def render_markdown(plan: Plan) -> str:
    lines = [
        "## 📋 Your Investment Plan",
        f"_{plan.mode.title()} mode · base {_fmt(plan.base)} · {plan.generated_at}_",
        "",
        "**Target allocation:** " + ", ".join(
            f"{k.replace('_', ' ').title()} {v:.0%}" for k, v in plan.target.items()),
        "",
        plan.rationale,
        "",
        "### Actions",
    ]
    if not plan.actions:
        lines.append("- Already aligned with your target — no action needed.")
    for a in plan.actions:
        tag = {"buy": "🟢 Buy", "trim": "🔻 Trim", "unallocated": "⚪ Unallocated"}.get(a.side, a.side)
        target = a.symbol or a.asset_class.replace("_", " ").title()
        lines.append(f"- **{tag} {target}** — {_fmt(a.amount)} · {a.reason}")
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def _save_markdown(plan: Plan) -> str:
    directory = _plans_dir()
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"plan_{datetime.now().strftime('%Y%m%d_%H%M')}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(plan))
    return path
```

- [ ] **Step 4: Run it, expect pass.**

- [ ] **Step 5: Commit**

```bash
git add src/advisor/plan.py tests/test_advisor_plan.py
git commit -m "feat(advisor): plan assembly + markdown renderer + generate_plan"
```

---

## Task 10: API — schemas, endpoints, router

**Files:**
- Create: `api/v1/schemas/advisor.py`, `api/v1/endpoints/advisor.py`
- Modify: `api/v1/router.py`
- Test: `tests/test_advisor_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_advisor_api.py
import tempfile

from fastapi.testclient import TestClient


def _client():
    from src.storage import DatabaseManager
    DatabaseManager._instance = None
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    DatabaseManager(db_url=f"sqlite:///{tmp.name}")  # prime singleton on temp DB
    from api.app import create_app  # existing factory; see note below
    return TestClient(create_app())


def test_get_profile_404_before_creation():
    client = _client()
    r = client.get("/api/v1/advisor/profile")
    assert r.status_code == 404


def test_put_then_get_profile_roundtrip():
    client = _client()
    body = {"risk_tolerance": "aggressive", "horizon_years": 20, "goals": ["growth"],
            "investable_cash": 25000.0, "base_currency": "USD",
            "target": {"weights": {"equity": 0.8, "cash": 0.2}, "max_position_pct": 0.2,
                       "max_sector_pct": 0.4, "locked": True}}
    r = client.put("/api/v1/advisor/profile", json=body)
    assert r.status_code == 200
    got = client.get("/api/v1/advisor/profile").json()
    assert got["risk_tolerance"] == "aggressive"
    assert got["target"]["locked"] is True


def test_generate_plan_requires_locked_target():
    client = _client()
    body = {"risk_tolerance": "moderate", "horizon_years": 10, "investable_cash": 1000.0,
            "target": {"weights": {"equity": 1.0}, "locked": False}}
    client.put("/api/v1/advisor/profile", json=body)
    r = client.post("/api/v1/advisor/plan")
    assert r.status_code == 400  # target not locked
```

> Confirmed: `api/app.py` exports both `create_app()` (`api/app.py:48`) and a module-level `app` (line 208). The test uses `create_app()` so each run gets a fresh app bound to the temp DB.

- [ ] **Step 2: Run it, expect failure.**

- [ ] **Step 3a: Schemas**

```python
# api/v1/schemas/advisor.py
# -*- coding: utf-8 -*-
"""Advisor API schemas."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class TargetModel(BaseModel):
    weights: Dict[str, float] = Field(default_factory=dict)
    max_position_pct: float = 0.15
    max_sector_pct: float = 0.30
    locked: bool = False


class ProfileRequest(BaseModel):
    risk_tolerance: str = "moderate"
    horizon_years: int = 10
    goals: List[str] = Field(default_factory=list)
    constraints: Dict[str, bool] = Field(default_factory=dict)
    investable_cash: float = 0.0
    base_currency: str = "USD"
    target: TargetModel = Field(default_factory=TargetModel)


class ProfileResponse(ProfileRequest):
    owner_id: str


class SuggestResponse(BaseModel):
    target: TargetModel


class ActionModel(BaseModel):
    side: str
    asset_class: str
    symbol: str = ""
    amount: float = 0.0
    reason: str = ""


class PlanResponse(BaseModel):
    mode: str
    base: float
    target: Dict[str, float]
    actions: List[ActionModel]
    rationale: str
    markdown: str
    generated_at: str
```

- [ ] **Step 3b: Endpoints**

```python
# api/v1/endpoints/advisor.py
# -*- coding: utf-8 -*-
"""Advisor endpoints: investor profile CRUD, allocation suggestion, plan generation.

Single-user v1: owner_id fixed to DEFAULT_OWNER_ID (mirrors portfolio's optional owner).
"""

import logging

from fastapi import APIRouter, HTTPException

from api.v1.schemas.advisor import (ActionModel, PlanResponse, ProfileRequest,
                                     ProfileResponse, SuggestResponse, TargetModel)
from src.advisor import DEFAULT_OWNER_ID
from src.advisor.allocation import suggest_allocation
from src.advisor.plan import generate_plan, render_markdown
from src.advisor.profile import InvestorProfile, Target
from src.advisor.repository import InvestorProfileRepository

logger = logging.getLogger(__name__)
router = APIRouter()


def _repo() -> InvestorProfileRepository:
    return InvestorProfileRepository()


@router.get("/profile", response_model=ProfileResponse, summary="Get the investor profile")
def get_profile() -> ProfileResponse:
    loaded = _repo().load(DEFAULT_OWNER_ID)
    if loaded is None:
        raise HTTPException(status_code=404, detail="No investor profile yet — create one.")
    profile, target = loaded
    return ProfileResponse(owner_id=profile.owner_id, risk_tolerance=profile.risk_tolerance,
                           horizon_years=profile.horizon_years, goals=profile.goals,
                           constraints=profile.constraints, investable_cash=profile.investable_cash,
                           base_currency=profile.base_currency, target=TargetModel(**vars(target)))


@router.put("/profile", response_model=ProfileResponse, summary="Create or update the profile")
def put_profile(body: ProfileRequest) -> ProfileResponse:
    profile = InvestorProfile(owner_id=DEFAULT_OWNER_ID, risk_tolerance=body.risk_tolerance,
                              horizon_years=body.horizon_years, goals=body.goals,
                              constraints=body.constraints, investable_cash=body.investable_cash,
                              base_currency=body.base_currency)
    target = Target(**body.target.model_dump())
    try:
        profile.validate()
        target.validate()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _repo().save(profile, target)
    return get_profile()


@router.post("/suggest", response_model=SuggestResponse, summary="Suggest a target from the profile")
def suggest() -> SuggestResponse:
    loaded = _repo().load(DEFAULT_OWNER_ID)
    if loaded is None:
        raise HTTPException(status_code=404, detail="No investor profile yet — create one.")
    profile, _ = loaded
    return SuggestResponse(target=TargetModel(**vars(suggest_allocation(profile))))


@router.post("/plan", response_model=PlanResponse, summary="Generate the whole-portfolio plan")
def create_plan() -> PlanResponse:
    try:
        plan = generate_plan(owner_id=DEFAULT_OWNER_ID)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:  # target not locked
        raise HTTPException(status_code=400, detail=str(exc))
    return PlanResponse(mode=plan.mode, base=plan.base, target=plan.target,
                        actions=[ActionModel(**vars(a)) for a in plan.actions],
                        rationale=plan.rationale, markdown=render_markdown(plan),
                        generated_at=plan.generated_at)
```

- [ ] **Step 3c: Register the router in `api/v1/router.py`**

Add `advisor` to the endpoints import line, then after the `watchlist` block:

```python
router.include_router(
    advisor.router,
    prefix="/advisor",
    tags=["advisor"],
)
```

- [ ] **Step 4: Run it, expect pass**

Run: `python3 -m pytest tests/test_advisor_api.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add api/v1/schemas/advisor.py api/v1/endpoints/advisor.py api/v1/router.py tests/test_advisor_api.py
git commit -m "feat(advisor): profile/suggest/plan API endpoints"
```

---

## Task 11: Web — API client, Plan page, route, nav, test

**Files:**
- Create: `apps/dsa-web/src/api/advisor.ts`, `apps/dsa-web/src/pages/PlanPage.tsx`, `apps/dsa-web/src/pages/__tests__/PlanPage.test.tsx`
- Modify: `apps/dsa-web/src/App.tsx`, the nav component

- [ ] **Step 1: API client**

```typescript
// apps/dsa-web/src/api/advisor.ts
import apiClient from './index';

export interface AdvisorTarget {
  weights: Record<string, number>;
  max_position_pct: number;
  max_sector_pct: number;
  locked: boolean;
}
export interface AdvisorProfile {
  owner_id?: string;
  risk_tolerance: string;
  horizon_years: number;
  goals: string[];
  constraints: Record<string, boolean>;
  investable_cash: number;
  base_currency: string;
  target: AdvisorTarget;
}
export interface PlanAction {
  side: string; asset_class: string; symbol: string; amount: number; reason: string;
}
export interface AdvisorPlan {
  mode: string; base: number; target: Record<string, number>;
  actions: PlanAction[]; rationale: string; markdown: string; generated_at: string;
}

export const advisorApi = {
  async getProfile(): Promise<AdvisorProfile | null> {
    try {
      const res = await apiClient.get<AdvisorProfile>('/api/v1/advisor/profile');
      return res.data;
    } catch (e: unknown) {
      return null; // 404 => no profile yet
    }
  },
  async saveProfile(profile: AdvisorProfile): Promise<AdvisorProfile> {
    const res = await apiClient.put<AdvisorProfile>('/api/v1/advisor/profile', profile);
    return res.data;
  },
  async suggest(): Promise<AdvisorTarget> {
    const res = await apiClient.post<{ target: AdvisorTarget }>('/api/v1/advisor/suggest');
    return res.data.target;
  },
  async generatePlan(): Promise<AdvisorPlan> {
    const res = await apiClient.post<AdvisorPlan>('/api/v1/advisor/plan');
    return res.data;
  },
};
```

- [ ] **Step 2: Write the failing component test**

```tsx
// apps/dsa-web/src/pages/__tests__/PlanPage.test.tsx
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import PlanPage from '../PlanPage';
import { advisorApi } from '../../api/advisor';

vi.mock('../../api/advisor');

const baseProfile = {
  risk_tolerance: 'moderate', horizon_years: 10, goals: [], constraints: {},
  investable_cash: 1000, base_currency: 'USD',
  target: { weights: { equity: 1 }, max_position_pct: 0.15, max_sector_pct: 0.3, locked: true },
};

describe('PlanPage', () => {
  beforeEach(() => vi.resetAllMocks());

  it('renders the profile form and generates a plan', async () => {
    (advisorApi.getProfile as any).mockResolvedValue(baseProfile);
    (advisorApi.generatePlan as any).mockResolvedValue({
      mode: 'deploy', base: 1000, target: { equity: 1 }, rationale: 'Deploying your cash.',
      generated_at: '2026-09-18T10:00:00', markdown: '## Plan', actions: [
        { side: 'buy', asset_class: 'equity', symbol: 'INGR', amount: 150, reason: 'fills equity target' },
      ],
    });
    render(<PlanPage />);
    await waitFor(() => expect(screen.getByText(/Investment Plan/i)).toBeInTheDocument());
    await userEvent.click(screen.getByRole('button', { name: /generate plan/i }));
    await waitFor(() => expect(screen.getByText(/INGR/)).toBeInTheDocument());
  });

  it('prompts to create a profile when none exists', async () => {
    (advisorApi.getProfile as any).mockResolvedValue(null);
    render(<PlanPage />);
    await waitFor(() => expect(screen.getByText(/create your profile/i)).toBeInTheDocument());
  });
});
```

Run: `cd apps/dsa-web && npx vitest run src/pages/__tests__/PlanPage.test.tsx` → FAIL (no `PlanPage`).

- [ ] **Step 3: Implement the page**

```tsx
// apps/dsa-web/src/pages/PlanPage.tsx
import { useEffect, useState } from 'react';
import { advisorApi, AdvisorProfile, AdvisorPlan } from '../api/advisor';

const EMPTY: AdvisorProfile = {
  risk_tolerance: 'moderate', horizon_years: 10, goals: [], constraints: {},
  investable_cash: 0, base_currency: 'USD',
  target: { weights: { equity: 0.7, fixed_income: 0.2, cash: 0.1 }, max_position_pct: 0.15, max_sector_pct: 0.3, locked: false },
};

export default function PlanPage() {
  const [profile, setProfile] = useState<AdvisorProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [plan, setPlan] = useState<AdvisorPlan | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    advisorApi.getProfile().then((p) => { setProfile(p); setLoading(false); });
  }, []);

  const startProfile = () => setProfile({ ...EMPTY });

  const save = async () => {
    if (!profile) return;
    setBusy(true); setError(null);
    try { setProfile(await advisorApi.saveProfile(profile)); }
    catch (e) { setError('Could not save your profile. Check the values and try again.'); }
    finally { setBusy(false); }
  };

  const suggest = async () => {
    if (!profile) return;
    const target = await advisorApi.suggest();
    setProfile({ ...profile, target });
  };

  const generate = async () => {
    setBusy(true); setError(null);
    try { setPlan(await advisorApi.generatePlan()); }
    catch (e) { setError('Lock your target allocation first, then generate.'); }
    finally { setBusy(false); }
  };

  if (loading) return <div className="page-pad">Loading…</div>;

  if (!profile) {
    return (
      <div className="page-pad">
        <h1>Investment Plan</h1>
        <p>Create your profile to generate a whole-portfolio plan.</p>
        <button onClick={startProfile}>Create your profile</button>
      </div>
    );
  }

  const locked = profile.target.locked;
  return (
    <div className="page-pad">
      <h1>Your Investment Plan</h1>

      <section>
        <h2>My Profile</h2>
        <label>Risk tolerance
          <select value={profile.risk_tolerance}
                  onChange={(e) => setProfile({ ...profile, risk_tolerance: e.target.value })}>
            <option value="conservative">Conservative</option>
            <option value="moderate">Moderate</option>
            <option value="aggressive">Aggressive</option>
          </select>
        </label>
        <label>Horizon (years)
          <input type="number" value={profile.horizon_years}
                 onChange={(e) => setProfile({ ...profile, horizon_years: Number(e.target.value) })} />
        </label>
        <label>Investable cash
          <input type="number" value={profile.investable_cash}
                 onChange={(e) => setProfile({ ...profile, investable_cash: Number(e.target.value) })} />
        </label>
        <button onClick={suggest}>Suggest target</button>
        <button onClick={save} disabled={busy}>Save profile</button>
      </section>

      <section>
        <h2>Target allocation {locked ? '🔒' : '(unlocked)'}</h2>
        <ul>
          {Object.entries(profile.target.weights).map(([k, v]) => (
            <li key={k}>{k.replace('_', ' ')}: {(v * 100).toFixed(0)}%</li>
          ))}
        </ul>
        <label>
          <input type="checkbox" checked={locked}
                 onChange={(e) => setProfile({ ...profile, target: { ...profile.target, locked: e.target.checked } })} />
          Lock target
        </label>
        <button onClick={save} disabled={busy}>Save</button>
      </section>

      <button onClick={generate} disabled={busy || !locked}>Generate plan</button>
      {error && <p role="alert">{error}</p>}

      {plan && (
        <section>
          <h2>Plan · {plan.mode} mode · base ${plan.base.toLocaleString()}</h2>
          <p>{plan.rationale}</p>
          <ul>
            {plan.actions.map((a, i) => (
              <li key={i}><strong>{a.side} {a.symbol || a.asset_class}</strong> — ${a.amount.toLocaleString()} · {a.reason}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
```

> `className="page-pad"` mirrors whatever wrapper the other pages use — open `PortfolioPage.tsx` and copy its top-level layout wrapper class so spacing matches. Styling can follow existing page conventions; the test only asserts text/behavior.

- [ ] **Step 4: Add the route + nav link**

In `apps/dsa-web/src/App.tsx`: add `import PlanPage from './pages/PlanPage';` and, inside the `<Shell>` routes, `<Route path="/plan" element={<PlanPage />} />`.

Find the nav (`grep -rn 'to="/portfolio"' apps/dsa-web/src`) and add a sibling link, e.g. `<NavLink to="/plan">Plan</NavLink>` matching the existing link markup.

- [ ] **Step 5: Run tests, expect pass**

Run: `cd apps/dsa-web && npx vitest run src/pages/__tests__/PlanPage.test.tsx` → PASS.
Then a full front-end pass: `npx vitest run`.

- [ ] **Step 6: Commit**

```bash
git add apps/dsa-web/src/api/advisor.ts apps/dsa-web/src/pages/PlanPage.tsx \
        apps/dsa-web/src/pages/__tests__/PlanPage.test.tsx apps/dsa-web/src/App.tsx
# plus the nav component you edited
git commit -m "feat(advisor): Plan page, API client, route + nav"
```

---

## Final verification

- [ ] `python3 -m pytest tests/test_advisor_*.py -q` — all green.
- [ ] `cd apps/dsa-web && npx vitest run` — all green.
- [ ] Manual smoke: start the server (`python server.py`), open `/plan`, create a profile, click **Suggest target**, tweak, **Lock**, **Save**, **Generate plan** — a deploy-from-cash plan renders (portfolio is empty today). Confirm a `output/plans/plan_*.md` file was written.
- [ ] Announce completion and use **superpowers:finishing-a-development-branch**.

## Notes / risks

- **Currency:** rebalance mode assumes snapshot `market_value_base` is in the profile's base currency. The live portfolio is empty, so deploy mode (pure profile currency) is what runs now; cross-currency FX normalization is deferred alongside taxes and is noted in the plan output.
- **Accessors verified:** `build_overlap_board`, `get_watchlist_tickers`, `DatabaseManager.session_scope` (commits), and `api.app.create_app()` all confirmed present — no open lookups.
- **No LLM anywhere in Phase 1.** Phase 2 (AdvisorAgent) wraps these functions as tools and falls back to `generate_plan`.
