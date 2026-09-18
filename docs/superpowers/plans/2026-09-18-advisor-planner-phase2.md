# Advisor / Whole-Portfolio Planner — Phase 2 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) tracking.

**Goal:** Add the LLM layer as a *lean single structured call*: the model picks & orders which equity candidates best fit the investor's profile (using the deep-dive verdicts) and writes the plan's rationale. All numbers stay in the deterministic Phase 1 engine, and any LLM failure/quota degrades cleanly to Phase 1's deterministic ordering + templated rationale.

**Architecture:** A new `src/advisor/select.py` performs one `litellm.completion` call returning JSON `{order, rationale}`; `generate_plan(smart=True)` reorders the candidate list per the LLM (filtered to real tickers) *before* the existing `size_positions`, so amounts remain exact. A `smart` flag threads through the API and a "Refine with AI" toggle on the Plan page. `engine` ("ai" | "deterministic") reports which path produced the plan.

**Tech Stack:** Python 3.12, pytest, litellm (Gemini), FastAPI, React/Vite/TS, vitest. Branch: `feat/advisor-planner-phase2` (off `feat/advisor-planner-phase1`).

**Design spec:** `docs/superpowers/specs/2026-07-18-advisor-planner-design.md` (Phase 2). Decision (2026-09-18): **lean single-call**, not a full tool-calling agent — quota-safe and a clean fit.

**Run tests:** `python3 -m pytest tests/test_advisor_*.py -q` · `cd apps/dsa-web && npx vitest run`

---

## Safety invariant (the whole point)

The LLM only chooses **which** candidates and their **order**, plus the narrative text. It can never change a dollar amount, invent a ticker (output is filtered to the real candidate set), or block a plan (any failure → deterministic fallback). `size_positions` (Phase 1, tested) always computes the money.

## File Structure

**Create:**
- `src/advisor/select.py` — `select_candidates_with_llm(...)`, prompt builder, JSON parse/validate, `_default_complete` (litellm).
- `tests/test_advisor_select.py`, `tests/test_advisor_smart_plan.py`.
- `apps/dsa-web/src/pages/__tests__/PlanPage.smart.test.tsx` (or extend existing PlanPage test).

**Modify:**
- `src/advisor/plan.py` — `Plan.engine` field; `generate_plan(smart=False, select_fn=None)` smart path; `render_markdown` shows the engine.
- `api/v1/endpoints/advisor.py` — `POST /advisor/plan` accepts `smart` (default True); `PlanResponse.engine`.
- `api/v1/schemas/advisor.py` — add `engine` to `PlanResponse`.
- `apps/dsa-web/src/api/advisor.ts` — `generatePlan(smart)` + `engine` on `AdvisorPlan`.
- `apps/dsa-web/src/pages/PlanPage.tsx` — "Refine with AI" toggle + engine badge.

---

## Task 1: LLM candidate selector (`select.py`)

**Files:** Create `src/advisor/select.py`, `tests/test_advisor_select.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_advisor_select.py
import json
from src.advisor.candidates import Candidate
from src.advisor.select import select_candidates_with_llm


def _cands():
    return [Candidate("INGR", strategy_count=4, sector="Consumer", verdict="Buy"),
            Candidate("TRV", strategy_count=4, sector="Financials"),
            Candidate("OKLO", strategy_count=1, sector="Energy", verdict="Hold")]


def test_llm_reorders_and_filters_to_real_tickers():
    # LLM prefers TRV then INGR, and hallucinates FAKE (must be dropped).
    fake = lambda system, user: json.dumps({"order": ["TRV", "FAKE", "INGR"], "rationale": "Quality first."})
    ordered, rationale = select_candidates_with_llm(_cands(), profile=None, equity_budget=1000.0,
                                                    max_position_pct=0.15, complete_fn=fake)
    assert [c.ticker for c in ordered] == ["TRV", "INGR"]  # FAKE dropped, OKLO omitted by LLM
    assert rationale == "Quality first."


def test_empty_or_invalid_selection_returns_none():
    # No valid tickers -> signal failure so the caller falls back to deterministic.
    fake = lambda system, user: json.dumps({"order": ["FAKE"], "rationale": "x"})
    assert select_candidates_with_llm(_cands(), None, 1000.0, 0.15, complete_fn=fake) is None


def test_bad_json_returns_none():
    fake = lambda system, user: "not json at all"
    assert select_candidates_with_llm(_cands(), None, 1000.0, 0.15, complete_fn=fake) is None


def test_llm_exception_returns_none():
    def boom(system, user):
        raise RuntimeError("quota")
    assert select_candidates_with_llm(_cands(), None, 1000.0, 0.15, complete_fn=boom) is None
```

- [ ] **Step 2: Run, expect failure** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

```python
# src/advisor/select.py
# -*- coding: utf-8 -*-
"""LLM candidate selection — one structured call. Numbers are NOT decided here.

Given the equity candidates (with deep-dive verdicts) + the investor profile, the
LLM returns a JSON ordering (a subset/reordering of the input tickers) plus a
rationale. Output is filtered to the real candidate set; any failure, bad JSON, or
empty result returns None so the caller falls back to the deterministic order.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

CompleteFn = Callable[[str, str], str]  # (system, user) -> raw text

_SYSTEM = (
    "You are a portfolio planning assistant. From a list of candidate stocks, choose "
    "and rank the ones that best fit the investor's profile and risk tolerance. You do "
    "NOT decide dollar amounts. Return STRICT JSON: "
    '{"order": ["TICKER", ...], "rationale": "one short paragraph"}. '
    "Only use tickers from the provided list. Prefer higher-conviction, on-profile names; "
    "you may drop names that don't fit."
)


def _build_user(candidates, profile, equity_budget: float, max_position_pct: float) -> str:
    lines = [f"Investable equity budget: ${equity_budget:,.0f}",
             f"Max per position: {max_position_pct:.0%}"]
    if profile is not None:
        lines.append(f"Investor: risk={profile.risk_tolerance}, horizon={profile.horizon_years}y, "
                     f"goals={', '.join(profile.goals) or 'n/a'}")
    lines.append("Candidates:")
    for c in candidates:
        bits = [f"{c.ticker}", f"{c.strategy_count}/4", f"sector={c.sector}"]
        if c.verdict:
            bits.append(f"deep-dive={c.verdict}")
        lines.append("  - " + ", ".join(bits))
    return "\n".join(lines)


def _extract_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.DOTALL)  # tolerate code fences / prose
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _default_complete(system: str, user: str) -> str:
    import litellm
    from src.config import (get_config, get_effective_agent_primary_model,
                            get_api_keys_for_model, extra_litellm_params)
    config = get_config()
    model = get_effective_agent_primary_model(config)
    kwargs = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.2,
        "timeout": 30,
    }
    keys = get_api_keys_for_model(model, config)
    if keys:
        kwargs["api_key"] = keys[0]
    kwargs.update(extra_litellm_params(model, config))
    resp = litellm.completion(**kwargs)
    return resp.choices[0].message.content or ""


def select_candidates_with_llm(candidates, profile, equity_budget: float,
                               max_position_pct: float,
                               complete_fn: Optional[CompleteFn] = None
                               ) -> Optional[Tuple[List, str]]:
    if not candidates:
        return None
    complete = complete_fn or _default_complete
    try:
        raw = complete(_SYSTEM, _build_user(candidates, profile, equity_budget, max_position_pct))
    except Exception as exc:
        logger.warning("[advisor.select] LLM call failed: %s", exc)
        return None

    data = _extract_json(raw)
    if not isinstance(data, dict):
        return None
    order = data.get("order") or []
    rationale = str(data.get("rationale") or "").strip()
    by_ticker = {c.ticker: c for c in candidates}
    ordered = [by_ticker[t.strip().upper()] for t in order
               if isinstance(t, str) and t.strip().upper() in by_ticker]
    if not ordered:
        return None
    return ordered, (rationale or "Selected to fit your profile.")
```

- [ ] **Step 4: Run, expect pass.**

- [ ] **Step 5: Commit**

```bash
git add src/advisor/select.py tests/test_advisor_select.py
git commit -m "feat(advisor): LLM candidate selector (one structured call, filtered + fail-safe)"
```

---

## Task 2: Wire the smart path into `generate_plan`

**Files:** Modify `src/advisor/plan.py`; Create `tests/test_advisor_smart_plan.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_advisor_smart_plan.py
from src.advisor.candidates import Candidate
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
```

- [ ] **Step 2: Run, expect failure** (`Plan` has no `engine`; `generate_plan` has no `smart`).

- [ ] **Step 3: Implement** — edits to `src/advisor/plan.py`:

Add `engine: str = "deterministic"` to the `Plan` dataclass (after `rationale`).

Change the `generate_plan` signature and candidate section:

```python
def generate_plan(owner_id: str = DEFAULT_OWNER_ID,
                  repo=None,
                  portfolio_service=None,
                  board_provider: Optional[Callable] = None,
                  watchlist_provider: Optional[Callable] = None,
                  verdict_provider: Optional[Callable] = None,
                  candidate_limit: int = 8,
                  smart: bool = False,
                  select_fn: Optional[Callable] = None,
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

    engine = "deterministic"
    llm_rationale = None
    if smart and candidates:
        equity_budget = max(gap.class_delta.get("equity", 0.0), 0.0)
        selector = select_fn or _default_select
        try:
            picked = selector(candidates, profile, equity_budget, target.max_position_pct)
        except Exception:
            picked = None
        if picked:
            ordered, llm_rationale = picked
            candidates = ordered
            engine = "ai"

    actions = size_positions(gap, candidates, current, target)
    rationale = llm_rationale or _rationale(current, target, actions)
    plan = Plan(mode=current.mode, base=round(current.base, 2), target=dict(target.weights),
                actions=actions, rationale=rationale, engine=engine)

    if save:
        _save_markdown(plan)
    return plan
```

Add near `_default_repo`:

```python
def _default_select(candidates, profile, equity_budget, max_position_pct):
    from src.advisor.select import select_candidates_with_llm
    return select_candidates_with_llm(candidates, profile, equity_budget, max_position_pct)
```

Update `render_markdown` header line to show the engine:

```python
        f"_{plan.mode.title()} mode · base {_fmt(plan.base)} · "
        f"{'AI-refined' if plan.engine == 'ai' else 'deterministic'} · {plan.generated_at}_",
```

- [ ] **Step 4: Run, expect pass.** (`pytest tests/test_advisor_smart_plan.py tests/test_advisor_plan.py -q`)

- [ ] **Step 5: Commit**

```bash
git add src/advisor/plan.py tests/test_advisor_smart_plan.py
git commit -m "feat(advisor): smart generate_plan path (LLM order + rationale, deterministic fallback)"
```

---

## Task 3: API — `smart` flag + `engine` in response

**Files:** Modify `api/v1/schemas/advisor.py`, `api/v1/endpoints/advisor.py`; extend `tests/test_advisor_api.py`

- [ ] **Step 1: Add the failing test** (append to `tests/test_advisor_api.py`)

```python
def test_plan_smart_uses_selector_and_reports_engine(monkeypatch):
    client = _client()
    client.put("/api/v1/advisor/profile", json={
        "risk_tolerance": "moderate", "horizon_years": 10, "investable_cash": 5000.0,
        "target": {"weights": {"equity": 1.0}, "max_position_pct": 0.5, "locked": True}})
    # Patch the low-level LLM call so no network/quota is used.
    import src.advisor.select as sel
    monkeypatch.setattr(sel, "_default_complete",
                        lambda system, user: '{"order": [], "rationale": "x"}')  # empty -> None -> fallback
    r = client.post("/api/v1/advisor/plan?smart=true")
    assert r.status_code == 200
    assert r.json()["engine"] in ("ai", "deterministic")


def test_plan_deterministic_engine_by_default():
    client = _client()
    client.put("/api/v1/advisor/profile", json={
        "risk_tolerance": "moderate", "horizon_years": 10, "investable_cash": 5000.0,
        "target": {"weights": {"equity": 1.0}, "max_position_pct": 0.5, "locked": True}})
    r = client.post("/api/v1/advisor/plan?smart=false")
    assert r.status_code == 200
    assert r.json()["engine"] == "deterministic"
```

- [ ] **Step 2: Run, expect failure** (`engine` missing / `smart` param ignored).

- [ ] **Step 3: Implement**

In `api/v1/schemas/advisor.py`, add to `PlanResponse`:

```python
    engine: str = "deterministic"
```

In `api/v1/endpoints/advisor.py`, update the plan endpoint:

```python
from fastapi import APIRouter, HTTPException, Query
...
@router.post("/plan", response_model=PlanResponse, summary="Generate the whole-portfolio plan")
def create_plan(smart: bool = Query(True, description="Use the LLM to refine candidate selection")) -> PlanResponse:
    try:
        plan = generate_plan(owner_id=DEFAULT_OWNER_ID, smart=smart)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return PlanResponse(mode=plan.mode, base=plan.base, target=plan.target,
                        actions=[ActionModel(**vars(a)) for a in plan.actions],
                        rationale=plan.rationale, markdown=render_markdown(plan),
                        generated_at=plan.generated_at, engine=plan.engine)
```

- [ ] **Step 4: Run, expect pass.** (`pytest tests/test_advisor_api.py -q`)

- [ ] **Step 5: Commit**

```bash
git add api/v1/schemas/advisor.py api/v1/endpoints/advisor.py tests/test_advisor_api.py
git commit -m "feat(advisor): API smart flag + engine field on plan response"
```

---

## Task 4: Web — "Refine with AI" toggle + engine badge

**Files:** Modify `apps/dsa-web/src/api/advisor.ts`, `apps/dsa-web/src/pages/PlanPage.tsx`; extend the PlanPage test

- [ ] **Step 1: API client** — in `advisor.ts`, add `engine` to `AdvisorPlan` and a `smart` arg:

```typescript
export interface AdvisorPlan {
  mode: string; base: number; target: Record<string, number>;
  actions: PlanAction[]; rationale: string; markdown: string; generated_at: string;
  engine: string;
}
...
  async generatePlan(smart = true): Promise<AdvisorPlan> {
    const res = await apiClient.post<AdvisorPlan>(`/api/v1/advisor/plan?smart=${smart}`);
    return res.data;
  },
```

- [ ] **Step 2: Extend the failing test** — add to `PlanPage.test.tsx`:

```tsx
  it('passes the AI toggle to generatePlan and shows the engine', async () => {
    (advisorApi.getProfile as any).mockResolvedValue(baseProfile);
    (advisorApi.generatePlan as any).mockResolvedValue({
      mode: 'deploy', base: 1000, target: { equity: 1 }, rationale: 'r', engine: 'ai',
      generated_at: 't', markdown: '#', actions: [],
    });
    render(<PlanPage />);
    await waitFor(() => expect(screen.getByText(/Investment Plan/i)).toBeInTheDocument());
    fireEvent.click(screen.getByRole('button', { name: /generate plan/i }));
    await waitFor(() => expect(advisorApi.generatePlan).toHaveBeenCalledWith(true));
    await waitFor(() => expect(screen.getByText(/AI-refined/i)).toBeInTheDocument());
  });
```

- [ ] **Step 3: Implement** — in `PlanPage.tsx`:
  - Add `const [useAI, setUseAI] = useState(true);`
  - A checkbox near the Generate button: `<label><input type="checkbox" checked={useAI} onChange={e=>setUseAI(e.target.checked)} /> Refine with AI</label>`
  - `generate` calls `advisorApi.generatePlan(useAI)`.
  - In the plan section header, show the engine: `{plan.engine === 'ai' ? 'AI-refined' : 'Deterministic'}`.
  - Update the default mocks in the two existing PlanPage tests to include `engine: 'deploy'`… (add `engine: 'ai'` / `'deterministic'`) so they still type-check.

- [ ] **Step 4: Run** `cd apps/dsa-web && npx vitest run src/pages/__tests__/PlanPage.test.tsx` then the full `npx vitest run`, and `npx tsc --noEmit`.

- [ ] **Step 5: Commit**

```bash
git add apps/dsa-web/src/api/advisor.ts apps/dsa-web/src/pages/PlanPage.tsx apps/dsa-web/src/pages/__tests__/PlanPage.test.tsx
git commit -m "feat(advisor): Plan page 'Refine with AI' toggle + engine badge"
```

---

## Final verification

- [ ] `python3 -m pytest tests/test_advisor_*.py -q` — all green.
- [ ] `cd apps/dsa-web && npx vitest run` — all green; `npx tsc --noEmit` clean.
- [ ] Offline smoke: `generate_plan(smart=True, select_fn=<fake>)` → `engine="ai"`, LLM order respected; `select_fn` returning None → `engine="deterministic"`, numbers identical. (No live Gemini needed; quota is exhausted today and the fallback path is what runs.)
- [ ] Announce completion → finishing-a-development-branch.

## Notes / risks

- **Numbers are never LLM-derived** — `size_positions` (Phase 1) always computes amounts; the LLM only reorders/filters candidates and writes prose. Hallucinated tickers are filtered out; empty/invalid/failed selection → deterministic fallback.
- **Quota:** exactly one Gemini call per smart plan, and it degrades silently. Today's exhausted quota means `engine` will report `deterministic` until quota resets — correct, not a bug.
- **Model resolution** mirrors `llm_adapter`'s direct-litellm branch (`get_effective_agent_primary_model` + `get_api_keys_for_model` + `extra_litellm_params`), so it uses the same configured Gemini model as the rest of the app.
