# Advisor / Whole-Portfolio Planner — Design (v1)

**Date:** 2026-07-18
**Status:** Approved design → next step is the Phase 1 implementation plan
**Author:** Jose + Claude (brainstorming session)

---

## 1. Goal

Turn the stock tool into the first slice of a "wealth team": given an investor
profile and (optionally) a real portfolio, produce a **whole-portfolio plan** — a
target allocation plus the specific **buys/trims** to reach it. The plan is
**decision-support** the user reviews and acts on; it never executes trades.

This is the **advisor** milestone: it wires together four things that already
exist (scan candidates, deep-dive verdicts, portfolio holdings, portfolio
analytics) behind a new *personalization* layer that today's system lacks.

## 2. What already exists (reused, not rebuilt)

| Capability | Where | Role in the plan |
|---|---|---|
| Scan → candidates | `src/scanner/overlap.py` (`build_overlap_board`) | *What's worth considering* |
| Deep-dive → AI verdicts | `src/scanner/deepdive.py` (`run_deep_dives`, `DeepDiveResult`) | *Is this equity actually good* |
| Holdings, cash, avg cost, FX | `src/services/portfolio_service.py` (`get_portfolio_snapshot`) | *What you own now* |
| Portfolio analytics | `src/agent/agents/portfolio_agent.py` | Optional sizing/risk overlay |
| Agent contract | `src/agent/agents/base_agent.py`, `src/agent/protocols.py` | Pattern the AdvisorAgent follows |
| Watchlist | `STOCK_LIST` / `src/scanner/watchlist.py` | Extra candidate source |
| DB / persistence | `src/storage.py` (SQLAlchemy, `data/stock_analysis.db`), repo pattern in `src/repositories/` | Store the profile |

**The gap this fills:** there is no concept of *your* risk tolerance, goals, or
target allocation, and nothing that synthesizes profile + holdings + candidates
into a plan made *for you*.

## 3. Decisions (from brainstorming)

1. **Output:** whole-portfolio plan (target allocation + buys/trims).
2. **Target source — Hybrid:** advisor *suggests* a target from the profile; the
   user edits and **locks** it; the planner always plans to the *locked* target.
3. **Allocation shape:** asset-class buckets (Equity / Fixed-income / Cash /
   Crypto / Commodity) + **guardrail caps** (max % per stock, max % per sector).
4. **Style — Conviction + guardrails:** concentrated bets are allowed while under
   their caps; the planner only trims a position/sector that is *over* its cap.
   It does **not** pull everything toward equal weight.
5. **Starting point — Support both:** if holdings exist → plan a *rebalance*; if
   the portfolio is empty → *deploy-from-cash* mode. One sizing engine, two entry
   points.
6. **Engine — Tool-calling AdvisorAgent** (`BaseAgent` subclass) over deterministic
   math tools, with a deterministic fallback when the LLM is unavailable.
7. **Home:** a new **Plan page** in the React dashboard (`apps/dsa-web`).
8. **Deferred, with honest in-plan notes:** taxes / account-type (TFSA/RRSP vs
   taxable), and full financial planning (income, contributions, dollar goals).

## 4. Scope

**In scope (v1):**
- Investor profile intake + storage (risk, horizon, goals, constraints, investable cash).
- Deterministic suggested-allocation from the profile; user edit + lock.
- Asset-class classification of holdings/candidates.
- Rebalance mode (real holdings) **and** deploy-from-cash mode (empty portfolio).
- Deterministic gap analysis + position sizing honoring guardrail caps.
- Equity buy candidates sourced from Overlap Board + deep-dive + watchlist.
- AdvisorAgent LLM layer for candidate selection + narrative, with fallback.
- Plan rendered on a new dashboard page and saved to `output/plans/`.
- A visible "educational, not licensed advice, does not account for taxes" banner.

**Out of scope (v1), named in the plan output so nothing is silently missing:**
- Tax / account-type awareness.
- Specific bond/crypto/commodity *security* selection (these sleeves get **dollar
  targets** and an optional broad-ETF suggestion, never fabricated picks).
- Full financial planning (retirement adequacy, contribution schedules, debt,
  emergency fund, assets outside the tracked portfolio).
- Multi-user auth/isolation (the data model is `owner_id`-keyed so it is
  *ready*, but v1 runs single-user under the existing admin login).
- Any trade execution.

## 5. Architecture

A new **`src/advisor/`** package plus a thin agent, API, and web page. The core is
**deterministic and LLM-free**; the agent is a smart layer on top that degrades
back to the deterministic core.

```
apps/dsa-web  ──HTTP──►  api/v1/endpoints/advisor.py
                              │
                              ▼
                     src/advisor/           (deterministic core — Phase 1)
                       profile.py           InvestorProfile + Target models
                       classify.py          ticker → asset class + sector
                       allocation.py        suggest_allocation(profile) → Target
                       snapshot.py          current picture (holdings | all-cash)
                       candidates.py        equity buys from overlap/deepdive/watchlist
                       gap.py               compute_gap(current, target)
                       sizing.py            size_positions(gap, candidates, caps, cash)
                       plan.py              Plan model + markdown renderer
                       repository.py        load/save profile (investor_profiles table)
                              ▲
                              │ calls the same functions as tools
                     src/agent/agents/advisor_agent.py   (LLM layer — Phase 2)
```

### 5.1 Components

Each is independently testable (clear input → output, no hidden state).

- **`profile.py`** — `InvestorProfile` (risk_tolerance ∈ {conservative, moderate,
  aggressive}; horizon_years; goals: list; constraints: e.g. exclude_crypto;
  investable_cash: float; base_currency) and `Target` (asset-class weights that
  sum to 1.0; `max_position_pct`; `max_sector_pct`; `locked: bool`). Pure data +
  validation (weights normalize/sum-check, caps in (0,1]).

- **`classify.py`** — `classify_asset(ticker) -> AssetClass` and
  `classify_sector(ticker) -> str`. Deterministic rules with a user-overridable
  map: known crypto (`*-USD` set: BTC-USD, ETH-USD…) → Crypto; known bond ETFs
  (BND, AGG, TLT…) → FixedIncome; known commodity ETFs (CORN, WEAT…) → Commodity;
  literal cash → Cash; **default → Equity**. Sector for equities comes from
  existing data (`data_provider` `get_stock_info` / fundamental data), falling
  back to "Unknown". The known-lists live in one small module-level config so
  they're easy to extend.

- **`allocation.py`** — `suggest_allocation(profile) -> Target`. Deterministic
  rule table mapping (risk_tolerance, horizon) → asset-class weights + default
  caps (e.g. moderate/10yr → Equity 0.75 / FixedIncome 0.15 / Cash 0.10, per-stock
  cap 0.15, per-sector cap 0.30). Honors constraints (exclude_crypto → 0 crypto).
  Returns an **unlocked** `Target` the user edits.

- **`snapshot.py`** — `get_current(owner_id) -> CurrentPicture`. If
  `PortfolioService.get_portfolio_snapshot` returns holdings → real positions
  (symbol, market value in base currency, asset class, sector). If empty →
  **deploy-from-cash** mode. Wraps FX via the existing service. This is the single
  place that decides rebalance-vs-deploy.
  **One money formula for both modes (removes ambiguity):** the plan's total
  investable base = `holdings_value + snapshot_cash + profile.investable_cash`,
  where `profile.investable_cash` is *new* money to deploy. Deploy-from-cash is
  simply the degenerate case (holdings and snapshot cash are 0, base =
  investable_cash). Target percentages are applied to this single base in both
  modes, so `gap.py`/`sizing.py` need no mode-specific branches.

- **`candidates.py`** — `fetch_candidates(limit) -> list[Candidate]`. Equity buy
  ideas merged from the Overlap Board, deep-dive results (carrying the AI verdict
  + trend + sentiment), and the watchlist. Deduplicated; carries a `fit_score`
  input (verdict rank, sector). No LLM.

- **`gap.py`** — `compute_gap(current, target) -> Gap`. Pure math: for each asset
  class, `target_value − current_value = delta`. Also computes per-sector and
  per-position overages vs the caps (drives *conviction + guardrails*: only
  positions/sectors **over** cap produce trims).

- **`sizing.py`** — `size_positions(gap, candidates, current, target) -> list[Action]`.
  Produces concrete `Action(side=buy|trim, symbol, amount, reason)`:
  - Trims: positions/sectors over cap, down to the cap.
  - Buys: fill positive asset-class deltas from candidates, each new position
    capped by `max_position_pct` and its sector by `max_sector_pct`, spending no
    more than available cash. Equity deltas → specific tickers; non-equity deltas
    → a single "increase <class> by $X (e.g. via <broad ETF>)" action.
  - If deltas can't be filled (too few qualifying candidates) → an
    `unallocated` note rather than forcing low-quality picks.

- **`plan.py`** — `Plan` (target, current, gap, actions, rationale, generated_at,
  disclaimer, mode) + `render_markdown(plan)`. The disclaimer and the deferred-
  scope notes (taxes, non-equity picks) are always present.

- **`repository.py`** — `InvestorProfileRepository` over a new `investor_profiles`
  table (`owner_id`, `profile_json`, `target_json`, `locked`, timestamps),
  following the existing `src/repositories/portfolio_repo.py` pattern.

- **`src/agent/agents/advisor_agent.py`** (Phase 2) — `AdvisorAgent(BaseAgent)`.
  `agent_name = "advisor"`. Tools wrap the deterministic functions
  (`suggest_allocation`, `get_current`, `fetch_candidates`, `compute_gap`,
  `size_positions`). The LLM: confirms/annotates the plan, **selects** which
  candidates fill each equity buy using deep-dive verdicts + profile fit, and
  writes the plain-English rationale. **All numbers come from the tools.** If the
  LLM errors or hits quota, `generate_plan` falls back to a deterministic
  selection (candidates ranked by verdict, filling deltas under caps) + a
  templated rationale — so a valid plan always returns.

- **`api/v1/endpoints/advisor.py`** — `GET/PUT /advisor/profile`,
  `POST /advisor/plan` (generate), `GET /advisor/plan` (last saved). Schemas in
  `api/v1/schemas/advisor.py`; registered in `api/v1/router.py`.

- **`apps/dsa-web` Plan page** — profile form + target editor (edit + lock) +
  Generate button + plan display (allocation gap, action list, rationale,
  disclaimer). New route/tab; API client following existing patterns.

## 6. Data flow (one plan)

```
profile (stored) ─► suggest_allocation ─► suggested Target ─► user edits ─► 🔒 lock
                                                                              │
POST /advisor/plan ◄──────────────────────────────────────────────────────────┘
        │
        ▼  generate_plan(owner_id):
   get_current            → holdings snapshot  OR  all-cash (deploy mode)
   classify each holding  → asset class + sector
   compute_gap            → per-class deltas + cap overages
   fetch_candidates       → equity ideas (overlap + deep-dive + watchlist)
   size_positions         → buys/trims under caps, within cash
   [LLM] select + narrate → pick equity buys by verdict/fit + write "why"
        │                    (fallback: deterministic pick + templated text)
        ▼
   Plan → render_markdown → saved to output/plans/plan_<date>.md → returned to page
```

## 7. Error handling & degradation

- **No profile** → generate returns a clear "create your profile first" error; the
  page routes the user to the form.
- **LLM unavailable / quota** → deterministic fallback path (proven by the
  deep-dive today). Plan still generates.
- **Empty portfolio** → `snapshot.py` auto-selects deploy-from-cash mode.
- **Insufficient candidates** → `unallocated` note in the plan, no forced buys.
- **Scan/deep-dive data stale or missing** → candidates fall back to
  watchlist-only; plan notes reduced candidate coverage.
- Each stage is wrapped so a partial failure yields a plan with a caveat, never a
  crash (mirrors the failure-isolated section builders in `main.py`).

## 8. Compliance framing

- Output is labeled **educational decision-support**, "not a licensed financial
  advisor," and "does not account for taxes." No personalized-advice claims.
- **No execution path** exists anywhere in the design — the plan proposes; the
  human decides and acts in their own brokerage.
- `owner_id`-keyed storage means multi-user is a *deliberate future step* (auth,
  isolation, and a heavier compliance review) — explicitly not enabled in v1.

## 9. Testing strategy (all offline, no live LLM / no quota)

- **Unit:** `classify` (each asset class + default); `suggest_allocation` (rule
  table + constraints + weights sum to 1); `compute_gap` (deltas + cap overages);
  `size_positions` (caps respected, cash respected, conviction bet under cap left
  alone, over-cap trimmed, unallocated when candidates scarce); `plan` renderer
  (disclaimer + notes always present).
- **Modes:** `snapshot` picks deploy-from-cash when empty, rebalance when holdings
  exist (both with a fake `PortfolioService`).
- **AdvisorAgent (Phase 2):** with a mocked LLM (fixed tool sequence) and with an
  LLM that raises → assert deterministic fallback still returns a valid plan and
  the numbers equal the deterministic tools' output.
- **API:** profile GET/PUT round-trip; plan generation with injected fakes.
- **Frontend:** Plan page component test (form → lock → generate → render) with a
  mocked API client.
- Run: `python3 -m pytest tests/test_advisor_*.py -q` and `npx vitest run`.

## 10. File structure

**Create:**
- `src/advisor/__init__.py`, `profile.py`, `classify.py`, `allocation.py`,
  `snapshot.py`, `candidates.py`, `gap.py`, `sizing.py`, `plan.py`, `repository.py`
- `src/agent/agents/advisor_agent.py` (Phase 2)
- `api/v1/endpoints/advisor.py`, `api/v1/schemas/advisor.py`
- `apps/dsa-web/src/…` Plan page + API client (exact paths in the plan)
- `tests/test_advisor_*.py`, frontend test
- DB migration/`storage.py` addition for `investor_profiles`

**Modify:**
- `api/v1/router.py` (register advisor router)
- `apps/dsa-web` route/nav registration

## 11. Phasing

**Phase 1 — deterministic planner (zero-LLM, usable on its own):** profile model +
storage + API, classifier, suggest/gap/sizing math with guardrail caps, plan
renderer, and the Plan page. Delivers real, generate-able plans. **This is the
first implementation plan.**

**Phase 2 — AdvisorAgent layer:** wrap the Phase 1 functions as agent tools for
smart candidate selection + narrative, degrading to Phase 1 when the LLM is
unavailable. Phase 1 *is* the fallback, so nothing is discarded.

Each phase produces working, tested software on its own.
