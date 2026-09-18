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
    engine: str = "deterministic"   # "ai" when the LLM refined candidate selection
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def _plans_dir() -> str:
    return os.path.join("output", "plans")


def _default_repo():
    from src.advisor.repository import InvestorProfileRepository
    return InvestorProfileRepository()


def _default_select(candidates, profile, equity_budget, max_position_pct):
    from src.advisor.select import select_candidates_with_llm
    return select_candidates_with_llm(candidates, profile, equity_budget, max_position_pct)


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

    # Smart path: the LLM only reorders/filters candidates + writes prose. Numbers
    # stay in size_positions. Any failure or empty result falls back to deterministic.
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
            candidates, llm_rationale = picked
            engine = "ai"

    actions = size_positions(gap, candidates, current, target)
    rationale = llm_rationale or _rationale(current, target, actions)
    plan = Plan(mode=current.mode, base=round(current.base, 2), target=dict(target.weights),
                actions=actions, rationale=rationale, engine=engine)

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
        f"_{plan.mode.title()} mode · base {_fmt(plan.base)} · "
        f"{'AI-refined' if plan.engine == 'ai' else 'deterministic'} · {plan.generated_at}_",
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
