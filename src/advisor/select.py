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
