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
