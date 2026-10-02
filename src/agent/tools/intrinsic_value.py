# -*- coding: utf-8 -*-
"""Intrinsic-value estimation from yfinance ``.info`` data.

Two independent, transparent models are combined:

* **DCF** (Discounted Cash Flow) — projects the company's free cash flow for
  ``projection_years`` at a (conservatively capped) growth rate, adds a
  Gordon-growth terminal value, discounts everything back at the cost of
  capital, and converts enterprise value to a per-share equity value by
  removing net debt. This is the headline estimate when free cash flow is
  positive and share count is known.
* **Graham** — Benjamin Graham's two sanity checks: the *Graham Number*
  ``sqrt(22.5 * EPS * book value per share)`` (a conservative floor) and his
  revised growth formula ``EPS * (8.5 + 2g)``. Used as a cross-check against
  the DCF, and as a fallback headline when a DCF cannot be computed.

The defaults are deliberately **conservative** (growth capped at 10%, a 10%
discount rate, 2.5% terminal growth) so the fair value is hard to overstate.
Everything here is pure arithmetic over a plain dict, so it is deterministic
and unit-testable without any network access.

This is an educational estimate, **not** investment advice.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Conservative defaults (see module docstring).
DEFAULT_DISCOUNT_RATE = 0.10
DEFAULT_TERMINAL_GROWTH = 0.025
DEFAULT_GROWTH_CAP = 0.10
DEFAULT_GROWTH_FALLBACK = 0.05
DEFAULT_PROJECTION_YEARS = 10

# How close DCF and Graham must be (relative) to count as agreeing.
_AGREEMENT_TOLERANCE = 0.25
# Price must sit this far from fair value to read as under/over-valued.
_VERDICT_MARGIN = 0.10


def _num(info: Dict[str, Any], *keys: str) -> Optional[float]:
    """First finite, numeric value among ``keys`` in ``info`` (else None)."""
    for key in keys:
        val = info.get(key)
        if val is None:
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if f != f or f in (float("inf"), float("-inf")):  # NaN / inf guard
            continue
        return f
    return None


def _dcf_per_share(
    fcf: Optional[float],
    shares: Optional[float],
    growth: float,
    discount_rate: float,
    terminal_growth: float,
    years: int,
    net_debt: Optional[float],
) -> Optional[float]:
    """Per-share equity value from a one-stage DCF, or None if not computable."""
    if not fcf or fcf <= 0 or not shares or shares <= 0:
        return None
    if discount_rate <= terminal_growth:  # terminal value would diverge
        return None
    pv = 0.0
    cash_flow = fcf
    for year in range(1, years + 1):
        cash_flow *= 1 + growth
        pv += cash_flow / ((1 + discount_rate) ** year)
    terminal_value = cash_flow * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv += terminal_value / ((1 + discount_rate) ** years)
    equity_value = pv - (net_debt or 0.0)  # enterprise value -> equity value
    if equity_value <= 0:
        return None
    return round(equity_value / shares, 2)


def _graham_number(eps: Optional[float], bvps: Optional[float]) -> Optional[float]:
    """Graham Number: sqrt(22.5 * EPS * book value per share)."""
    if not eps or eps <= 0 or not bvps or bvps <= 0:
        return None
    return round((22.5 * eps * bvps) ** 0.5, 2)


def _graham_revised(eps: Optional[float], growth: float) -> Optional[float]:
    """Graham's revised formula: EPS * (8.5 + 2g), g in percentage points."""
    if not eps or eps <= 0:
        return None
    return round(eps * (8.5 + 2 * (growth * 100)), 2)


def compute_intrinsic_value(
    info: Dict[str, Any],
    *,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    terminal_growth: float = DEFAULT_TERMINAL_GROWTH,
    growth_cap: float = DEFAULT_GROWTH_CAP,
    projection_years: int = DEFAULT_PROJECTION_YEARS,
) -> Optional[Dict[str, Any]]:
    """Estimate a per-share intrinsic value from a yfinance ``.info`` dict.

    Returns a dict (see module docstring for fields) or ``None`` when neither a
    DCF nor a Graham estimate can be produced from the available data.
    """
    if not info:
        return None

    price = _num(info, "currentPrice", "regularMarketPrice", "previousClose")
    shares = _num(info, "sharesOutstanding")
    fcf = _num(info, "freeCashflow")
    eps = _num(info, "trailingEps")
    bvps = _num(info, "bookValue")  # yfinance reports book value per share

    # Conservative growth: prefer earnings growth, else revenue growth; never
    # extrapolate a negative or an above-cap rate.
    raw_growth = _num(info, "earningsGrowth")
    if raw_growth is None:
        raw_growth = _num(info, "revenueGrowth")
    if raw_growth is None:
        growth = DEFAULT_GROWTH_FALLBACK
    else:
        growth = max(0.0, min(raw_growth, growth_cap))

    total_debt = _num(info, "totalDebt")
    total_cash = _num(info, "totalCash", "totalCashPerShare")  # fallback rarely hit
    net_debt: Optional[float] = None
    if total_debt is not None or total_cash is not None:
        net_debt = (total_debt or 0.0) - (total_cash or 0.0)

    dcf = _dcf_per_share(fcf, shares, growth, discount_rate, terminal_growth, projection_years, net_debt)
    graham = _graham_number(eps, bvps)
    graham_revised = _graham_revised(eps, growth)

    if dcf is not None:
        fair_value, method = dcf, "dcf"
    elif graham is not None:
        fair_value, method = graham, "graham"
    else:
        return None  # nothing computable

    agreement: Optional[str] = None
    if dcf is not None and graham is not None:
        spread = abs(dcf - graham) / max(dcf, graham)
        agreement = "agree" if spread <= _AGREEMENT_TOLERANCE else "diverge"

    upside_pct: Optional[float] = None
    margin_of_safety_pct: Optional[float] = None
    verdict: Optional[str] = None
    if price and price > 0:
        upside_pct = round((fair_value - price) / price * 100, 1)
        margin_of_safety_pct = round((fair_value - price) / fair_value * 100, 1)
        if price <= fair_value * (1 - _VERDICT_MARGIN):
            verdict = "undervalued"
        elif price >= fair_value * (1 + _VERDICT_MARGIN):
            verdict = "overvalued"
        else:
            verdict = "fair"

    return {
        "fair_value": fair_value,
        "method": method,
        "current_price": round(price, 2) if price is not None else None,
        "upside_pct": upside_pct,
        "margin_of_safety_pct": margin_of_safety_pct,
        "verdict": verdict,
        "dcf": dcf,
        "graham": graham,
        "graham_revised": graham_revised,
        "agreement": agreement,
        "assumptions": {
            "growth_rate_pct": round(growth * 100, 1),
            "discount_rate_pct": round(discount_rate * 100, 1),
            "terminal_growth_pct": round(terminal_growth * 100, 1),
            "projection_years": projection_years,
        },
    }
