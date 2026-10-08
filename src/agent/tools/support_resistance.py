# -*- coding: utf-8 -*-
"""Support & resistance levels via classic floor-trader pivot points.

Pivot points are a standard, fully deterministic way to project the next
session's support and resistance from the most recent completed bar's high,
low and close:

    P  = (High + Low + Close) / 3           (the central pivot)
    R1 = 2P - Low        S1 = 2P - High
    R2 = P + (High - Low) S2 = P - (High - Low)
    R3 = High + 2(P - Low) S3 = Low - 2(High - P)

Levels above the current price act as resistance, levels below it as support.
This is pure arithmetic over price data, so it works for every market (not
only the yfinance ones) and is trivially unit-testable.

Educational technical levels — not a trade recommendation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def compute_pivot_levels(high: float, low: float, close: float) -> Dict[str, float]:
    """Classic pivot, R1–R3 and S1–S3 from one bar's high/low/close."""
    p = (high + low + close) / 3.0
    rng = high - low
    return {
        "pivot": round(p, 2),
        "r1": round(2 * p - low, 2),
        "s1": round(2 * p - high, 2),
        "r2": round(p + rng, 2),
        "s2": round(p - rng, 2),
        "r3": round(high + 2 * (p - low), 2),
        "s3": round(low - 2 * (high - p), 2),
    }


def _nearest(levels: List[float], price: float, *, below: bool) -> Optional[float]:
    """Highest level below ``price`` (support) or lowest above it (resistance)."""
    if below:
        candidates = [lv for lv in levels if lv < price]
        return max(candidates) if candidates else None
    candidates = [lv for lv in levels if lv > price]
    return min(candidates) if candidates else None


def _col(row: Dict[str, Any], *names: str) -> Optional[float]:
    """First finite numeric value among ``names`` (case-insensitive) in a row."""
    for name in names:
        for key in (name, name.capitalize(), name.upper()):
            if key in row and row[key] is not None:
                try:
                    f = float(row[key])
                except (TypeError, ValueError):
                    continue
                if f == f and f not in (float("inf"), float("-inf")):
                    return f
    return None


def pivot_levels_from_history(df: Any) -> Optional[Dict[str, Any]]:
    """Compute pivot support/resistance from an OHLC price-history DataFrame.

    Uses the most recent completed bar for the high/low/close and its close as
    the reference price. Returns a dict (levels, nearest support/resistance,
    basis date) or ``None`` when the data is missing or unusable.
    """
    if df is None:
        return None
    try:
        if getattr(df, "empty", False) or len(df) == 0:
            return None
        last = df.iloc[-1].to_dict()
    except Exception:
        return None

    high = _col(last, "high")
    low = _col(last, "low")
    close = _col(last, "close")
    if high is None or low is None or close is None or high < low:
        return None

    levels = compute_pivot_levels(high, low, close)
    price = round(close, 2)
    ordered = [levels["s3"], levels["s2"], levels["s1"], levels["pivot"], levels["r1"], levels["r2"], levels["r3"]]

    basis_date = None
    raw_date = last.get("date") or last.get("Date") or last.get("trade_date")
    if raw_date is not None:
        basis_date = str(raw_date)[:10]

    return {
        **levels,
        "current_price": price,
        "nearest_support": _nearest(ordered, price, below=True),
        "nearest_resistance": _nearest(ordered, price, below=False),
        "basis_date": basis_date,
        "period": "daily",
    }
