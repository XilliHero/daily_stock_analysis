"""
Technical Indicator Engine
==========================
Computes MA5/MA10/MA20, RSI-14, bias ratio, trend signal, and battle plan
for each ticker using the downloaded price history.  No AI or API key required.

Decision logic mirrors the old report's 6-point checklist:
  1. Bullish MA alignment  (MA5 > MA10 > MA20)
  2. Reasonable bias ratio (|bias from MA5| < 5%)
  3. RSI in healthy zone   (30–70)
  4. No major negative news
  5. Price above all MAs
  6. Not overbought (RSI < 70)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TechnicalSignal:
    ticker: str

    # Price & indicators
    price: float | None
    ma5: float | None
    ma10: float | None
    ma20: float | None
    rsi: float | None
    bias_pct: float | None      # (price – MA5) / MA5 × 100

    # Trend & decision
    trend: str                  # "Strong Bullish" / "Bullish" / "Neutral" / "Bearish" / "Strong Bearish" / "Unknown"
    signal: str                 # "Strong Buy" / "Buy" / "Watch" / "Sell"
    score: int                  # 0–100
    action_urgency: str         # "Act Now" / "Today" / "Not Urgent"

    # Key levels
    support: float | None
    resistance: float | None

    # Battle plan
    ideal_entry: float | None
    secondary_entry: float | None
    stop_loss: float | None
    target: float | None
    position_size_pct: int      # 0, 5, or 10

    # Checklist  →  list of (label, passed:True/False/None)
    checklist: list[tuple[str, bool | None]] = field(default_factory=list)

    # Risk flags  →  list of plain strings
    flags: list[str] = field(default_factory=list)

    # One-line summary sentence
    summary: str = ""


# ---------------------------------------------------------------------------
# Core indicator helpers
# ---------------------------------------------------------------------------

def _ma(series: pd.Series, window: int) -> float | None:
    if len(series) < window:
        return None
    return round(float(series.tail(window).mean()), 4)


def _rsi(series: pd.Series, period: int = 14) -> float | None:
    """Wilder's RSI via exponential moving average (com = period − 1)."""
    if len(series) < period + 1:
        return None
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 1)


def _support_resistance(
    series: pd.Series, lookback: int = 20
) -> tuple[float | None, float | None]:
    """Simple support / resistance from recent low / high."""
    recent = series.tail(lookback).dropna()
    if recent.empty:
        return None, None
    return round(float(recent.min()), 2), round(float(recent.max()), 2)


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def _compute_signal(
    price: float | None,
    ma5: float | None,
    ma10: float | None,
    ma20: float | None,
    rsi: float | None,
    bias_pct: float | None,
) -> tuple[str, str, int, str]:
    """Return (trend, signal, score, urgency)."""

    if price is None or ma5 is None or ma10 is None or ma20 is None:
        return "Unknown", "Watch", 30, "Not Urgent"

    bullish = ma5 > ma10 > ma20
    bearish = ma5 < ma10 < ma20
    overbought = rsi is not None and rsi >= 70
    oversold   = rsi is not None and rsi <= 30
    bias_safe  = bias_pct is not None and abs(bias_pct) < 5.0
    bias_high  = bias_pct is not None and bias_pct >= 5.0

    if bullish:
        if not overbought and bias_safe:
            return "Strong Bullish", "Strong Buy", 85, "Act Now"
        elif overbought:
            return "Bullish", "Buy", 62, "Today"
        elif bias_high:
            return "Bullish", "Buy", 58, "Today"
        else:
            return "Bullish", "Buy", 70, "Today"
    elif bearish:
        if oversold:
            # Potential bounce candidate — still watch but score higher
            return "Strong Bearish", "Watch", 35, "Not Urgent"
        return "Strong Bearish", "Watch", 20, "Not Urgent"
    elif ma5 > ma10:
        return "Neutral", "Watch", 50, "Not Urgent"
    else:
        return "Bearish", "Watch", 30, "Not Urgent"


def _compute_battle_plan(
    price: float | None,
    ma5: float | None,
    ma10: float | None,
    ma20: float | None,
    support: float | None,
    resistance: float | None,
    signal: str,
) -> tuple[float | None, float | None, float | None, float | None, int]:
    """Return (ideal_entry, secondary_entry, stop_loss, target, position_size_pct)."""

    if price is None or ma5 is None or ma10 is None or ma20 is None:
        return None, None, None, None, 0

    if signal in ("Strong Buy", "Buy"):
        ideal     = round(ma5 * 0.999, 2)    # just under MA5
        secondary = round(ma10, 2)
        stop      = round(ma20 * 0.98, 2)    # 2 % below MA20
        # Target: nearest resistance if meaningfully above price, else +6 %
        if resistance and resistance > price * 1.01:
            target = round(resistance, 2)
        else:
            target = round(price * 1.06, 2)
        size = 10 if signal == "Strong Buy" else 5
    else:
        ideal     = None
        secondary = None
        stop      = round(support * 0.97, 2) if support else round(price * 0.95, 2)
        target    = None
        size      = 0

    return ideal, secondary, stop, target, size


def _build_checklist(
    price: float | None,
    ma5: float | None,
    ma10: float | None,
    ma20: float | None,
    bias_pct: float | None,
    rsi: float | None,
) -> list[tuple[str, bool | None]]:
    """Return 6-item checklist matching the format of the old report."""

    def _tri(condition, fallback: bool | None = None) -> bool | None:
        try:
            return bool(condition)
        except Exception:
            return fallback

    ma_ok = _tri(
        ma5 is not None and ma10 is not None and ma20 is not None and ma5 > ma10 > ma20
    ) if ma5 is not None else None

    bias_ok = _tri(abs(bias_pct) < 5.0) if bias_pct is not None else None

    rsi_ok = _tri(30 <= rsi <= 70) if rsi is not None else None

    above_ma20 = (
        _tri(price > ma20) if (price is not None and ma20 is not None) else None
    )

    return [
        ("Bullish MA alignment (MA5 > MA10 > MA20)", ma_ok),
        ("Reasonable bias ratio (|bias| < 5%)", bias_ok),
        ("RSI in healthy zone (30–70)", rsi_ok),
        ("No major negative news", True),         # filled by caller if news available
        ("Price above MA20 (trend intact)", above_ma20),
        ("Not overbought (RSI < 70)", _tri(rsi < 70) if rsi is not None else None),
    ]


def _build_summary(
    ticker: str, signal: str, trend: str,
    ma5: float | None, ma10: float | None, ma20: float | None,
    bias_pct: float | None, rsi: float | None,
) -> str:
    if signal == "Strong Buy":
        return (
            f"{ticker} shows a strong bullish trend (MA5 > MA10 > MA20) with "
            f"bias of {bias_pct:+.2f}% — optimal entry conditions, act now."
        )
    elif signal == "Buy":
        rsi_note = f"RSI {rsi:.0f} — slight caution." if rsi and rsi >= 65 else "Entry on pullback to MA5 preferred."
        return f"{ticker} is in a bullish alignment. {rsi_note}"
    elif trend in ("Strong Bearish", "Bearish"):
        return (
            f"{ticker} is in a confirmed bearish trend (MA5 < MA10 < MA20); "
            f"avoid new long positions. Monitor for reversal above MA20."
        )
    elif trend == "Neutral":
        return f"{ticker} shows mixed signals — MAs are not yet aligned. Watch for direction."
    else:
        return f"{ticker}: insufficient data for a confident recommendation."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_technicals(
    stock_prices: pd.DataFrame,
    quotes: dict[str, dict],
) -> dict[str, TechnicalSignal]:
    """
    Compute full technical analysis for every ticker.

    Parameters
    ----------
    stock_prices : DataFrame — closing prices (rows = dates, cols = tickers)
    quotes       : dict from fetch_current_quotes — live price + intraday hi/lo

    Returns
    -------
    dict mapping ticker → TechnicalSignal
    """
    results: dict[str, TechnicalSignal] = {}

    for ticker in stock_prices.columns:
        series = stock_prices[ticker].dropna()
        q      = quotes.get(ticker, {})

        # Use live price if available, otherwise last close
        price = q.get("price") or (float(series.iloc[-1]) if len(series) > 0 else None)

        # Indicators
        ma5  = _ma(series, 5)
        ma10 = _ma(series, 10)
        ma20 = _ma(series, 20)
        rsi  = _rsi(series)

        # Support / resistance from recent 20-day window, sharpened by intraday
        sup, res = _support_resistance(series, 20)
        day_low  = q.get("day_low")
        day_high = q.get("day_high")
        if day_low  and sup  and day_low  < sup:  sup = round(day_low,  2)
        if day_high and res  and day_high > res:  res = round(day_high, 2)

        bias_pct = (
            round((price - ma5) / ma5 * 100, 2)
            if (price and ma5) else None
        )

        trend, signal, score, urgency = _compute_signal(
            price, ma5, ma10, ma20, rsi, bias_pct
        )

        ideal, secondary, stop, target, pos_size = _compute_battle_plan(
            price, ma5, ma10, ma20, sup, res, signal
        )

        # Risk flags
        flags: list[str] = []
        if rsi is not None and rsi >= 70:
            flags.append(f"RSI is overbought ({rsi:.1f} > 70) — elevated pullback risk.")
        if rsi is not None and rsi <= 30:
            flags.append(f"RSI is oversold ({rsi:.1f} < 30) — potential bounce candidate.")
        if bias_pct is not None and bias_pct >= 7.0:
            flags.append(f"High bias ({bias_pct:+.2f}% above MA5) — avoid chasing the move.")
        if ma5 and ma10 and ma20 and ma5 < ma10 < ma20:
            flags.append("Bearish MA alignment — no new long positions recommended.")

        checklist = _build_checklist(price, ma5, ma10, ma20, bias_pct, rsi)
        summary   = _build_summary(ticker, signal, trend, ma5, ma10, ma20, bias_pct, rsi)

        results[ticker] = TechnicalSignal(
            ticker=ticker,
            price=price,
            ma5=ma5,
            ma10=ma10,
            ma20=ma20,
            rsi=rsi,
            bias_pct=bias_pct,
            trend=trend,
            signal=signal,
            score=score,
            action_urgency=urgency,
            support=sup,
            resistance=res,
            ideal_entry=ideal,
            secondary_entry=secondary,
            stop_loss=stop,
            target=target,
            position_size_pct=pos_size,
            checklist=checklist,
            flags=flags,
            summary=summary,
        )

    return results
