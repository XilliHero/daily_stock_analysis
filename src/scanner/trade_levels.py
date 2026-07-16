# -*- coding: utf-8 -*-
"""Actionable trade levels — turn a pick into a concrete setup.

Given a stock's current price and its ATR (already computed by the screener),
derive a buy zone, a stop-loss, and a target with a risk:reward ratio, so the
report says not just "this looks good" but "buy here, stop here, target here".

ATR-based by default (volatility-aware); falls back to fixed percentages when
ATR is unavailable. Educational levels only — not financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ATR multiples: enter on a small dip, stop below the noise, target a ~1.5:1 payoff.
_BUY_DIP_ATR = 0.5
_STOP_ATR = 2.0
_TARGET_ATR = 3.0

# Fallback fixed percentages when ATR is missing.
_FALLBACK_BUY_DIP = 0.02
_FALLBACK_STOP = 0.08
_FALLBACK_TARGET = 0.15


@dataclass
class TradeLevels:
    buy_low: float
    buy_high: float
    stop_loss: float
    target: float
    risk_reward: float


def compute_trade_levels(
    current_price: float, atr_pct: Optional[float] = None
) -> Optional[TradeLevels]:
    """Derive buy zone / stop / target from price and ATR%. None if price invalid."""
    if not current_price or current_price <= 0:
        return None

    if atr_pct and atr_pct > 0:
        atr = atr_pct / 100.0 * current_price
        buy_high = current_price
        buy_low = current_price - _BUY_DIP_ATR * atr
        stop = current_price - _STOP_ATR * atr
        target = current_price + _TARGET_ATR * atr
    else:
        buy_high = current_price
        buy_low = current_price * (1 - _FALLBACK_BUY_DIP)
        stop = current_price * (1 - _FALLBACK_STOP)
        target = current_price * (1 + _FALLBACK_TARGET)

    stop = max(stop, 0.01)
    entry = (buy_low + buy_high) / 2
    risk = entry - stop
    reward = target - entry
    rr = round(reward / risk, 1) if risk > 0 else 0.0
    return TradeLevels(
        buy_low=buy_low, buy_high=buy_high, stop_loss=stop, target=target, risk_reward=rr
    )


def format_trade_setup(current_price: float, levels: TradeLevels) -> str:
    """One-line markdown fragment for a detail card."""
    stop_pct = (levels.stop_loss - current_price) / current_price * 100
    tgt_pct = (levels.target - current_price) / current_price * 100
    return (
        f"Buy ${levels.buy_low:.2f}–${levels.buy_high:.2f} · "
        f"Stop ${levels.stop_loss:.2f} ({stop_pct:+.0f}%) · "
        f"Target ${levels.target:.2f} ({tgt_pct:+.0f}%) · "
        f"R:R {levels.risk_reward:.1f}:1"
    )
