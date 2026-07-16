# -*- coding: utf-8 -*-
"""Tests for actionable trade-level computation."""

from src.scanner.trade_levels import compute_trade_levels, format_trade_setup


def test_atr_based_levels_ordering():
    levels = compute_trade_levels(current_price=100.0, atr_pct=2.0)  # ATR = $2
    assert levels is not None
    # buy_low < buy_high(=price) < target ; stop < buy_low
    assert levels.stop_loss < levels.buy_low < levels.buy_high == 100.0 < levels.target
    # ATR multiples: stop = 100 - 2*2 = 96, target = 100 + 3*2 = 106, buy_low = 99
    assert levels.stop_loss == 96.0
    assert levels.target == 106.0
    assert levels.buy_low == 99.0
    assert levels.risk_reward > 0


def test_fallback_percentages_when_no_atr():
    levels = compute_trade_levels(current_price=200.0, atr_pct=0.0)
    assert levels.buy_low == 200.0 * 0.98
    assert levels.stop_loss == 200.0 * 0.92
    assert levels.target == 200.0 * 1.15


def test_none_for_invalid_price():
    assert compute_trade_levels(0.0, 2.0) is None
    assert compute_trade_levels(-5.0, 2.0) is None


def test_risk_reward_is_reasonable():
    # ATR setup: entry ~99.5, risk ~3.5, reward ~6.5 → R:R ~1.9
    levels = compute_trade_levels(100.0, 2.0)
    assert 1.5 <= levels.risk_reward <= 2.5


def test_format_trade_setup_string():
    levels = compute_trade_levels(100.0, 2.0)
    s = format_trade_setup(100.0, levels)
    assert "Buy $99.00–$100.00" in s
    assert "Stop $96.00" in s
    assert "Target $106.00" in s
    assert "R:R" in s
