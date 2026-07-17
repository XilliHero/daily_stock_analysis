# -*- coding: utf-8 -*-
"""Tests for the real-time alert digest (offline, synthetic inputs)."""

from datetime import date
from types import SimpleNamespace

from src.scanner.alerts import build_alert_digest


def _watchlist(items):
    return SimpleNamespace(items=[SimpleNamespace(ticker=t, alerts=a) for t, a in items])


def _overlap(stocks):
    return SimpleNamespace(
        stocks=[
            SimpleNamespace(ticker=t, strategy_count=c, strategies=s)
            for t, c, s in stocks
        ]
    )


AS_OF = date(2026, 7, 16)


def test_none_when_nothing_actionable():
    wl = _watchlist([("XLE", ["✓ in today's value scan"]), ("CORN", ["🔺 crossed above MA20"])])
    ov = _overlap([("ABT", 2, ["value", "growth"])])  # only 2 strategies
    assert build_alert_digest(wl, ov, min_strategies=3, as_of=AS_OF) is None


def test_high_priority_watchlist_alerts_included():
    wl = _watchlist([
        ("OKLO", ["🟢 RSI oversold (27)", "📉 near 52-week low", "⚡ big move -8.7% today"]),
        ("XLE", ["🔺 crossed above MA20"]),  # medium — excluded
    ])
    digest = build_alert_digest(wl, None, as_of=AS_OF)
    assert digest is not None
    assert "OKLO" in digest and "RSI oversold" in digest
    assert "XLE" not in digest  # MA cross isn't high-priority


def test_high_conviction_overlap_included():
    ov = _overlap([
        ("LNC", 4, ["dividend", "growth", "recovery", "value"]),
        ("ABT", 3, ["dividend", "growth", "recovery"]),
        ("BNS", 2, ["dividend", "growth"]),  # below threshold
    ])
    digest = build_alert_digest(None, ov, min_strategies=3, as_of=AS_OF)
    assert "LNC" in digest and "4/4" in digest
    assert "ABT" in digest
    assert "BNS" not in digest


def test_health_line_alone_triggers_digest():
    digest = build_alert_digest(None, None, health_line="⚠️ Scan health degraded: growth missing", as_of=AS_OF)
    assert digest is not None
    assert "Scan health degraded" in digest


def test_header_has_date():
    wl = _watchlist([("OKLO", ["🟢 RSI oversold (27)"])])
    digest = build_alert_digest(wl, None, as_of=AS_OF)
    assert digest.startswith("🔔 Scanner Alerts — 2026-07-16")
