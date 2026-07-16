# -*- coding: utf-8 -*-
"""Watchlist Spotlight — a daily read on the user's own tracked tickers.

Separate from the 947-stock scan: this focuses on the personal STOCK_LIST
(e.g. XLE, BTC-USD, CORN, OKLO, MDA.TO, SIA.TO), fetching each one's current
price and technical state, raising simple alerts (RSI extremes, MA crosses,
52-week extremes, big moves), and noting when a watchlist name also shows up
in today's scan.

The price fetcher and scan-membership lookup are injectable so the logic is
unit-testable without network access.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from src.scanner.screener_agent import ScreenerAgent, _extract_ticker_df

logger = logging.getLogger(__name__)

HISTORY_PERIOD = "1y"  # enough for 52-week range and MA50
NEAR_EXTREME_PCT = 3.0  # within 3% of 52w high/low
BIG_MOVE_PCT = 5.0

PriceFetcher = Callable[[List[str]], Dict[str, "object"]]
ScanMembership = Callable[[List[str]], Dict[str, List[str]]]


@dataclass
class WatchItem:
    ticker: str
    price: float
    change_pct: float = 0.0
    ma20: Optional[float] = None
    ma50: Optional[float] = None
    rsi: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    drawdown_pct: Optional[float] = None  # % below the 52w high
    in_scan: List[str] = field(default_factory=list)  # strategies it appears in today
    alerts: List[str] = field(default_factory=list)


@dataclass
class WatchlistReport:
    items: List[WatchItem] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)  # tickers with no price data

    @property
    def has_any(self) -> bool:
        return bool(self.items)


def get_watchlist_tickers() -> List[str]:
    """Parse STOCK_LIST from the environment into a ticker list."""
    raw = os.getenv("STOCK_LIST", "") or ""
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _default_price_fetcher(tickers: List[str]) -> Dict[str, "object"]:
    import yfinance as yf

    if not tickers:
        return {}
    try:
        data = yf.download(
            tickers, period=HISTORY_PERIOD, group_by="ticker",
            progress=False, threads=True, timeout=30,
        )
    except Exception as exc:
        logger.warning("[watchlist] price fetch failed: %s", exc)
        return {}
    out: Dict[str, object] = {}
    for t in tickers:
        df = _extract_ticker_df(data, t, len(tickers))
        if df is not None and not df.empty:
            out[t] = df
    return out


def _default_scan_membership(tickers: List[str]) -> Dict[str, List[str]]:
    """Which strategies list each ticker in today's (latest) scan."""
    from src.scanner.changes import _all_dates, _group, _top_n
    from src.scanner.scorecard.pick_store import load_all_picks

    group = _group(load_all_picks())
    dates = _all_dates(group)
    if not dates:
        return {}
    latest = dates[-1]
    wanted = set(tickers)
    membership: Dict[str, List[str]] = {}
    for strat, by_date in group.items():
        day = by_date.get(latest)
        if not day:
            continue
        for ticker in _top_n(day, 50):
            if ticker in wanted:
                membership.setdefault(ticker, []).append(strat)
    return {t: sorted(s) for t, s in membership.items()}


def _rsi(close: np.ndarray) -> Optional[float]:
    if len(close) < 15:
        return None
    try:
        return round(float(ScreenerAgent._compute_rsi(close)), 1)
    except Exception:
        return None


def _compute_item(ticker: str, df, in_scan: List[str]) -> Optional[WatchItem]:
    close = df["Close"].dropna().to_numpy(dtype=float)
    if len(close) < 2:
        return None
    high = df["High"].dropna().to_numpy(dtype=float)
    low = df["Low"].dropna().to_numpy(dtype=float)

    current = float(close[-1])
    prev = float(close[-2])
    change_pct = (current - prev) / prev * 100 if prev else 0.0
    ma20 = float(np.mean(close[-20:])) if len(close) >= 20 else None
    ma50 = float(np.mean(close[-50:])) if len(close) >= 50 else None
    rsi = _rsi(close)
    high_52w = float(np.max(high)) if len(high) else None
    low_52w = float(np.min(low)) if len(low) else None
    drawdown = (high_52w - current) / high_52w * 100 if high_52w else None

    item = WatchItem(
        ticker=ticker, price=current, change_pct=change_pct,
        ma20=ma20, ma50=ma50, rsi=rsi, high_52w=high_52w, low_52w=low_52w,
        drawdown_pct=drawdown, in_scan=in_scan,
    )
    item.alerts = _build_alerts(item, close, ma20)
    return item


def _build_alerts(item: WatchItem, close: np.ndarray, ma20: Optional[float]) -> List[str]:
    alerts: List[str] = []
    if item.rsi is not None:
        if item.rsi < 30:
            alerts.append(f"🟢 RSI oversold ({item.rsi})")
        elif item.rsi > 70:
            alerts.append(f"🔴 RSI overbought ({item.rsi})")

    # MA20 cross: compare yesterday vs today relative to a 20-day mean.
    if ma20 is not None and len(close) >= 21:
        prev_ma20 = float(np.mean(close[-21:-1]))
        prev_close = float(close[-2])
        if prev_close <= prev_ma20 and close[-1] > ma20:
            alerts.append("🔺 crossed above MA20")
        elif prev_close >= prev_ma20 and close[-1] < ma20:
            alerts.append("🔻 crossed below MA20")

    if item.high_52w and item.price >= item.high_52w * (1 - NEAR_EXTREME_PCT / 100):
        alerts.append("📈 near 52-week high")
    if item.low_52w and item.price <= item.low_52w * (1 + NEAR_EXTREME_PCT / 100):
        alerts.append("📉 near 52-week low")

    if abs(item.change_pct) >= BIG_MOVE_PCT:
        alerts.append(f"⚡ big move {item.change_pct:+.1f}% today")

    if item.in_scan:
        alerts.append(f"✓ in today's {', '.join(item.in_scan)} scan")

    return alerts


def build_watchlist(
    tickers: Optional[List[str]] = None,
    price_fetcher: Optional[PriceFetcher] = None,
    scan_membership: Optional[ScanMembership] = None,
) -> Optional[WatchlistReport]:
    """Build the watchlist spotlight. Returns None if STOCK_LIST is empty."""
    tickers = tickers if tickers is not None else get_watchlist_tickers()
    if not tickers:
        return None
    fetch = price_fetcher or _default_price_fetcher
    membership_fn = scan_membership or _default_scan_membership

    price_data = fetch(tickers)
    membership = membership_fn(tickers)

    report = WatchlistReport()
    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None:
            report.missing.append(ticker)
            continue
        item = _compute_item(ticker, df, membership.get(ticker, []))
        if item is None:
            report.missing.append(ticker)
        else:
            report.items.append(item)
    return report


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _fmt(value: Optional[float], suffix: str = "") -> str:
    return f"{value:.2f}{suffix}" if value is not None else "—"


def _ma_state(item: WatchItem) -> str:
    if item.ma20 is None:
        return "—"
    return "above" if item.price >= item.ma20 else "below"


def render_watchlist_markdown(report: WatchlistReport) -> str:
    """Return the ``## ⭐ Watchlist Spotlight`` markdown block."""
    lines: List[str] = ["## ⭐ Watchlist Spotlight"]
    lines.append("Your tracked tickers — price, momentum, and today's alerts.")
    lines.append("")
    lines.append("| Ticker | Price | 1D | RSI | vs MA20 | Below 52w High |")
    lines.append("|--------|-------|----|----|---------|----------------|")
    for it in report.items:
        rsi = f"{it.rsi:.0f}" if it.rsi is not None else "—"
        dd = f"-{it.drawdown_pct:.0f}%" if it.drawdown_pct is not None else "—"
        lines.append(
            f"| **{it.ticker}** | ${_fmt(it.price)} | {it.change_pct:+.1f}% "
            f"| {rsi} | {_ma_state(it)} | {dd} |"
        )

    alert_items = [it for it in report.items if it.alerts]
    if alert_items:
        lines.append("")
        lines.append("**Alerts:**")
        for it in alert_items:
            lines.append(f"- **{it.ticker}**: {' · '.join(it.alerts)}")

    if report.missing:
        lines.append(f"\n_No data for: {', '.join(report.missing)}._")

    return "\n".join(lines)
