# -*- coding: utf-8 -*-
"""Scorecard evaluator — measures how past picks performed.

For every ``(strategy, ticker)`` pair it finds the **first day** the stock
appeared in a scan and the price then, fetches the current price, and computes
the cumulative return since that first appearance. Results are aggregated per
strategy (and overall): average return, hit-rate, best/worst, count.

Pure price math — no LLM calls. The current-price fetcher is injectable so the
logic is unit-testable without network access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence

from src.scanner.scorecard.pick_store import PickRecord, load_all_picks

logger = logging.getLogger(__name__)

# Fetch split/dividend-adjusted daily closes: (tickers, start_date) ->
# {TICKER: {"YYYY-MM-DD": adj_close, ...}}. Comparing adjusted history at the
# first-appearance date against the latest adjusted close makes returns immune
# to stock splits (e.g. KLAC's 10:1 split) and dividend distortions.
PriceHistoryFetcher = Callable[[Sequence[str], date], Dict[str, Dict[str, float]]]


@dataclass
class PickPerformance:
    """Cumulative performance of one (strategy, ticker) since first appearance.

    ``baseline_price``/``current_price`` are split- and dividend-adjusted closes,
    so ``return_pct`` is a true total return even across corporate actions.
    """

    strategy: str
    ticker: str
    name: str
    first_date: str
    baseline_price: float
    current_price: float
    return_pct: float  # e.g. 0.123 == +12.3%
    days_tracked: int
    active: bool  # still present in the strategy's most recent scan


@dataclass
class StrategyScore:
    """Aggregated performance for one strategy (or 'ALL')."""

    strategy: str
    count: int = 0
    avg_return: float = 0.0
    hit_rate: float = 0.0  # fraction of picks with return > 0
    median_days_tracked: float = 0.0
    n_no_data: int = 0
    best: Optional[PickPerformance] = None
    worst: Optional[PickPerformance] = None
    picks: List[PickPerformance] = field(default_factory=list)


@dataclass
class Scorecard:
    """Full scorecard as of a given date."""

    as_of: str
    per_strategy: List[StrategyScore] = field(default_factory=list)
    overall: Optional[StrategyScore] = None
    n_tickers: int = 0
    n_no_data: int = 0
    since_date: str = ""


def _parse_date(iso: str) -> Optional[date]:
    try:
        return datetime.strptime(iso, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _default_price_fetcher(
    tickers: Sequence[str], start_date: date
) -> Dict[str, Dict[str, float]]:
    """Batch-fetch adjusted daily closes from ``start_date`` via yfinance.

    Returns ``{TICKER: {"YYYY-MM-DD": adj_close}}``. Chunks the request; tickers
    with no data are simply absent. A few days of pad before ``start_date`` gives
    a baseline bar even when the first scan landed on a market holiday.
    """
    import yfinance as yf

    history: Dict[str, Dict[str, float]] = {}
    tickers = [t for t in tickers if t]
    start = (start_date - timedelta(days=5)).isoformat()
    chunk_size = 50
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            data = yf.download(
                chunk,
                start=start,
                progress=False,
                threads=True,
                auto_adjust=True,
            )
        except Exception as exc:  # network / yfinance internal errors
            logger.warning("[scorecard] price fetch failed for chunk %d: %s", i, exc)
            continue
        _merge_close_history(data, chunk, history)
    return history


def _merge_close_history(data, chunk: Sequence[str], out: Dict[str, Dict[str, float]]) -> None:
    """Extract per-ticker {date: adj_close} maps from a yfinance download frame."""
    if data is None or getattr(data, "empty", True):
        return
    try:
        close = data["Close"]
    except (KeyError, TypeError):
        return

    def _series_to_map(series) -> Dict[str, float]:
        series = series.dropna()
        return {
            idx.strftime("%Y-%m-%d"): float(val)
            for idx, val in series.items()
        }

    # Multiple tickers → DataFrame (columns = tickers); single → Series.
    if hasattr(close, "columns"):
        for ticker in close.columns:
            m = _series_to_map(close[ticker])
            if m:
                out[str(ticker).upper()] = m
    else:
        if len(chunk) == 1:
            m = _series_to_map(close)
            if m:
                out[str(chunk[0]).upper()] = m


def _baseline_and_current(
    hist: Dict[str, float], first_date: date
) -> Optional[tuple[float, float]]:
    """From a {date: adj_close} map, return (baseline at/after first_date, latest)."""
    if not hist:
        return None
    dated = sorted((datetime.strptime(d, "%Y-%m-%d").date(), p) for d, p in hist.items())
    baseline = next((p for d, p in dated if d >= first_date), None)
    if baseline is None or baseline <= 0:
        return None
    current = dated[-1][1]
    if current <= 0:
        return None
    return baseline, current


def _group_first_appearance(records: List[PickRecord]):
    """Return ({(strategy,ticker): (first_date, first_price, name)},
    {strategy: latest_date}) — picks with no usable price are skipped."""
    first: Dict[tuple, tuple] = {}
    latest_by_strategy: Dict[str, date] = {}

    for r in records:
        d = _parse_date(r.scan_date)
        if d is None or r.price is None or r.price <= 0 or not r.ticker:
            continue
        strat = r.strategy
        if strat not in latest_by_strategy or d > latest_by_strategy[strat]:
            latest_by_strategy[strat] = d

        key = (strat, r.ticker)
        prev = first.get(key)
        if prev is None or d < prev[0]:
            first[key] = (d, r.price, r.name)

    # Track which (strategy, ticker) appeared on the strategy's latest date.
    active_keys = set()
    for r in records:
        d = _parse_date(r.scan_date)
        if d is None:
            continue
        if latest_by_strategy.get(r.strategy) == d and r.price:
            active_keys.add((r.strategy, r.ticker))

    return first, latest_by_strategy, active_keys


def _aggregate(strategy: str, picks: List[PickPerformance], n_no_data: int) -> StrategyScore:
    score = StrategyScore(strategy=strategy, picks=picks, n_no_data=n_no_data)
    if not picks:
        return score
    returns = [p.return_pct for p in picks]
    score.count = len(picks)
    score.avg_return = sum(returns) / len(returns)
    score.hit_rate = sum(1 for r in returns if r > 0) / len(returns)
    score.median_days_tracked = float(median(p.days_tracked for p in picks))
    score.best = max(picks, key=lambda p: p.return_pct)
    score.worst = min(picks, key=lambda p: p.return_pct)
    return score


def build_scorecard(
    records: Optional[List[PickRecord]] = None,
    price_fetcher: Optional[PriceHistoryFetcher] = None,
    as_of: Optional[date] = None,
) -> Scorecard:
    """Compute the full scorecard from pick records + adjusted price history."""
    if records is None:
        records = load_all_picks()
    fetch = price_fetcher or _default_price_fetcher
    today = as_of or date.today()

    first, _latest, active_keys = _group_first_appearance(records)

    unique_tickers = sorted({ticker for (_strat, ticker) in first.keys()})
    earliest = min((fd for (fd, _p, _n) in first.values()), default=today)
    history = fetch(unique_tickers, earliest) if unique_tickers else {}
    history = {str(k).upper(): v for k, v in history.items()}

    per_strategy_picks: Dict[str, List[PickPerformance]] = {}
    per_strategy_nodata: Dict[str, int] = {}
    total_no_data = 0

    for (strat, ticker), (first_date, _recorded_price, name) in first.items():
        prices = _baseline_and_current(history.get(ticker.upper(), {}), first_date)
        if prices is None:
            per_strategy_nodata[strat] = per_strategy_nodata.get(strat, 0) + 1
            total_no_data += 1
            continue
        baseline, current = prices
        perf = PickPerformance(
            strategy=strat,
            ticker=ticker,
            name=name,
            first_date=first_date.isoformat(),
            baseline_price=baseline,
            current_price=current,
            return_pct=(current - baseline) / baseline,
            days_tracked=(today - first_date).days,
            active=(strat, ticker) in active_keys,
        )
        per_strategy_picks.setdefault(strat, []).append(perf)

    per_strategy = [
        _aggregate(strat, picks, per_strategy_nodata.get(strat, 0))
        for strat, picks in sorted(per_strategy_picks.items())
    ]
    # Include strategies that had picks but all lacked price data.
    for strat, n in per_strategy_nodata.items():
        if strat not in per_strategy_picks:
            per_strategy.append(_aggregate(strat, [], n))
    per_strategy.sort(key=lambda s: s.strategy)

    all_picks = [p for picks in per_strategy_picks.values() for p in picks]
    overall = _aggregate("ALL", all_picks, total_no_data)

    since_date = ""
    if first:
        since_date = min(fd for (fd, _p, _n) in first.values()).isoformat()

    return Scorecard(
        as_of=today.isoformat(),
        per_strategy=per_strategy,
        overall=overall,
        n_tickers=len(unique_tickers),
        n_no_data=total_no_data,
        since_date=since_date,
    )
