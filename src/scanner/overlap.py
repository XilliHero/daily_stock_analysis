# -*- coding: utf-8 -*-
"""High-Conviction Overlap Board — stocks flagged by multiple strategies today.

When the same stock surfaces in several strategies at once (e.g. LNC in value +
growth + dividend + recovery), that's a much stronger signal than a single-list
appearance. This surfaces those overlaps from the most recent scan, ranked by
how many strategies agree.

Pure set math over stored picks — no network, no LLM calls. Reuses the grouping
helpers from the What-Changed detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.scanner.changes import _all_dates, _group, _top_n
from src.scanner.scorecard.pick_store import PickRecord, load_all_picks

DEFAULT_TOP_N = 20
DEFAULT_MIN_STRATEGIES = 2


@dataclass
class OverlapStock:
    ticker: str
    name: str
    strategy_count: int
    strategies: List[str]
    best_rank: int      # best (lowest) rank across the strategies it appears in
    best_score: float


@dataclass
class OverlapBoard:
    scan_date: str
    stocks: List[OverlapStock] = field(default_factory=list)

    @property
    def has_any(self) -> bool:
        return bool(self.stocks)


def build_overlap_board(
    records: Optional[List[PickRecord]] = None,
    top_n: int = DEFAULT_TOP_N,
    min_strategies: int = DEFAULT_MIN_STRATEGIES,
) -> Optional[OverlapBoard]:
    """Find stocks in >= ``min_strategies`` of the latest scan's per-strategy top-N."""
    if records is None:
        records = load_all_picks()
    group = _group(records)
    dates = _all_dates(group)
    if not dates:
        return None
    latest = dates[-1]

    agg: dict = {}
    for strat, by_date in group.items():
        day = by_date.get(latest)
        if not day:
            continue
        for ticker, rank in _top_n(day, top_n).items():
            rec = day[ticker]
            score = rec.score if rec.score is not None else 0.0
            entry = agg.get(ticker)
            if entry is None:
                agg[ticker] = {
                    "name": rec.name,
                    "strategies": [strat],
                    "best_rank": rank,
                    "best_score": score,
                }
            else:
                entry["strategies"].append(strat)
                entry["best_rank"] = min(entry["best_rank"], rank)
                entry["best_score"] = max(entry["best_score"], score)

    stocks = [
        OverlapStock(
            ticker=ticker,
            name=data["name"],
            strategy_count=len(data["strategies"]),
            strategies=sorted(data["strategies"]),
            best_rank=data["best_rank"],
            best_score=data["best_score"],
        )
        for ticker, data in agg.items()
        if len(data["strategies"]) >= min_strategies
    ]
    # Most strategies first, then best rank, then highest score.
    stocks.sort(key=lambda s: (-s.strategy_count, s.best_rank, -s.best_score))

    return OverlapBoard(scan_date=latest, stocks=stocks)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

_MAX_ROWS = 15


def render_overlap_markdown(board: OverlapBoard) -> str:
    """Return the ``## 🎯 High-Conviction Board`` markdown block."""
    lines: List[str] = []
    lines.append("## 🎯 High-Conviction Board")
    lines.append(
        f"Stocks flagged by multiple strategies in today's scan ({board.scan_date}) — "
        "the more strategies agree, the stronger the signal."
    )

    if not board.has_any:
        lines.append("")
        lines.append("_No multi-strategy overlaps in today's scan._")
        return "\n".join(lines)

    lines.append("")
    lines.append("| Ticker | Name | Agree | Strategies | Best Rank |")
    lines.append("|--------|------|-------|------------|-----------|")
    for s in board.stocks[:_MAX_ROWS]:
        fire = " 🔥" if s.strategy_count >= 3 else ""
        lines.append(
            f"| **{s.ticker}**{fire} | {s.name} | {s.strategy_count}/4 "
            f"| {', '.join(s.strategies)} | #{s.best_rank} |"
        )

    extra = len(board.stocks) - _MAX_ROWS
    if extra > 0:
        lines.append(f"\n_+{extra} more with 2+ strategy overlap._")

    return "\n".join(lines)
