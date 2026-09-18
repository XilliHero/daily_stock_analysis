# -*- coding: utf-8 -*-
"""Equity buy candidates, merged from the overlap board + watchlist + deep-dive.

All external sources are injectable so the merge/rank logic is testable offline.
Ranking: higher strategy_count first (conviction), then board order, watchlist last.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from src.advisor.classify import classify_asset


@dataclass
class Candidate:
    ticker: str
    name: str = ""
    strategy_count: int = 0
    sector: str = "Unknown"
    verdict: str = ""
    sentiment: str = ""


def _default_board():
    from src.scanner.overlap import build_overlap_board
    return build_overlap_board()


def _default_watchlist():
    from src.scanner.watchlist import get_watchlist_tickers  # parses STOCK_LIST -> list[str]
    return get_watchlist_tickers()


def fetch_candidates(limit: int = 5,
                     board_provider: Optional[Callable[[], object]] = None,
                     watchlist_provider: Optional[Callable[[], List[str]]] = None,
                     verdict_provider: Optional[Callable[[str], Optional[dict]]] = None) -> List[Candidate]:
    if limit <= 0:
        return []
    board = (board_provider or _default_board)()
    stocks = list(getattr(board, "stocks", []) or [])
    # Sort a copy by conviction desc, preserving original order within a tier.
    ranked = sorted(enumerate(stocks), key=lambda p: (-int(getattr(p[1], "strategy_count", 0)), p[0]))

    out: List[Candidate] = []
    seen = set()

    def _add(ticker, name="", strategy_count=0, sector="Unknown"):
        key = (ticker or "").strip().upper()
        if not key or key in seen or classify_asset(key) != "equity":
            return
        verdict = sentiment = ""
        if verdict_provider:
            v = verdict_provider(key) or {}
            verdict, sentiment = v.get("verdict", ""), v.get("sentiment", "")
        out.append(Candidate(ticker=key, name=name, strategy_count=strategy_count,
                             sector=sector, verdict=verdict, sentiment=sentiment))
        seen.add(key)

    for _, s in ranked:
        _add(getattr(s, "ticker", ""), getattr(s, "name", ""),
             int(getattr(s, "strategy_count", 0)), getattr(s, "sector", "Unknown"))

    for sym in (watchlist_provider or _default_watchlist)() or []:
        _add(sym)

    return out[:limit]
