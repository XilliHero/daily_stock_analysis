"""
Briefing Data Fetcher
=====================
Fetches data for the morning pre-market briefing:
  - Overnight futures / index quotes (S&P 500, Nasdaq, TSX)
  - Recent news headlines per watchlist ticker
  - Dynamically generated key watchpoints for the day
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import yfinance as yf

logger = logging.getLogger(__name__)

# Futures / index proxies used in the briefing
FUTURES: list[tuple[str, str]] = [
    ("S&P 500", "ES=F"),
    ("Nasdaq", "NQ=F"),
    ("TSX", "^GSPTSX"),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FutureQuote:
    name: str
    ticker: str
    price: float | None
    change: float | None       # absolute change from previous close
    change_pct: float | None   # percent change from previous close


@dataclass
class NewsItem:
    title: str
    publisher: str
    url: str


@dataclass
class BriefingData:
    futures: list[FutureQuote]
    news: dict[str, list[NewsItem]]   # ticker → top headlines
    watchpoints: list[str]            # 2–3 key things to watch today
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _fetch_future(name: str, ticker: str) -> FutureQuote:
    """Fetch a single futures / index quote via yfinance fast_info."""
    try:
        fi = yf.Ticker(ticker).fast_info
        price = fi.last_price
        prev = fi.previous_close
        change = (price - prev) if (price is not None and prev) else None
        change_pct = (change / prev * 100) if (prev and change is not None) else None
        return FutureQuote(
            name=name, ticker=ticker,
            price=price, change=change, change_pct=change_pct,
        )
    except Exception as exc:
        logger.warning("Could not fetch future %s (%s): %s", name, ticker, exc)
        return FutureQuote(name=name, ticker=ticker, price=None, change=None, change_pct=None)


def _fetch_news(ticker: str, max_items: int = 2) -> list[NewsItem]:
    """Return up to max_items recent news headlines for a ticker."""
    try:
        raw = yf.Ticker(ticker).news or []
        items: list[NewsItem] = []
        for article in raw[:max_items]:
            # yfinance ≥0.2.x wraps articles in a 'content' sub-dict
            content = article.get("content", {})
            title = content.get("title") or article.get("title", "")
            publisher = (
                content.get("provider", {}).get("displayName")
                or article.get("publisher", "")
            )
            url = (
                content.get("canonicalUrl", {}).get("url")
                or article.get("link", "")
            )
            if title:
                items.append(NewsItem(title=title, publisher=publisher, url=url))
        return items
    except Exception as exc:
        logger.warning("Could not fetch news for %s: %s", ticker, exc)
        return []


# ---------------------------------------------------------------------------
# Watchpoint generator
# ---------------------------------------------------------------------------

def _generate_watchpoints(
    futures: list[FutureQuote],
    news: dict[str, list[NewsItem]],
) -> list[str]:
    """Produce 2–3 concise, data-driven things to watch today."""
    points: list[str] = []

    # 1. Overall futures direction
    valid = [f for f in futures if f.change_pct is not None]
    up   = [f for f in valid if f.change_pct > 0]
    down = [f for f in valid if f.change_pct < 0]

    if len(up) >= 2:
        names = " and ".join(f.name for f in up[:2])
        pcts  = " / ".join(f"{f.change_pct:+.2f}%" for f in up[:2])
        points.append(f"Futures leaning bullish — {names} up {pcts} overnight.")
    elif len(down) >= 2:
        names = " and ".join(f.name for f in down[:2])
        pcts  = " / ".join(f"{f.change_pct:+.2f}%" for f in down[:2])
        points.append(f"Futures under pressure — {names} down {pcts} overnight.")
    elif valid:
        parts = ", ".join(f"{f.name} {f.change_pct:+.1f}%" for f in valid)
        points.append(f"Mixed overnight signals ({parts}) — expect a cautious open.")

    # 2. Biggest single futures / index mover
    movers = sorted(valid, key=lambda f: abs(f.change_pct), reverse=True)
    if movers and abs(movers[0].change_pct) >= 0.4:
        m = movers[0]
        points.append(
            f"{m.name} ({m.ticker}) shows the largest overnight swing at {m.change_pct:+.2f}% — "
            f"watch for follow-through at the open."
        )

    # 3. Stock with most pre-market headlines
    busiest = max(
        ((t, items) for t, items in news.items() if items),
        key=lambda kv: len(kv[1]),
        default=None,
    )
    if busiest:
        ticker, items = busiest
        points.append(
            f"{ticker} has the most pre-market headlines today — review before trading."
        )

    return points[:3]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_briefing_data(watchlist: list[str]) -> BriefingData:
    """Fetch all data required for the morning briefing email."""
    logger.info("Fetching futures quotes …")
    futures = [_fetch_future(name, ticker) for name, ticker in FUTURES]

    logger.info("Fetching news headlines for %s …", watchlist)
    news = {ticker: _fetch_news(ticker) for ticker in watchlist}

    watchpoints = _generate_watchpoints(futures, news)
    return BriefingData(futures=futures, news=news, watchpoints=watchpoints)
