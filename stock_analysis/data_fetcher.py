"""
Fetches price history, current quotes, and recent news via yfinance.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from stock_analysis.config import BENCHMARK, LOOKBACK_DAYS, WATCHLIST

logger = logging.getLogger(__name__)


def _download(tickers: list[str], days: int) -> pd.DataFrame:
    """Download adjusted close prices, returning a clean DataFrame."""
    end = datetime.today()
    # Extra buffer so we always end up with ≥LOOKBACK_DAYS trading days
    start = end - timedelta(days=days + 45)

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {tickers}")

    # Multi-ticker download → MultiIndex columns; single ticker → flat columns
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Drop rows where every ticker is NaN, then keep the last `days` rows
    closes = closes.dropna(how="all")
    return closes.tail(days)


def fetch_price_history(
    watchlist: list[str] = WATCHLIST,
    benchmark: str = BENCHMARK,
    days: int = LOOKBACK_DAYS,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
        stock_prices  – DataFrame, columns = watchlist tickers, index = dates
        market_prices – Series,    ^GSPC closing prices aligned to same dates
    """
    all_tickers = watchlist + [benchmark]
    prices = _download(all_tickers, days)

    # Split out benchmark
    if benchmark in prices.columns:
        market_prices = prices[benchmark].rename(benchmark)
        stock_prices = prices.drop(columns=[benchmark])
    else:
        raise RuntimeError(f"Benchmark {benchmark} not found in downloaded data.")

    # Forward-fill small gaps (e.g. MDA.TO holiday vs US market)
    stock_prices = stock_prices.ffill().dropna(how="all")
    market_prices = market_prices.ffill().dropna()

    # Align on common dates
    common_idx = stock_prices.index.intersection(market_prices.index)
    stock_prices = stock_prices.loc[common_idx]
    market_prices = market_prices.loc[common_idx]

    logger.info(
        "Fetched %d trading days for %d tickers + benchmark.",
        len(common_idx),
        len(watchlist),
    )
    return stock_prices, market_prices


def fetch_current_quotes(watchlist: list[str] = WATCHLIST) -> dict[str, dict]:
    """
    Returns a dict: ticker → {price, prev_close, change_pct, volume}.
    Falls back gracefully on individual ticker failures.
    """
    quotes: dict[str, dict] = {}
    for ticker in watchlist:
        try:
            t = yf.Ticker(ticker)
            fi = t.fast_info
            price = fi.last_price
            prev = fi.previous_close
            change_pct = ((price - prev) / prev * 100) if prev else 0.0
            quotes[ticker] = {
                "price": price,
                "prev_close": prev,
                "change_pct": change_pct,
                "day_high": getattr(fi, "day_high", None),
                "day_low": getattr(fi, "day_low", None),
                "volume": getattr(fi, "last_volume", None),
                "fifty_two_week_high": getattr(fi, "year_high", None),
                "fifty_two_week_low": getattr(fi, "year_low", None),
            }
        except Exception as exc:
            logger.warning("Could not fetch quote for %s: %s", ticker, exc)
            quotes[ticker] = {
                "price": None,
                "prev_close": None,
                "change_pct": None,
                "day_high": None,
                "day_low": None,
                "volume": None,
                "fifty_two_week_high": None,
                "fifty_two_week_low": None,
            }
    return quotes


def fetch_news(
    watchlist: list[str] = WATCHLIST,
    max_per_ticker: int = 3,
    hours: int = 48,
) -> dict[str, list[dict]]:
    """
    Fetch recent news headlines for each ticker via yfinance.

    Returns
    -------
    dict mapping ticker → list of {title, url, publisher, published_at}
    Items are limited to the last `hours` hours, up to `max_per_ticker` each.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    result: dict[str, list[dict]] = {}

    for ticker in watchlist:
        try:
            t = yf.Ticker(ticker)
            raw_news = t.news or []
            items: list[dict] = []

            for item in raw_news[:15]:           # inspect first 15 to find fresh ones
                # yfinance v0.2+ nests content differently; handle both layouts
                content = item.get("content") or {}
                ts = (
                    content.get("pubDate")
                    or item.get("providerPublishTime")
                )
                if isinstance(ts, int):
                    pub = datetime.fromtimestamp(ts, tz=timezone.utc)
                elif isinstance(ts, str):
                    try:
                        pub = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except Exception:
                        pub = datetime.now(timezone.utc)
                else:
                    pub = datetime.now(timezone.utc)

                if pub < cutoff:
                    continue                      # too old

                title = (
                    content.get("title")
                    or item.get("title", "")
                )
                url = (
                    (content.get("canonicalUrl") or {}).get("url")
                    or item.get("link", "")
                )
                publisher = (
                    (content.get("provider") or {}).get("displayName")
                    or item.get("publisher", "")
                )

                items.append(
                    {
                        "title": title,
                        "url": url,
                        "publisher": publisher,
                        "published_at": pub,
                    }
                )
                if len(items) >= max_per_ticker:
                    break

            result[ticker] = items
            logger.debug("Fetched %d news items for %s", len(items), ticker)

        except Exception as exc:
            logger.warning("Could not fetch news for %s: %s", ticker, exc)
            result[ticker] = []

    return result
