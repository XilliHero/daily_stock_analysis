# -*- coding: utf-8 -*-
"""
Universe Agent — builds and caches the stock universe for scanning.

Pulls index constituents for US (S&P 500, S&P 400, S&P 600) and
Canadian (TSX Composite) markets via yfinance/Wikipedia, applies
sector and market-cap tier filters, and returns a clean ticker list.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

# Sector filters use GICS sector names as returned by yfinance
VALID_SECTORS = {
    "Technology", "Healthcare", "Financials", "Energy",
    "Consumer Cyclical", "Consumer Defensive", "Industrials",
    "Basic Materials", "Communication Services", "Real Estate", "Utilities",
}

# Market-cap tiers (approximate USD boundaries)
CAP_TIERS = {
    "large": (10_000_000_000, float("inf")),
    "mid": (2_000_000_000, 10_000_000_000),
    "small": (300_000_000, 2_000_000_000),
}


@dataclass
class StockEntry:
    """A single stock in the universe."""

    ticker: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    market_cap: float = 0.0
    exchange: str = ""
    country: str = ""
    cap_tier: str = ""


@dataclass
class UniverseResult:
    """Output of the Universe Agent."""

    stocks: List[StockEntry] = field(default_factory=list)
    total_before_filter: int = 0
    total_after_filter: int = 0
    regions_loaded: List[str] = field(default_factory=list)
    duration_s: float = 0.0
    errors: List[str] = field(default_factory=list)


class UniverseAgent:
    """Builds the stock universe from index constituents."""

    def __init__(
        self,
        regions: str = "us_ca",
        cap_tiers: Optional[List[str]] = None,
        include_sectors: Optional[List[str]] = None,
        exclude_sectors: Optional[List[str]] = None,
    ):
        self.regions = regions
        self.cap_tiers = cap_tiers or ["large", "mid", "small"]
        self.include_sectors = set(include_sectors) if include_sectors else None
        self.exclude_sectors = set(exclude_sectors) if exclude_sectors else set()

        self._cache: Optional[UniverseResult] = None
        self._cache_time: float = 0.0
        self._cache_ttl: float = 86400  # 24 hours

    def run(self, force_refresh: bool = False) -> UniverseResult:
        if self._cache and not force_refresh and (time.time() - self._cache_time < self._cache_ttl):
            logger.info("[UniverseAgent] returning cached universe (%d stocks)", len(self._cache.stocks))
            return self._cache

        t0 = time.time()
        result = UniverseResult()
        all_tickers: Dict[str, StockEntry] = {}

        if self.regions in ("us", "us_ca"):
            us_stocks, us_errors = self._load_us_universe()
            all_tickers.update(us_stocks)
            result.errors.extend(us_errors)
            result.regions_loaded.append("us")

        if self.regions in ("ca", "us_ca"):
            ca_stocks, ca_errors = self._load_ca_universe()
            all_tickers.update(ca_stocks)
            result.errors.extend(ca_errors)
            result.regions_loaded.append("ca")

        result.total_before_filter = len(all_tickers)
        filtered = self._apply_filters(list(all_tickers.values()))
        result.stocks = filtered
        result.total_after_filter = len(filtered)
        result.duration_s = round(time.time() - t0, 2)

        self._cache = result
        self._cache_time = time.time()

        logger.info(
            "[UniverseAgent] universe built: %d → %d stocks (regions=%s) in %.1fs",
            result.total_before_filter, result.total_after_filter,
            result.regions_loaded, result.duration_s,
        )
        return result

    def _load_us_universe(self) -> tuple[Dict[str, StockEntry], List[str]]:
        stocks: Dict[str, StockEntry] = {}
        errors: List[str] = []

        index_sources = {
            "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            "S&P 400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
            "S&P 600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
        }

        for index_name, url in index_sources.items():
            try:
                tables = pd.read_html(url)
                if not tables:
                    errors.append(f"{index_name}: no tables found")
                    continue

                df = tables[0]
                symbol_col = self._find_column(df, ["Symbol", "Ticker", "Ticker symbol"])
                name_col = self._find_column(df, ["Security", "Company", "Name"])
                sector_col = self._find_column(df, ["GICS Sector", "Sector", "GICS sector"])
                industry_col = self._find_column(df, ["GICS Sub-Industry", "Sub-Industry", "GICS sub-industry", "Industry"])

                if symbol_col is None:
                    errors.append(f"{index_name}: could not find symbol column")
                    continue

                cap_tier = "large" if "500" in index_name else "mid" if "400" in index_name else "small"

                for _, row in df.iterrows():
                    ticker = str(row.get(symbol_col, "")).strip().replace(".", "-")
                    if not ticker:
                        continue
                    stocks[ticker] = StockEntry(
                        ticker=ticker,
                        name=str(row.get(name_col, "")).strip() if name_col else "",
                        sector=str(row.get(sector_col, "")).strip() if sector_col else "",
                        industry=str(row.get(industry_col, "")).strip() if industry_col else "",
                        exchange="US",
                        country="US",
                        cap_tier=cap_tier,
                    )

                logger.info("[UniverseAgent] %s: loaded %d tickers", index_name, len(df))
            except Exception as e:
                errors.append(f"{index_name}: {e}")
                logger.warning("[UniverseAgent] failed to load %s: %s", index_name, e)

        return stocks, errors

    def _load_ca_universe(self) -> tuple[Dict[str, StockEntry], List[str]]:
        stocks: Dict[str, StockEntry] = {}
        errors: List[str] = []

        url = "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index"
        try:
            tables = pd.read_html(url)
            df = None
            for table in tables:
                cols_lower = [str(c).lower() for c in table.columns]
                if any("symbol" in c or "ticker" in c for c in cols_lower):
                    df = table
                    break

            if df is None:
                errors.append("TSX Composite: could not find constituents table")
                return stocks, errors

            symbol_col = self._find_column(df, ["Symbol", "Ticker", "Ticker symbol"])
            name_col = self._find_column(df, ["Company", "Name", "Security"])
            sector_col = self._find_column(df, ["Sector", "GICS Sector", "GICS sector"])

            if symbol_col is None:
                errors.append("TSX Composite: could not find symbol column")
                return stocks, errors

            for _, row in df.iterrows():
                raw_ticker = str(row.get(symbol_col, "")).strip()
                if not raw_ticker:
                    continue
                ticker = raw_ticker if ".TO" in raw_ticker else f"{raw_ticker}.TO"
                ticker = ticker.replace(".", "-", ticker.count(".") - 1)

                stocks[ticker] = StockEntry(
                    ticker=ticker,
                    name=str(row.get(name_col, "")).strip() if name_col else "",
                    sector=str(row.get(sector_col, "")).strip() if sector_col else "",
                    exchange="TSX",
                    country="CA",
                    cap_tier="large",
                )

            logger.info("[UniverseAgent] TSX Composite: loaded %d tickers", len(stocks))
        except Exception as e:
            errors.append(f"TSX Composite: {e}")
            logger.warning("[UniverseAgent] failed to load TSX Composite: %s", e)

        return stocks, errors

    def _apply_filters(self, stocks: List[StockEntry]) -> List[StockEntry]:
        filtered = []
        for s in stocks:
            if s.cap_tier and s.cap_tier not in self.cap_tiers:
                continue
            if self.include_sectors and s.sector not in self.include_sectors:
                continue
            if s.sector in self.exclude_sectors:
                continue
            filtered.append(s)
        return filtered

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        cols = {str(c).strip(): c for c in df.columns}
        for candidate in candidates:
            if candidate in cols:
                return cols[candidate]
            for col_name, col_actual in cols.items():
                if candidate.lower() == col_name.lower():
                    return col_actual
        return None
