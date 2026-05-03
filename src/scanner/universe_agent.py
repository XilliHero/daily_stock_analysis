# -*- coding: utf-8 -*-
"""
Universe Agent — builds and caches the stock universe for scanning.

Primary source: static CSV files shipped with the repo (data/scanner/).
Optional refresh: pulls live data from Wikipedia to update the CSVs.

Covers US (S&P 500, S&P 400, S&P 600) and Canadian (TSX Composite).
"""

from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "scanner"

CSV_FILES = {
    "sp500": {"file": "sp500.csv", "cap_tier": "large", "exchange": "US", "country": "US"},
    "sp400": {"file": "sp400.csv", "cap_tier": "mid", "exchange": "US", "country": "US"},
    "sp600": {"file": "sp600.csv", "cap_tier": "small", "exchange": "US", "country": "US"},
    "tsx": {"file": "tsx.csv", "cap_tier": "large", "exchange": "TSX", "country": "CA"},
}

VALID_SECTORS = {
    "Technology", "Healthcare", "Financials", "Energy",
    "Consumer Cyclical", "Consumer Defensive", "Industrials",
    "Basic Materials", "Communication Services", "Real Estate", "Utilities",
    "Information Technology",
}

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
    """Builds the stock universe from static CSV files."""

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
            us_stocks, us_errors = self._load_from_csv("us")
            all_tickers.update(us_stocks)
            result.errors.extend(us_errors)
            result.regions_loaded.append("us")

        if self.regions in ("ca", "us_ca"):
            ca_stocks, ca_errors = self._load_from_csv("ca")
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

    def _load_from_csv(self, region: str) -> tuple[Dict[str, StockEntry], List[str]]:
        stocks: Dict[str, StockEntry] = {}
        errors: List[str] = []

        targets = []
        if region == "us":
            targets = ["sp500", "sp400", "sp600"]
        elif region == "ca":
            targets = ["tsx"]

        for key in targets:
            meta = CSV_FILES[key]
            csv_path = DATA_DIR / meta["file"]

            if not csv_path.exists():
                errors.append(f"{key}: CSV not found at {csv_path}. Run --refresh-universe to create it.")
                continue

            try:
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    ticker = str(row.get("Symbol", "")).strip()
                    if not ticker:
                        continue
                    stocks[ticker] = StockEntry(
                        ticker=ticker,
                        name=str(row.get("Name", "")).strip(),
                        sector=str(row.get("Sector", "")).strip(),
                        industry=str(row.get("Industry", "")).strip(),
                        exchange=meta["exchange"],
                        country=meta["country"],
                        cap_tier=meta["cap_tier"],
                    )
                logger.info("[UniverseAgent] %s: loaded %d tickers from CSV", key, len(df))
            except Exception as e:
                errors.append(f"{key}: failed to read CSV: {e}")
                logger.warning("[UniverseAgent] failed to load %s: %s", key, e)

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

    # ------------------------------------------------------------------
    # Wikipedia refresh (optional, called via --refresh-universe)
    # ------------------------------------------------------------------

    @classmethod
    def refresh_from_wikipedia(cls) -> Dict[str, int]:
        """Fetch latest constituents from Wikipedia and save to CSV files."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        counts: Dict[str, int] = {}

        us_sources = {
            "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
            "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
        }

        for key, url in us_sources.items():
            try:
                tables = cls._fetch_tables(url)
                if not tables:
                    logger.warning("[Refresh] %s: no tables found", key)
                    continue

                df = tables[0]
                symbol_col = cls._find_column(df, ["Symbol", "Ticker", "Ticker symbol"])
                name_col = cls._find_column(df, ["Security", "Company", "Name"])
                sector_col = cls._find_column(df, ["GICS Sector", "Sector", "GICS sector"])
                industry_col = cls._find_column(df, ["GICS Sub-Industry", "Sub-Industry", "GICS sub-industry", "Industry"])

                if symbol_col is None:
                    logger.warning("[Refresh] %s: no symbol column found", key)
                    continue

                rows = []
                for _, row in df.iterrows():
                    ticker = str(row.get(symbol_col, "")).strip().replace(".", "-")
                    if not ticker:
                        continue
                    rows.append({
                        "Symbol": ticker,
                        "Name": str(row.get(name_col, "")).strip() if name_col else "",
                        "Sector": str(row.get(sector_col, "")).strip() if sector_col else "",
                        "Industry": str(row.get(industry_col, "")).strip() if industry_col else "",
                    })

                out_df = pd.DataFrame(rows)
                out_path = DATA_DIR / CSV_FILES[key]["file"]
                out_df.to_csv(out_path, index=False)
                counts[key] = len(rows)
                logger.info("[Refresh] %s: saved %d tickers to %s", key, len(rows), out_path)
            except Exception as e:
                logger.warning("[Refresh] %s failed: %s", key, e)

        # TSX Composite
        try:
            tsx_url = "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index"
            tables = cls._fetch_tables(tsx_url)
            df = None
            for table in tables:
                cols_lower = [str(c).lower() for c in table.columns]
                if any("symbol" in c or "ticker" in c for c in cols_lower):
                    df = table
                    break

            if df is not None:
                symbol_col = cls._find_column(df, ["Symbol", "Ticker", "Ticker symbol"])
                name_col = cls._find_column(df, ["Company", "Name", "Security"])
                sector_col = cls._find_column(df, ["Sector", "GICS Sector", "GICS sector"])

                if symbol_col:
                    rows = []
                    for _, row in df.iterrows():
                        raw = str(row.get(symbol_col, "")).strip()
                        if not raw:
                            continue
                        ticker = raw if ".TO" in raw else f"{raw}.TO"
                        ticker = ticker.replace(".", "-", ticker.count(".") - 1)
                        rows.append({
                            "Symbol": ticker,
                            "Name": str(row.get(name_col, "")).strip() if name_col else "",
                            "Sector": str(row.get(sector_col, "")).strip() if sector_col else "",
                            "Industry": "",
                        })

                    out_df = pd.DataFrame(rows)
                    out_path = DATA_DIR / CSV_FILES["tsx"]["file"]
                    out_df.to_csv(out_path, index=False)
                    counts["tsx"] = len(rows)
                    logger.info("[Refresh] tsx: saved %d tickers to %s", len(rows), out_path)
        except Exception as e:
            logger.warning("[Refresh] TSX failed: %s", e)

        return counts

    @staticmethod
    def _fetch_tables(url: str) -> List[pd.DataFrame]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/125.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return pd.read_html(StringIO(resp.text))

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
