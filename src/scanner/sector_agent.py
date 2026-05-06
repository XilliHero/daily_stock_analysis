# -*- coding: utf-8 -*-
"""
Sector Agent — sector-level context for screened candidates.

Evaluates sector ETF performance, relative strength, and rotation
patterns to determine which sectors are in favour. Adds sector
context to each candidate's assessment.

No LLM calls — uses ETF price data from yfinance.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import yfinance as yf

from src.scanner.screener_agent import StockSignals
from src.scanner.strategy_profiles import StrategyProfile

logger = logging.getLogger(__name__)

SECTOR_ETFS: Dict[str, str] = {
    "Technology": "XLK",
    "Information Technology": "XLK",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Communication Services": "XLC",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

BROAD_MARKET_ETF = "SPY"


@dataclass
class SectorMetrics:
    """Performance metrics for a single sector."""

    sector: str
    etf: str
    current_price: float = 0.0
    change_1w: float = 0.0
    change_1m: float = 0.0
    relative_strength_1m: float = 0.0
    momentum_score: float = 0.0
    trend: str = "neutral"  # bullish / neutral / bearish
    rank: int = 0


@dataclass
class SectorAnalysis:
    """Sector context for one candidate stock."""

    ticker: str
    sector: str
    sector_trend: str = "neutral"
    sector_rank: int = 0
    sector_relative_strength: float = 0.0
    sector_momentum: float = 0.0
    sector_score: float = 0.0


@dataclass
class SectorResult:
    """Output of the Sector Agent."""

    sector_rankings: List[SectorMetrics] = field(default_factory=list)
    stock_analyses: List[SectorAnalysis] = field(default_factory=list)
    duration_s: float = 0.0
    errors: List[str] = field(default_factory=list)


class SectorAgent:
    """Sector-level performance and rotation analysis."""

    def __init__(self, strategy: StrategyProfile):
        self.strategy = strategy

    def run(self, candidates: List[StockSignals]) -> SectorResult:
        t0 = time.time()
        result = SectorResult()

        sector_metrics = self._analyze_sectors()
        result.sector_rankings = sorted(
            sector_metrics.values(), key=lambda s: s.momentum_score, reverse=True
        )

        for i, sm in enumerate(result.sector_rankings):
            sm.rank = i + 1

        metrics_by_sector = {sm.sector: sm for sm in result.sector_rankings}

        for sig in candidates:
            sa = self._assess_stock(sig, metrics_by_sector)
            result.stock_analyses.append(sa)

        result.duration_s = round(time.time() - t0, 2)
        logger.info(
            "[SectorAgent] done: %d sectors ranked, %d stocks assessed in %.1fs",
            len(result.sector_rankings), len(result.stock_analyses), result.duration_s,
        )
        return result

    def _analyze_sectors(self) -> Dict[str, SectorMetrics]:
        unique_etfs = sorted(set(SECTOR_ETFS.values()))
        etf_list = unique_etfs + [BROAD_MARKET_ETF]
        metrics: Dict[str, SectorMetrics] = {}
        etf_metrics: Dict[str, SectorMetrics] = {}

        try:
            data = yf.download(
                etf_list,
                period="35d",
                group_by="ticker",
                progress=False,
                threads=True,
            )
        except Exception as e:
            logger.warning("[SectorAgent] ETF download failed: %s", e)
            return metrics

        spy_returns = self._get_returns(data, BROAD_MARKET_ETF, len(etf_list))

        for etf in unique_etfs:
            try:
                sector = next(k for k, v in SECTOR_ETFS.items() if v == etf)
                sm = SectorMetrics(sector=sector, etf=etf)

                if len(etf_list) == 1:
                    df = data
                else:
                    if etf not in data.columns.get_level_values(0):
                        continue
                    df = data[etf]

                if df.empty or len(df) < 5:
                    continue

                close = df["Close"].dropna().values
                if len(close) < 5:
                    continue

                sm.current_price = float(close[-1])

                if len(close) >= 5:
                    sm.change_1w = (float(close[-1]) - float(close[-5])) / float(close[-5]) * 100

                if len(close) >= 21:
                    sm.change_1m = (float(close[-1]) - float(close[-21])) / float(close[-21]) * 100

                if spy_returns["1m"] != 0:
                    sm.relative_strength_1m = sm.change_1m - spy_returns["1m"]

                sm.momentum_score = self._compute_momentum(close)
                sm.trend = self._determine_trend(sm)

                etf_metrics[etf] = sm
            except Exception as e:
                logger.debug("[SectorAgent] %s failed: %s", etf, e)

        for sector_name, etf in SECTOR_ETFS.items():
            if etf in etf_metrics:
                metrics[sector_name] = etf_metrics[etf]

        return metrics

    def _get_returns(self, data: Any, ticker: str, total_tickers: int) -> Dict[str, float]:
        returns = {"1w": 0.0, "1m": 0.0}
        try:
            if total_tickers == 1:
                df = data
            else:
                df = data[ticker]
            close = df["Close"].dropna().values
            if len(close) >= 5:
                returns["1w"] = (float(close[-1]) - float(close[-5])) / float(close[-5]) * 100
            if len(close) >= 21:
                returns["1m"] = (float(close[-1]) - float(close[-21])) / float(close[-21]) * 100
        except Exception:
            pass
        return returns

    @staticmethod
    def _compute_momentum(close: np.ndarray) -> float:
        if len(close) < 10:
            return 0.0

        ma5 = float(np.mean(close[-5:]))
        ma10 = float(np.mean(close[-10:]))
        ma20 = float(np.mean(close[-20:])) if len(close) >= 20 else ma10

        score = 0.0
        current = float(close[-1])

        if current > ma5:
            score += 20.0
        if current > ma10:
            score += 20.0
        if current > ma20:
            score += 20.0
        if ma5 > ma10:
            score += 20.0
        if ma10 > ma20:
            score += 20.0

        return score

    @staticmethod
    def _determine_trend(sm: SectorMetrics) -> str:
        if sm.momentum_score >= 70 and sm.change_1w > 0:
            return "bullish"
        if sm.momentum_score <= 30 and sm.change_1w < 0:
            return "bearish"
        return "neutral"

    def _assess_stock(
        self, sig: StockSignals, metrics_by_sector: Dict[str, SectorMetrics]
    ) -> SectorAnalysis:
        sa = SectorAnalysis(ticker=sig.ticker, sector=sig.sector)

        sm = metrics_by_sector.get(sig.sector)
        if sm is None:
            sa.sector_score = 50.0
            return sa

        sa.sector_trend = sm.trend
        sa.sector_rank = sm.rank
        sa.sector_relative_strength = sm.relative_strength_1m
        sa.sector_momentum = sm.momentum_score

        score = 50.0

        if sm.trend == "bullish":
            score += 15.0
        elif sm.trend == "bearish":
            score -= 15.0

        if sm.rank <= 3:
            score += 10.0
        elif sm.rank >= 9:
            score -= 10.0

        score += sm.relative_strength_1m * 0.5

        preferred = self.strategy.preferred_sectors
        avoided = self.strategy.avoided_sectors
        if preferred and sig.sector in preferred:
            score += 10.0
        if avoided and sig.sector in avoided:
            score -= 10.0

        sa.sector_score = round(max(0.0, min(100.0, score)), 2)
        return sa
