# -*- coding: utf-8 -*-
"""
Fundamental Agent — deep financial analysis for screened candidates.

Pulls quarterly/annual financials from yfinance for each candidate,
evaluates earnings quality, balance-sheet health, and growth trends,
then produces a fundamental score per stock.

No LLM calls — pure financial data processing.
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


@dataclass
class FundamentalData:
    """Fundamental metrics for one stock."""

    ticker: str
    name: str = ""

    # Profitability
    revenue_ttm: float = 0.0
    net_income_ttm: float = 0.0
    gross_margin: float = 0.0
    operating_margin: float = 0.0
    net_margin: float = 0.0
    roe: float = 0.0
    roa: float = 0.0

    # Growth (YoY)
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None

    # Balance sheet
    total_debt: float = 0.0
    total_cash: float = 0.0
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    book_value_per_share: Optional[float] = None

    # Cash flow
    free_cash_flow: float = 0.0
    operating_cash_flow: float = 0.0

    # Valuation (carried from screener + enriched)
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    market_cap: float = 0.0

    # Dividend history
    dividend_years: int = 0

    # Scoring
    score: float = 0.0
    grade: str = ""  # A/B/C/D/F
    flags: List[str] = field(default_factory=list)


@dataclass
class FundamentalResult:
    """Output of the Fundamental Agent."""

    analyses: List[FundamentalData] = field(default_factory=list)
    total_analyzed: int = 0
    duration_s: float = 0.0
    errors: List[str] = field(default_factory=list)


class FundamentalAgent:
    """Deep financial analysis for screened candidates."""

    def __init__(self, strategy: StrategyProfile):
        self.strategy = strategy
        self.criteria = strategy.screening

    def run(self, candidates: List[StockSignals]) -> FundamentalResult:
        t0 = time.time()
        result = FundamentalResult(total_analyzed=len(candidates))

        for sig in candidates:
            try:
                data = self._analyze(sig)
                if data:
                    result.analyses.append(data)
            except Exception as e:
                result.errors.append(f"{sig.ticker}: {e}")
                logger.debug("[FundamentalAgent] %s failed: %s", sig.ticker, e)

        result.analyses.sort(key=lambda d: d.score, reverse=True)
        result.duration_s = round(time.time() - t0, 2)

        logger.info(
            "[FundamentalAgent] done: %d analyzed, %d succeeded in %.1fs",
            result.total_analyzed, len(result.analyses), result.duration_s,
        )
        return result

    def _analyze(self, sig: StockSignals) -> Optional[FundamentalData]:
        ticker_obj = yf.Ticker(sig.ticker)
        info = ticker_obj.info or {}

        fd = FundamentalData(
            ticker=sig.ticker,
            name=sig.name or info.get("shortName", ""),
        )

        self._extract_profitability(fd, info, ticker_obj)
        self._extract_balance_sheet(fd, info, ticker_obj)
        self._extract_cash_flow(fd, info, ticker_obj)
        self._extract_valuation(fd, info, sig)
        self._extract_growth(fd, info, ticker_obj)
        self._extract_dividend_history(fd, info, ticker_obj)

        fd.score = self._compute_score(fd)
        fd.grade = self._assign_grade(fd.score)

        return fd

    def _extract_profitability(
        self, fd: FundamentalData, info: Dict[str, Any], ticker: yf.Ticker
    ) -> None:
        fd.revenue_ttm = float(info.get("totalRevenue", 0) or 0)
        fd.net_income_ttm = float(info.get("netIncomeToCommon", 0) or 0)
        fd.gross_margin = float(info.get("grossMargins", 0) or 0) * 100
        fd.operating_margin = float(info.get("operatingMargins", 0) or 0) * 100
        fd.net_margin = float(info.get("profitMargins", 0) or 0) * 100
        fd.roe = float(info.get("returnOnEquity", 0) or 0) * 100
        fd.roa = float(info.get("returnOnAssets", 0) or 0) * 100

        if fd.net_margin > 15:
            fd.flags.append("high_margin")
        if fd.roe > 20:
            fd.flags.append("high_roe")
        if fd.net_margin < 0:
            fd.flags.append("negative_margin")

    def _extract_balance_sheet(
        self, fd: FundamentalData, info: Dict[str, Any], ticker: yf.Ticker
    ) -> None:
        fd.total_debt = float(info.get("totalDebt", 0) or 0)
        fd.total_cash = float(info.get("totalCash", 0) or 0)

        de = info.get("debtToEquity")
        if de is not None:
            fd.debt_to_equity = float(de) / 100
        cr = info.get("currentRatio")
        if cr is not None:
            fd.current_ratio = float(cr)
        bvps = info.get("bookValue")
        if bvps is not None:
            fd.book_value_per_share = float(bvps)

        if fd.debt_to_equity is not None and fd.debt_to_equity > 2.0:
            fd.flags.append("high_debt")
        if fd.current_ratio is not None and fd.current_ratio < 1.0:
            fd.flags.append("low_liquidity")
        if fd.total_cash > fd.total_debt:
            fd.flags.append("net_cash")

    def _extract_cash_flow(
        self, fd: FundamentalData, info: Dict[str, Any], ticker: yf.Ticker
    ) -> None:
        fd.free_cash_flow = float(info.get("freeCashflow", 0) or 0)
        fd.operating_cash_flow = float(info.get("operatingCashflow", 0) or 0)

        if fd.free_cash_flow > 0:
            fd.flags.append("positive_fcf")
        if fd.free_cash_flow < 0:
            fd.flags.append("negative_fcf")

    def _extract_valuation(
        self, fd: FundamentalData, info: Dict[str, Any], sig: StockSignals
    ) -> None:
        fd.pe_ratio = sig.pe_ratio or info.get("trailingPE")
        fd.forward_pe = info.get("forwardPE")
        fd.pb_ratio = sig.pb_ratio or info.get("priceToBook")
        fd.ps_ratio = info.get("priceToSalesTrailing12Months")
        fd.peg_ratio = info.get("pegRatio")
        fd.ev_to_ebitda = info.get("enterpriseToEbitda")
        fd.dividend_yield = sig.dividend_yield
        fd.payout_ratio = info.get("payoutRatio")
        if fd.payout_ratio is not None:
            fd.payout_ratio = float(fd.payout_ratio) * 100
        fd.market_cap = sig.market_cap or float(info.get("marketCap", 0) or 0)

    def _extract_growth(
        self, fd: FundamentalData, info: Dict[str, Any], ticker: yf.Ticker
    ) -> None:
        rg = info.get("revenueGrowth")
        if rg is not None:
            fd.revenue_growth = float(rg) * 100
        eg = info.get("earningsGrowth")
        if eg is not None:
            fd.earnings_growth = float(eg) * 100

        if fd.revenue_growth is not None and fd.revenue_growth > 15:
            fd.flags.append("strong_revenue_growth")
        if fd.earnings_growth is not None and fd.earnings_growth > 20:
            fd.flags.append("strong_earnings_growth")

    def _extract_dividend_history(
        self, fd: FundamentalData, info: Dict[str, Any], ticker: yf.Ticker
    ) -> None:
        try:
            dividends = ticker.dividends
            if dividends is not None and len(dividends) > 0:
                years_with_div = dividends.index.year.unique()
                fd.dividend_years = len(years_with_div)
                if fd.dividend_years >= 10:
                    fd.flags.append("dividend_veteran")
                elif fd.dividend_years >= 5:
                    fd.flags.append("dividend_consistent")
        except Exception:
            pass

    def _compute_score(self, fd: FundamentalData) -> float:
        score = 50.0
        strategy = self.strategy.name

        # Profitability (all strategies care, but weight differs)
        if fd.roe > 15:
            score += 8.0
        elif fd.roe > 10:
            score += 4.0
        if fd.net_margin > 10:
            score += 5.0

        # Balance sheet health
        if "net_cash" in fd.flags:
            score += 6.0
        if "high_debt" in fd.flags:
            score -= 8.0
        if "low_liquidity" in fd.flags:
            score -= 5.0

        # Cash flow
        if "positive_fcf" in fd.flags:
            score += 5.0
        if "negative_fcf" in fd.flags:
            score -= 6.0

        # Strategy-specific scoring
        if strategy == "value":
            if fd.pe_ratio and 0 < fd.pe_ratio < 15:
                score += 10.0
            elif fd.pe_ratio and 0 < fd.pe_ratio < 20:
                score += 5.0
            if fd.pb_ratio and 0 < fd.pb_ratio < 1.5:
                score += 8.0
            if fd.ev_to_ebitda and 0 < fd.ev_to_ebitda < 10:
                score += 6.0

        elif strategy == "growth":
            if "strong_revenue_growth" in fd.flags:
                score += 12.0
            if "strong_earnings_growth" in fd.flags:
                score += 10.0
            if fd.peg_ratio and 0 < fd.peg_ratio < 1.5:
                score += 8.0
            if fd.gross_margin > 50:
                score += 5.0

        elif strategy == "dividend":
            if fd.dividend_yield and fd.dividend_yield > 3.0:
                score += 10.0
            elif fd.dividend_yield and fd.dividend_yield > 2.0:
                score += 5.0
            if fd.payout_ratio and 30 < fd.payout_ratio < 70:
                score += 8.0
            elif fd.payout_ratio and fd.payout_ratio > 90:
                score -= 5.0
            if "dividend_veteran" in fd.flags:
                score += 10.0
            elif "dividend_consistent" in fd.flags:
                score += 5.0

        elif strategy == "recovery":
            if "positive_fcf" in fd.flags:
                score += 8.0
            if fd.current_ratio and fd.current_ratio > 1.5:
                score += 6.0
            if fd.debt_to_equity and fd.debt_to_equity < 1.0:
                score += 6.0
            if "strong_earnings_growth" in fd.flags:
                score += 8.0

        return round(max(0.0, min(100.0, score)), 2)

    @staticmethod
    def _assign_grade(score: float) -> str:
        if score >= 80:
            return "A"
        if score >= 65:
            return "B"
        if score >= 50:
            return "C"
        if score >= 35:
            return "D"
        return "F"
