# -*- coding: utf-8 -*-
"""
Screener Agent — fast first-pass to reduce the universe to candidates.

Fetches 30 days of daily OHLCV for every ticker via yfinance,
computes technical and valuation signals, scores each stock,
and returns a ranked shortlist.

No LLM calls — pure data processing.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.scanner.strategy_profiles import ScreeningCriteria, StrategyProfile
from src.scanner.universe_agent import StockEntry

logger = logging.getLogger(__name__)

BATCH_SIZE = 50
HISTORY_DAYS = 40  # fetch a few extra to cover weekends/holidays


@dataclass
class StockSignals:
    """Screening signals computed for one stock."""

    ticker: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    exchange: str = ""
    country: str = ""
    cap_tier: str = ""

    # Price data
    current_price: float = 0.0
    prev_close: float = 0.0
    high_52w: float = 0.0
    low_52w: float = 0.0
    change_pct: float = 0.0

    # Signals (True = triggered)
    volume_spike: bool = False
    volume_ratio: float = 0.0
    breakout: bool = False
    momentum_crossover: bool = False
    gap_up: bool = False
    gap_pct: float = 0.0
    rsi: float = 50.0
    rsi_signal: bool = False
    atr_pct: float = 0.0

    # Valuation (from yfinance info)
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    market_cap: float = 0.0

    # Drawdown from 52-week high
    drawdown_pct: float = 0.0

    # Scoring
    signals_triggered: int = 0
    score: float = 0.0
    signal_names: List[str] = field(default_factory=list)


@dataclass
class ScreenerResult:
    """Output of the Screener Agent."""

    shortlist: List[StockSignals] = field(default_factory=list)
    total_scanned: int = 0
    total_shortlisted: int = 0
    duration_s: float = 0.0
    errors: List[str] = field(default_factory=list)


class ScreenerAgent:
    """Fast technical + valuation screening pass."""

    def __init__(self, strategy: StrategyProfile):
        self.strategy = strategy
        self.criteria = strategy.screening

    def run(self, stocks: List[StockEntry], top_n: int = 50) -> ScreenerResult:
        t0 = time.time()
        result = ScreenerResult(total_scanned=len(stocks))

        tickers = [s.ticker for s in stocks]
        stock_map = {s.ticker: s for s in stocks}

        all_signals: List[StockSignals] = []

        for i in range(0, len(tickers), BATCH_SIZE):
            batch = tickers[i:i + BATCH_SIZE]
            batch_signals = self._screen_batch(batch, stock_map)
            all_signals.extend(batch_signals)
            logger.info(
                "[ScreenerAgent] batch %d-%d: %d/%d passed",
                i, min(i + BATCH_SIZE, len(tickers)),
                len(batch_signals), len(batch),
            )

        qualified = [s for s in all_signals if s.signals_triggered >= self.criteria.min_signals]
        qualified.sort(key=lambda s: s.score, reverse=True)
        result.shortlist = qualified[:top_n]
        result.total_shortlisted = len(result.shortlist)
        result.duration_s = round(time.time() - t0, 2)

        logger.info(
            "[ScreenerAgent] done: %d scanned → %d qualified → top %d in %.1fs",
            result.total_scanned, len(qualified), result.total_shortlisted, result.duration_s,
        )
        return result

    def _screen_batch(
        self, tickers: List[str], stock_map: Dict[str, StockEntry]
    ) -> List[StockSignals]:
        results: List[StockSignals] = []

        try:
            data = yf.download(
                tickers,
                period=f"{HISTORY_DAYS}d",
                group_by="ticker",
                progress=False,
                threads=True,
            )
        except Exception as e:
            logger.warning("[ScreenerAgent] yf.download failed: %s", e)
            return results

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df = data
                else:
                    if ticker not in data.columns.get_level_values(0):
                        continue
                    df = data[ticker]

                if df.empty or len(df) < 10:
                    continue

                entry = stock_map.get(ticker)
                signals = self._compute_signals(ticker, df, entry)
                if signals and signals.signals_triggered > 0:
                    results.append(signals)
            except Exception as e:
                logger.debug("[ScreenerAgent] %s failed: %s", ticker, e)

        return results

    def _compute_signals(
        self, ticker: str, df: pd.DataFrame, entry: Optional[StockEntry]
    ) -> Optional[StockSignals]:
        df = df.dropna(subset=["Close"])
        if len(df) < 10:
            return None

        close = df["Close"].values
        volume = df["Volume"].values
        high = df["High"].values
        low = df["Low"].values
        open_price = df["Open"].values

        current = float(close[-1])
        prev = float(close[-2]) if len(close) > 1 else current

        sig = StockSignals(
            ticker=ticker,
            name=entry.name if entry else "",
            sector=entry.sector if entry else "",
            industry=entry.industry if entry else "",
            exchange=entry.exchange if entry else "",
            country=entry.country if entry else "",
            cap_tier=entry.cap_tier if entry else "",
            current_price=current,
            prev_close=prev,
            change_pct=((current - prev) / prev * 100) if prev else 0.0,
        )

        # Volume spike
        avg_vol = float(np.mean(volume[-20:])) if len(volume) >= 20 else float(np.mean(volume))
        if avg_vol > 0:
            sig.volume_ratio = float(volume[-1]) / avg_vol
            if sig.volume_ratio >= self.criteria.volume_spike_min:
                sig.volume_spike = True
                sig.signal_names.append("volume_spike")

        # Price breakout (close above N-day high)
        lookback = min(self.criteria.breakout_lookback_days, len(high) - 1)
        if lookback > 0:
            period_high = float(np.max(high[-lookback - 1:-1]))
            if current > period_high:
                sig.breakout = True
                sig.signal_names.append("breakout")

        # Momentum crossover (5-day MA > 10-day MA, both rising)
        if len(close) >= 10:
            ma5 = float(np.mean(close[-5:]))
            ma10 = float(np.mean(close[-10:]))
            prev_ma5 = float(np.mean(close[-6:-1]))
            if ma5 > ma10 and ma5 > prev_ma5:
                sig.momentum_crossover = True
                sig.signal_names.append("momentum")

        # Gap up
        if len(open_price) >= 1 and prev > 0:
            sig.gap_pct = (float(open_price[-1]) - prev) / prev * 100
            if sig.gap_pct >= self.criteria.gap_up_pct:
                sig.gap_up = True
                sig.signal_names.append("gap_up")

        # RSI (14-period)
        sig.rsi = self._compute_rsi(close)
        if self.criteria.rsi_oversold and sig.rsi < self.criteria.rsi_oversold:
            sig.rsi_signal = True
            sig.signal_names.append("rsi_oversold")
        elif self.criteria.rsi_momentum_low and self.criteria.rsi_momentum_high:
            if self.criteria.rsi_momentum_low <= sig.rsi <= self.criteria.rsi_momentum_high:
                sig.rsi_signal = True
                sig.signal_names.append("rsi_momentum")

        # ATR percentage
        if len(high) >= 14:
            atr = self._compute_atr(high, low, close)
            sig.atr_pct = (atr / current * 100) if current > 0 else 0.0

        # 52-week high/low and drawdown
        if len(close) >= 5:
            sig.high_52w = float(np.max(high))
            sig.low_52w = float(np.min(low))
            if sig.high_52w > 0:
                sig.drawdown_pct = (sig.high_52w - current) / sig.high_52w * 100

        # Drawdown signal (for value/recovery strategies)
        if self.criteria.drawdown_from_high_min and sig.drawdown_pct >= self.criteria.drawdown_from_high_min:
            sig.signal_names.append("drawdown")

        # Valuation signals (P/E, P/B, dividend yield)
        self._check_valuation_signals(sig)

        sig.signals_triggered = len(sig.signal_names)
        sig.score = self._compute_score(sig)

        return sig

    def _check_valuation_signals(self, sig: StockSignals) -> None:
        try:
            info = yf.Ticker(sig.ticker).info
            sig.pe_ratio = info.get("trailingPE") or info.get("forwardPE")
            sig.pb_ratio = info.get("priceToBook")
            sig.dividend_yield = info.get("dividendYield")
            if sig.dividend_yield:
                sig.dividend_yield *= 100  # convert to percentage
            sig.market_cap = info.get("marketCap", 0) or 0
        except Exception:
            return

        if self.criteria.pe_max and sig.pe_ratio and 0 < sig.pe_ratio <= self.criteria.pe_max:
            sig.signal_names.append("low_pe")

        if self.criteria.pb_max and sig.pb_ratio and 0 < sig.pb_ratio <= self.criteria.pb_max:
            sig.signal_names.append("low_pb")

        if self.criteria.dividend_yield_min and sig.dividend_yield and sig.dividend_yield >= self.criteria.dividend_yield_min:
            sig.signal_names.append("high_yield")

    def _compute_score(self, sig: StockSignals) -> float:
        score = float(sig.signals_triggered) * 10.0

        if sig.volume_spike:
            score += min(sig.volume_ratio, 5.0) * 3.0
        if sig.breakout:
            score += 8.0
        if sig.momentum_crossover:
            score += 6.0
        if sig.rsi_signal:
            if "rsi_oversold" in sig.signal_names:
                score += 7.0
            else:
                score += 4.0
        if "low_pe" in sig.signal_names:
            score += 6.0
        if "low_pb" in sig.signal_names:
            score += 5.0
        if "high_yield" in sig.signal_names:
            score += 5.0
        if "drawdown" in sig.signal_names:
            score += 4.0

        return round(score, 2)

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> float:
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = float(np.mean(gains[-period:]))
        avg_loss = float(np.mean(losses[-period:]))
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100.0 - (100.0 / (1.0 + rs)), 2)

    @staticmethod
    def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        if len(high) < period + 1:
            return 0.0
        tr_values = []
        for i in range(-period, 0):
            h = float(high[i])
            l = float(low[i])
            pc = float(close[i - 1])
            tr = max(h - l, abs(h - pc), abs(l - pc))
            tr_values.append(tr)
        return float(np.mean(tr_values))
