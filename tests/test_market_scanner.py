# -*- coding: utf-8 -*-
"""Unit tests for the market scanner agent team (offline, no network)."""

import numpy as np
import pytest

from src.scanner.strategy_profiles import (
    STRATEGY_PROFILES,
    ScreeningCriteria,
    StrategyProfile,
    get_strategy,
)
from src.scanner.screener_agent import ScreenerAgent, StockSignals
from src.scanner.fundamental_agent import FundamentalData
from src.scanner.sector_agent import SectorMetrics
from src.scanner.report_agent import RankedStock, ReportAgent, ScanReport
from src.scanner.pipeline import PipelineConfig


class TestStrategyProfiles:

    def test_all_four_profiles_exist(self):
        assert set(STRATEGY_PROFILES.keys()) == {"value", "growth", "dividend", "recovery"}

    def test_get_strategy_returns_correct_profile(self):
        p = get_strategy("value")
        assert p.name == "value"
        assert p.screening.pe_max == 20.0

    def test_get_strategy_case_insensitive(self):
        p = get_strategy("  Growth  ")
        assert p.name == "growth"

    def test_get_strategy_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown scan strategy"):
            get_strategy("yolo")

    def test_weights_sum_to_one(self):
        for name, profile in STRATEGY_PROFILES.items():
            w = profile.weights
            total = w.screener + w.fundamental + w.intel + w.technical + w.risk + w.sector
            assert abs(total - 1.0) < 0.01, f"{name} weights sum to {total}"


class TestScreenerRSI:

    def test_rsi_bullish(self):
        close = np.array([10 + i * 0.5 for i in range(20)], dtype=float)
        rsi = ScreenerAgent._compute_rsi(close)
        assert rsi > 70

    def test_rsi_bearish(self):
        close = np.array([30 - i * 0.5 for i in range(20)], dtype=float)
        rsi = ScreenerAgent._compute_rsi(close)
        assert rsi < 30

    def test_rsi_short_series_returns_default(self):
        close = np.array([10.0, 11.0, 10.5], dtype=float)
        rsi = ScreenerAgent._compute_rsi(close)
        assert rsi == 50.0


class TestScreenerATR:

    def test_atr_basic(self):
        high = np.array([12.0] * 20, dtype=float)
        low = np.array([10.0] * 20, dtype=float)
        close = np.array([11.0] * 20, dtype=float)
        atr = ScreenerAgent._compute_atr(high, low, close)
        assert atr == pytest.approx(2.0, abs=0.01)

    def test_atr_short_series_returns_zero(self):
        high = np.array([12.0] * 5, dtype=float)
        low = np.array([10.0] * 5, dtype=float)
        close = np.array([11.0] * 5, dtype=float)
        atr = ScreenerAgent._compute_atr(high, low, close)
        assert atr == 0.0


class TestFundamentalGrade:

    def test_grade_assignment(self):
        from src.scanner.fundamental_agent import FundamentalAgent

        assert FundamentalAgent._assign_grade(85) == "A"
        assert FundamentalAgent._assign_grade(70) == "B"
        assert FundamentalAgent._assign_grade(55) == "C"
        assert FundamentalAgent._assign_grade(40) == "D"
        assert FundamentalAgent._assign_grade(20) == "F"


class TestSectorMetrics:

    def test_sector_metrics_defaults(self):
        sm = SectorMetrics(sector="Technology", etf="XLK")
        assert sm.trend == "neutral"
        assert sm.momentum_score == 0.0

    def test_sector_momentum_computation(self):
        from src.scanner.sector_agent import SectorAgent
        close = np.array([100 + i for i in range(25)], dtype=float)
        score = SectorAgent._compute_momentum(close)
        assert score == 100.0


class TestPipelineConfig:

    def test_default_config(self):
        cfg = PipelineConfig()
        assert cfg.strategy_name == "value"
        assert cfg.regions == "us_ca"
        assert "large" in cfg.cap_tiers
        assert "mid" in cfg.cap_tiers
        assert "small" in cfg.cap_tiers

    def test_custom_config(self):
        cfg = PipelineConfig(
            strategy_name="growth",
            regions="us",
            cap_tiers=["large"],
            include_sectors=["Technology"],
            top_n=10,
        )
        assert cfg.strategy_name == "growth"
        assert cfg.top_n == 10


class TestReportRendering:

    def test_empty_report_renders(self):
        strategy = get_strategy("value")
        reporter = ReportAgent(strategy)

        from src.scanner.screener_agent import ScreenerResult
        from src.scanner.fundamental_agent import FundamentalResult
        from src.scanner.sector_agent import SectorResult

        report = reporter.run(
            screener_result=ScreenerResult(),
            fundamental_result=FundamentalResult(),
            sector_result=SectorResult(),
            universe_size=0,
        )
        assert "VALUE" in report.markdown
        assert "Pipeline Summary" in report.markdown
        assert report.strategy == "value"
