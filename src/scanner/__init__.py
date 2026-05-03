# -*- coding: utf-8 -*-
"""Market scanner package — agent team for US & Canadian stock screening."""

from src.scanner.pipeline import (
    MarketScannerPipeline,
    PipelineConfig,
    PipelineResult,
    run_market_scan,
    run_multi_strategy_scan,
)
from src.scanner.strategy_profiles import (
    STRATEGY_PROFILES,
    StrategyProfile,
    get_strategy,
)

__all__ = [
    "MarketScannerPipeline",
    "PipelineConfig",
    "PipelineResult",
    "STRATEGY_PROFILES",
    "StrategyProfile",
    "get_strategy",
    "run_market_scan",
    "run_multi_strategy_scan",
]
