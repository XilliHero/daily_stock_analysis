# -*- coding: utf-8 -*-
"""
Market Scanner Pipeline — orchestrates the agent team.

Coordinates UniverseAgent → ScreenerAgent → FundamentalAgent + SectorAgent
(parallel) → ReportAgent, producing a final ScanReport.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional

from src.scanner.fundamental_agent import FundamentalAgent, FundamentalResult
from src.scanner.report_agent import ReportAgent, ScanReport
from src.scanner.screener_agent import ScreenerAgent, ScreenerResult
from src.scanner.sector_agent import SectorAgent, SectorResult
from src.scanner.strategy_profiles import StrategyProfile, get_strategy
from src.scanner.universe_agent import UniverseAgent, UniverseResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for a single pipeline run."""

    strategy_name: str = "value"
    regions: str = "us_ca"
    cap_tiers: List[str] = field(default_factory=lambda: ["large", "mid", "small"])
    include_sectors: Optional[List[str]] = None
    exclude_sectors: Optional[List[str]] = None
    top_n: int = 50
    fundamental_depth: str = "full"  # "skip", "light", "full"


@dataclass
class PipelineResult:
    """Full result from one pipeline run."""

    strategy: str
    report: Optional[ScanReport] = None
    universe_result: Optional[UniverseResult] = None
    screener_result: Optional[ScreenerResult] = None
    fundamental_result: Optional[FundamentalResult] = None
    sector_result: Optional[SectorResult] = None
    duration_s: float = 0.0
    errors: List[str] = field(default_factory=list)


class MarketScannerPipeline:
    """Orchestrates the full market scan agent team."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.strategy: StrategyProfile = get_strategy(config.strategy_name)

        depth = config.fundamental_depth or self.strategy.fundamental_depth
        self.skip_fundamental = depth == "skip"

    def run(self) -> PipelineResult:
        t0 = time.time()
        result = PipelineResult(strategy=self.config.strategy_name)

        logger.info(
            "[Pipeline] starting scan: strategy=%s regions=%s",
            self.config.strategy_name, self.config.regions,
        )

        # Stage 1: Build universe
        universe_agent = UniverseAgent(
            regions=self.config.regions,
            cap_tiers=self.config.cap_tiers,
            include_sectors=self.config.include_sectors,
            exclude_sectors=self.config.exclude_sectors,
        )
        universe_result = universe_agent.run()
        result.universe_result = universe_result

        if not universe_result.stocks:
            result.errors.append("Universe is empty — nothing to scan")
            result.duration_s = round(time.time() - t0, 2)
            return result

        logger.info(
            "[Pipeline] universe: %d stocks", len(universe_result.stocks),
        )

        # Stage 2: Fast screening
        screener = ScreenerAgent(self.strategy)
        screener_result = screener.run(universe_result.stocks, top_n=self.config.top_n)
        result.screener_result = screener_result

        if not screener_result.shortlist:
            result.errors.append("Screener returned no candidates")
            result.duration_s = round(time.time() - t0, 2)
            return result

        logger.info(
            "[Pipeline] screener: %d candidates", len(screener_result.shortlist),
        )

        # Stage 3: Fundamental + Sector (parallel)
        fundamental_result, sector_result = self._run_parallel_analysis(
            screener_result
        )
        result.fundamental_result = fundamental_result
        result.sector_result = sector_result

        # Stage 4: Report assembly
        reporter = ReportAgent(self.strategy)
        report = reporter.run(
            screener_result=screener_result,
            fundamental_result=fundamental_result,
            sector_result=sector_result,
            universe_size=len(universe_result.stocks),
        )
        result.report = report
        result.duration_s = round(time.time() - t0, 2)

        logger.info(
            "[Pipeline] complete: strategy=%s, %d picks in %.1fs",
            self.config.strategy_name, len(report.top_picks), result.duration_s,
        )
        return result

    def _run_parallel_analysis(
        self, screener_result: ScreenerResult
    ) -> tuple[FundamentalResult, SectorResult]:
        fundamental_result = FundamentalResult()
        sector_result = SectorResult()

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}

            if not self.skip_fundamental:
                fund_agent = FundamentalAgent(self.strategy)
                futures["fundamental"] = executor.submit(
                    fund_agent.run, screener_result.shortlist
                )

            sector_agent = SectorAgent(self.strategy)
            futures["sector"] = executor.submit(
                sector_agent.run, screener_result.shortlist
            )

            for future in as_completed(futures.values()):
                name = next(k for k, v in futures.items() if v is future)
                try:
                    res = future.result()
                    if name == "fundamental":
                        fundamental_result = res
                    else:
                        sector_result = res
                except Exception as e:
                    logger.warning("[Pipeline] %s agent failed: %s", name, e)

        return fundamental_result, sector_result


def run_market_scan(
    strategy_name: str = "value",
    regions: str = "us_ca",
    cap_tiers: Optional[List[str]] = None,
    include_sectors: Optional[List[str]] = None,
    exclude_sectors: Optional[List[str]] = None,
    top_n: int = 50,
) -> PipelineResult:
    """Convenience function to run a single-strategy market scan."""
    config = PipelineConfig(
        strategy_name=strategy_name,
        regions=regions,
        cap_tiers=cap_tiers or ["large", "mid", "small"],
        include_sectors=include_sectors,
        exclude_sectors=exclude_sectors,
        top_n=top_n,
    )
    pipeline = MarketScannerPipeline(config)
    return pipeline.run()


def run_multi_strategy_scan(
    strategies: Optional[List[str]] = None,
    regions: str = "us_ca",
    cap_tiers: Optional[List[str]] = None,
    include_sectors: Optional[List[str]] = None,
    exclude_sectors: Optional[List[str]] = None,
    top_n: int = 50,
) -> List[PipelineResult]:
    """Run the scan pipeline for multiple strategies sequentially."""
    strategies = strategies or ["value", "growth", "dividend", "recovery"]
    results: List[PipelineResult] = []

    for name in strategies:
        logger.info("[MultiScan] running strategy: %s", name)
        result = run_market_scan(
            strategy_name=name,
            regions=regions,
            cap_tiers=cap_tiers,
            include_sectors=include_sectors,
            exclude_sectors=exclude_sectors,
            top_n=top_n,
        )
        results.append(result)

    return results
