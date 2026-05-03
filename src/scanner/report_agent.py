# -*- coding: utf-8 -*-
"""
Report Agent — assembles the final market scan report.

Combines outputs from all upstream agents into a formatted
Markdown report with summary, ranked picks, sector overview,
and per-stock detail cards.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from src.scanner.fundamental_agent import FundamentalData, FundamentalResult
from src.scanner.screener_agent import ScreenerResult, StockSignals
from src.scanner.sector_agent import SectorAnalysis, SectorMetrics, SectorResult
from src.scanner.strategy_profiles import StrategyProfile

logger = logging.getLogger(__name__)


@dataclass
class RankedStock:
    """Final ranked stock with combined scores."""

    rank: int
    ticker: str
    name: str
    sector: str
    cap_tier: str
    current_price: float
    change_pct: float

    screener_score: float = 0.0
    fundamental_score: float = 0.0
    fundamental_grade: str = ""
    sector_score: float = 0.0

    composite_score: float = 0.0
    signal_names: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    roe: float = 0.0
    debt_to_equity: Optional[float] = None
    revenue_growth: Optional[float] = None

    sector_trend: str = "neutral"
    sector_rank: int = 0


@dataclass
class ScanReport:
    """Complete market scan report."""

    strategy: str
    strategy_description: str
    scan_date: str
    total_universe: int = 0
    total_screened: int = 0
    total_qualified: int = 0
    total_analyzed: int = 0

    top_picks: List[RankedStock] = field(default_factory=list)
    sector_overview: List[SectorMetrics] = field(default_factory=list)

    duration_s: float = 0.0
    markdown: str = ""
    errors: List[str] = field(default_factory=list)


class ReportAgent:
    """Assembles the final market scan report."""

    def __init__(self, strategy: StrategyProfile):
        self.strategy = strategy

    def run(
        self,
        screener_result: ScreenerResult,
        fundamental_result: FundamentalResult,
        sector_result: SectorResult,
        universe_size: int = 0,
    ) -> ScanReport:
        t0 = time.time()
        report = ScanReport(
            strategy=self.strategy.name,
            strategy_description=self.strategy.description,
            scan_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            total_universe=universe_size,
            total_screened=screener_result.total_scanned,
            total_qualified=screener_result.total_shortlisted,
            total_analyzed=fundamental_result.total_analyzed,
        )

        ranked = self._rank_stocks(screener_result, fundamental_result, sector_result)
        report.top_picks = ranked
        report.sector_overview = sector_result.sector_rankings
        report.errors = (
            screener_result.errors + fundamental_result.errors + sector_result.errors
        )
        report.markdown = self._render_markdown(report)
        report.duration_s = round(time.time() - t0, 2)

        logger.info(
            "[ReportAgent] done: %d top picks, report length %d chars in %.1fs",
            len(report.top_picks), len(report.markdown), report.duration_s,
        )
        return report

    def _rank_stocks(
        self,
        screener_result: ScreenerResult,
        fundamental_result: FundamentalResult,
        sector_result: SectorResult,
    ) -> List[RankedStock]:
        fund_map: Dict[str, FundamentalData] = {
            fd.ticker: fd for fd in fundamental_result.analyses
        }
        sector_map: Dict[str, SectorAnalysis] = {
            sa.ticker: sa for sa in sector_result.stock_analyses
        }

        weights = self.strategy.weights
        ranked: List[RankedStock] = []

        for sig in screener_result.shortlist:
            fd = fund_map.get(sig.ticker)
            sa = sector_map.get(sig.ticker)

            rs = RankedStock(
                rank=0,
                ticker=sig.ticker,
                name=sig.name,
                sector=sig.sector,
                cap_tier=sig.cap_tier,
                current_price=sig.current_price,
                change_pct=sig.change_pct,
                screener_score=sig.score,
                signal_names=list(sig.signal_names),
                pe_ratio=sig.pe_ratio,
                pb_ratio=sig.pb_ratio,
                dividend_yield=sig.dividend_yield,
            )

            if fd:
                rs.fundamental_score = fd.score
                rs.fundamental_grade = fd.grade
                rs.flags = list(fd.flags)
                rs.roe = fd.roe
                rs.debt_to_equity = fd.debt_to_equity
                rs.revenue_growth = fd.revenue_growth

            if sa:
                rs.sector_score = sa.sector_score
                rs.sector_trend = sa.sector_trend
                rs.sector_rank = sa.sector_rank

            screener_norm = min(sig.score / 60.0, 1.0) * 100
            fund_norm = fd.score if fd else 50.0
            sector_norm = sa.sector_score if sa else 50.0

            rs.composite_score = round(
                screener_norm * weights.screener
                + fund_norm * weights.fundamental
                + sector_norm * weights.sector,
                2,
            )

            ranked.append(rs)

        ranked.sort(key=lambda r: r.composite_score, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1

        return ranked

    def _render_markdown(self, report: ScanReport) -> str:
        lines: List[str] = []

        lines.append(f"# Market Scan Report — {report.strategy.upper()}")
        lines.append(f"**{report.strategy_description}**\n")
        lines.append(f"Date: {report.scan_date}\n")

        lines.append("## Pipeline Summary")
        lines.append(f"- Universe: **{report.total_universe:,}** stocks")
        lines.append(f"- Screened: **{report.total_screened:,}** stocks")
        lines.append(f"- Qualified: **{report.total_qualified}** passed signal thresholds")
        lines.append(f"- Analyzed: **{report.total_analyzed}** deep fundamental review")
        lines.append(f"- Top picks: **{len(report.top_picks)}**\n")

        if report.sector_overview:
            lines.append("## Sector Overview")
            lines.append("| Rank | Sector | ETF | 1W % | 1M % | Rel Str | Trend |")
            lines.append("|------|--------|-----|------|------|---------|-------|")
            for sm in report.sector_overview:
                lines.append(
                    f"| {sm.rank} | {sm.sector} | {sm.etf} | "
                    f"{sm.change_1w:+.1f}% | {sm.change_1m:+.1f}% | "
                    f"{sm.relative_strength_1m:+.1f} | {sm.trend} |"
                )
            lines.append("")

        if report.top_picks:
            lines.append("## Top Picks")
            lines.append(
                "| # | Ticker | Name | Sector | Price | Chg% | "
                "Score | Grade | Signals |"
            )
            lines.append(
                "|---|--------|------|--------|-------|------|"
                "-------|-------|---------|"
            )
            for rs in report.top_picks[:20]:
                signals_str = ", ".join(rs.signal_names[:3])
                name_short = rs.name[:20] if rs.name else ""
                lines.append(
                    f"| {rs.rank} | {rs.ticker} | {name_short} | {rs.sector} | "
                    f"${rs.current_price:.2f} | {rs.change_pct:+.1f}% | "
                    f"{rs.composite_score:.0f} | {rs.fundamental_grade} | "
                    f"{signals_str} |"
                )
            lines.append("")

            lines.append("## Detail Cards\n")
            for rs in report.top_picks[:10]:
                lines.append(f"### #{rs.rank} {rs.ticker} — {rs.name}")
                lines.append(f"- **Sector**: {rs.sector} ({rs.sector_trend}, rank #{rs.sector_rank})")
                lines.append(f"- **Cap tier**: {rs.cap_tier}")
                lines.append(f"- **Price**: ${rs.current_price:.2f} ({rs.change_pct:+.1f}%)")
                lines.append(f"- **Composite score**: {rs.composite_score:.0f}")
                lines.append(
                    f"  - Screener: {rs.screener_score:.0f} | "
                    f"Fundamental: {rs.fundamental_score:.0f} ({rs.fundamental_grade}) | "
                    f"Sector: {rs.sector_score:.0f}"
                )
                lines.append(f"- **Signals**: {', '.join(rs.signal_names)}")

                val_parts = []
                if rs.pe_ratio:
                    val_parts.append(f"P/E {rs.pe_ratio:.1f}")
                if rs.pb_ratio:
                    val_parts.append(f"P/B {rs.pb_ratio:.1f}")
                if rs.dividend_yield:
                    val_parts.append(f"Yield {rs.dividend_yield:.1f}%")
                if rs.roe:
                    val_parts.append(f"ROE {rs.roe:.1f}%")
                if rs.debt_to_equity is not None:
                    val_parts.append(f"D/E {rs.debt_to_equity:.2f}")
                if rs.revenue_growth is not None:
                    val_parts.append(f"Rev Growth {rs.revenue_growth:.1f}%")
                if val_parts:
                    lines.append(f"- **Metrics**: {' | '.join(val_parts)}")

                if rs.flags:
                    lines.append(f"- **Flags**: {', '.join(rs.flags)}")
                lines.append("")

        if report.errors:
            lines.append("## Errors")
            for err in report.errors[:10]:
                lines.append(f"- {err}")
            if len(report.errors) > 10:
                lines.append(f"- ...and {len(report.errors) - 10} more")
            lines.append("")

        lines.append("---")
        lines.append(f"*Generated by Market Scanner ({report.strategy}) on {report.scan_date}*")

        return "\n".join(lines)
