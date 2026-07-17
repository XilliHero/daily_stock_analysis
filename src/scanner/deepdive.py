# -*- coding: utf-8 -*-
"""Auto deep-dive — the handoff from the scanner (Team A) to the AI agents (Team B).

After the daily scan, the highest-conviction cross-strategy picks (from the
Overlap Board) are run through Team B's full agent pipeline
(Technical → Intel → Risk → Decision) for a real AI verdict + trade levels, and
folded into the report.

Team A is LLM-free; each deep-dive costs a few Gemini calls, so this is limited
to a small, configurable number of names and is fully best-effort: any per-stock
failure is skipped, and a total failure just omits the section.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# analyze_fn(ticker) -> analysis response dict (or None)
AnalyzeFn = Callable[[str], Optional[dict]]


@dataclass
class DeepDiveResult:
    ticker: str
    name: str = ""
    strategy_count: int = 0
    strategies: List[str] = field(default_factory=list)
    current_price: Optional[float] = None
    change_pct: Optional[float] = None
    verdict: str = ""      # operation advice (buy / hold / reduce …)
    sentiment: str = ""
    trend: str = ""
    ideal_buy: Optional[str] = None
    stop_loss: Optional[str] = None
    take_profit: Optional[str] = None
    takeaway: str = ""


def select_deepdive_tickers(overlap_board, limit: int) -> list:
    """Top-`limit` names from the overlap board (already ranked by conviction)."""
    if limit <= 0 or overlap_board is None:
        return []
    return list(getattr(overlap_board, "stocks", [])[:limit])


def _default_analyze(ticker: str) -> Optional[dict]:
    from src.services.analysis_service import AnalysisService

    return AnalysisService().analyze_stock(
        ticker, report_type="simple", send_notification=False
    )


def _as_level(value) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value)


def run_deep_dives(picks: list, analyze_fn: Optional[AnalyzeFn] = None) -> List[DeepDiveResult]:
    """Run Team B on each pick; skip any whose analysis returns None or errors."""
    analyze = analyze_fn or _default_analyze
    results: List[DeepDiveResult] = []
    for pick in picks:
        ticker = getattr(pick, "ticker", "")
        try:
            resp = analyze(ticker)
        except Exception as exc:
            logger.warning("[deepdive] analysis failed for %s: %s", ticker, exc)
            continue
        if not resp:
            logger.info("[deepdive] no result for %s (skipped)", ticker)
            continue

        report = resp.get("report", {}) or {}
        meta = report.get("meta", {}) or {}
        summary = report.get("summary", {}) or {}
        strategy = report.get("strategy", {}) or {}

        results.append(
            DeepDiveResult(
                ticker=ticker,
                name=resp.get("stock_name") or getattr(pick, "name", "") or ticker,
                strategy_count=getattr(pick, "strategy_count", 0),
                strategies=list(getattr(pick, "strategies", []) or []),
                current_price=meta.get("current_price"),
                change_pct=meta.get("change_pct"),
                verdict=(summary.get("operation_advice") or "").strip(),
                sentiment=(summary.get("sentiment_label") or "").strip(),
                trend=(summary.get("trend_prediction") or "").strip(),
                ideal_buy=_as_level(strategy.get("ideal_buy")),
                stop_loss=_as_level(strategy.get("stop_loss")),
                take_profit=_as_level(strategy.get("take_profit")),
                takeaway=(summary.get("analysis_summary") or "").strip(),
            )
        )
    return results


def _fmt_price(value) -> str:
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_change(value) -> str:
    try:
        return f" ({float(value):+.1f}%)"
    except (TypeError, ValueError):
        return ""


def render_deepdive_markdown(results: List[DeepDiveResult]) -> Optional[str]:
    """Return the ``## 🔬 Deep Dive`` section, or None when there's nothing."""
    if not results:
        return None
    lines = [
        "## 🔬 Deep Dive — Top Conviction Picks",
        "AI analysis (Technical · Intel · Risk · Decision) of the highest-conviction names above.",
    ]
    for r in results:
        lines.append("")
        lines.append(f"### {r.ticker} — {r.name}  ·  {r.strategy_count}/4 strategies")
        lines.append(f"- **Price**: {_fmt_price(r.current_price)}{_fmt_change(r.change_pct)}")
        if r.verdict:
            lines.append(f"- **Verdict**: {r.verdict}")
        context = [x for x in (
            f"Sentiment: {r.sentiment}" if r.sentiment else "",
            f"Trend: {r.trend}" if r.trend else "",
        ) if x]
        if context:
            lines.append(f"- {' · '.join(context)}")
        # Sniper points are sometimes free-text from the LLM — keep them scannable.
        def _short(v: str, n: int = 70) -> str:
            v = " ".join(v.split())
            return v if len(v) <= n else v[: n - 1] + "…"

        levels = [x for x in (
            f"Buy {_short(r.ideal_buy)}" if r.ideal_buy else "",
            f"Stop {_short(r.stop_loss)}" if r.stop_loss else "",
            f"Target {_short(r.take_profit)}" if r.take_profit else "",
        ) if x]
        if levels:
            lines.append(f"- **Levels**: {' · '.join(levels)}")
        if r.takeaway:
            takeaway = r.takeaway if len(r.takeaway) <= 400 else r.takeaway[:397] + "…"
            lines.append(f"- **Takeaway**: {takeaway}")
    return "\n".join(lines)
