# -*- coding: utf-8 -*-
"""Render a :class:`Scorecard` as a markdown section for the daily report.

Kept visually consistent with ``ReportAgent._render_markdown``: a leading
``##`` header, a summary table, and a short per-pick "best/worst" callout.
"""

from __future__ import annotations

from typing import List, Optional

from src.scanner.scorecard.evaluator import PickPerformance, Scorecard, StrategyScore


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.1f}%"


def _best_worst_line(score: StrategyScore) -> str:
    parts = []
    if score.best is not None:
        parts.append(f"🟢 {score.best.ticker} {_pct(score.best.return_pct)}")
    if score.worst is not None and score.worst is not score.best:
        parts.append(f"🔴 {score.worst.ticker} {_pct(score.worst.return_pct)}")
    return " · ".join(parts)


def _strategy_row(score: StrategyScore) -> str:
    if score.count == 0:
        return (
            f"| {score.strategy.capitalize()} | — | — | 0 | "
            f"{'no price data' if score.n_no_data else 'no picks yet'} |"
        )
    return (
        f"| {score.strategy.capitalize()} "
        f"| {_pct(score.avg_return)} "
        f"| {score.hit_rate * 100:.0f}% "
        f"| {score.count} "
        f"| {_best_worst_line(score)} |"
    )


def render_scorecard_markdown(scorecard: Scorecard) -> str:
    """Return the ``## 📊 Performance Scorecard`` markdown block."""
    lines: List[str] = []
    lines.append("## 📊 Performance Scorecard")
    lines.append(
        "**How past picks have performed since the day they first appeared** "
        "(cumulative return to date, no fees)."
    )
    if scorecard.since_date:
        lines.append(
            f"\nTracking **{scorecard.n_tickers}** stocks since "
            f"**{scorecard.since_date}** · as of **{scorecard.as_of}**"
        )
    lines.append("")
    lines.append("| Strategy | Avg Return | Hit Rate | Tracked | Best / Worst |")
    lines.append("|----------|-----------|----------|---------|--------------|")

    for score in scorecard.per_strategy:
        lines.append(_strategy_row(score))

    if scorecard.overall is not None and scorecard.overall.count:
        o = scorecard.overall
        lines.append(
            f"| **All** | **{_pct(o.avg_return)}** | **{o.hit_rate * 100:.0f}%** "
            f"| **{o.count}** | {_best_worst_line(o)} |"
        )

    # Interpretation footnote — hit rate is the honest headline number.
    if scorecard.overall is not None and scorecard.overall.count:
        o = scorecard.overall
        verdict = "beating a coin flip" if o.hit_rate > 0.5 else "below a coin flip"
        lines.append(
            f"\n_{o.hit_rate * 100:.0f}% of all picks are in the green "
            f"({verdict}); average pick is {_pct(o.avg_return)}._"
        )

    if scorecard.n_no_data:
        lines.append(
            f"\n_{scorecard.n_no_data} pick(s) excluded — no current price available._"
        )

    return "\n".join(lines)
