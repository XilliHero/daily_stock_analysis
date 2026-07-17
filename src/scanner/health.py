# -*- coding: utf-8 -*-
"""Scan health monitoring — catch degraded scans before you trust them.

The 2026-07-14 scan went out silently despite being broken: the `growth`
strategy produced nothing and value scores were capped at ~40 (normal ~57)
because the fundamental ranking failed. This module inspects a multi-strategy
scan's results and flags those failure modes so a bad-data report is called
out (in the report and via alert) instead of read at face value.

Pure inspection of already-computed results — no network, no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

DEFAULT_EXPECTED = ("value", "growth", "dividend", "recovery")
SCORE_FLOOR = 45.0   # a strategy's best composite score below this is suspicious
TOP_N = 20           # inspect the same top-N the report table shows


@dataclass
class HealthReport:
    degraded: bool = False
    issues: List[str] = field(default_factory=list)


def assess_scan_health(
    results: Sequence,
    expected: Sequence[str] = DEFAULT_EXPECTED,
    score_floor: float = SCORE_FLOOR,
) -> HealthReport:
    """Inspect PipelineResult objects and report degradation issues."""
    issues: List[str] = []

    present = {
        r.strategy for r in results
        if getattr(r, "report", None) and getattr(r.report, "top_picks", None)
    }
    for strat in expected:
        if strat not in present:
            issues.append(f"Strategy '{strat}' produced no picks (missing or failed).")

    for r in results:
        report = getattr(r, "report", None)
        picks = getattr(report, "top_picks", None) if report else None
        if not picks:
            continue
        scores = [
            float(getattr(p, "composite_score", 0.0) or 0.0)
            for p in picks[:TOP_N]
        ]
        if not scores:
            continue
        distinct = {round(s, 2) for s in scores}
        if len(distinct) <= 1:
            issues.append(
                f"{r.strategy}: top scores are flat ({scores[0]:.0f}) — ranking likely failed."
            )
        elif max(scores) < score_floor:
            issues.append(
                f"{r.strategy}: best score {max(scores):.0f} is abnormally low "
                f"(< {score_floor:.0f}) — data may be incomplete."
            )

    for r in results:
        errs = getattr(r, "errors", None)
        if errs:
            issues.append(f"{r.strategy}: {errs[0]}")

    return HealthReport(degraded=bool(issues), issues=issues)


def render_health_warning(report: HealthReport) -> Optional[str]:
    """Markdown ``## ⚠️ Scan Health Warning`` block, or None when healthy."""
    if not report.degraded:
        return None
    lines = [
        "## ⚠️ Scan Health Warning",
        "This scan looks degraded — treat the picks below with caution.",
        "",
    ]
    lines.extend(f"- {issue}" for issue in report.issues)
    return "\n".join(lines)


def health_summary_line(report: HealthReport) -> Optional[str]:
    """One-liner for the alert digest, or None when healthy."""
    if not report.degraded:
        return None
    extra = f" (+{len(report.issues) - 1} more)" if len(report.issues) > 1 else ""
    return f"⚠️ Scan health degraded: {report.issues[0]}{extra}"
