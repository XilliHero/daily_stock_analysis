# -*- coding: utf-8 -*-
"""Tests for scan health monitoring (offline, synthetic results)."""

from types import SimpleNamespace

from src.scanner.health import (
    assess_scan_health,
    health_summary_line,
    render_health_warning,
)


def _pick(score):
    return SimpleNamespace(composite_score=score)


def _result(strategy, scores, errors=None):
    report = SimpleNamespace(top_picks=[_pick(s) for s in scores]) if scores else None
    return SimpleNamespace(strategy=strategy, report=report, errors=errors or [])


def _healthy_set():
    # four strategies, varied scores in the normal range
    return [
        _result("value", [57, 55, 53, 51, 49]),
        _result("growth", [50, 48, 46, 45, 44]),
        _result("dividend", [56, 54, 52, 50, 48]),
        _result("recovery", [34, 33, 32, 31, 30]),  # recovery scores are naturally lower
    ]


def test_healthy_scan_not_degraded():
    report = assess_scan_health(_healthy_set(), score_floor=30.0)
    assert report.degraded is False
    assert report.issues == []
    assert render_health_warning(report) is None
    assert health_summary_line(report) is None


def test_missing_strategy_flagged():
    results = [r for r in _healthy_set() if r.strategy != "growth"]
    report = assess_scan_health(results, score_floor=30.0)
    assert report.degraded is True
    assert any("growth" in i and "no picks" in i for i in report.issues)


def test_flat_scores_flagged():
    results = _healthy_set()
    results[0] = _result("value", [40, 40, 40, 40, 40])  # degenerate flat ranking
    report = assess_scan_health(results, score_floor=30.0)
    assert report.degraded is True
    assert any("value" in i and "flat" in i for i in report.issues)


def test_abnormally_low_scores_flagged():
    results = _healthy_set()
    results[0] = _result("value", [40, 39, 38, 37, 36])  # varied but all below floor
    report = assess_scan_health(results, score_floor=45.0)
    assert report.degraded is True
    assert any("value" in i and "abnormally low" in i for i in report.issues)


def test_errors_flagged():
    results = _healthy_set()
    results[1] = _result("growth", [50, 48], errors=["Screener returned no candidates"])
    report = assess_scan_health(results, score_floor=30.0)
    assert any("Screener returned no candidates" in i for i in report.issues)


def test_render_and_summary_when_degraded():
    results = [r for r in _healthy_set() if r.strategy != "growth"]
    report = assess_scan_health(results, score_floor=30.0)
    md = render_health_warning(report)
    assert md.startswith("## ⚠️ Scan Health Warning")
    assert "growth" in md
    assert health_summary_line(report).startswith("⚠️ Scan health degraded")
