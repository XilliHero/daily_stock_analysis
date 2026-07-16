# -*- coding: utf-8 -*-
"""Tests for the scorecard markdown backfill parser (offline)."""

from src.scanner.scorecard.backfill import parse_scan_markdown, run_backfill

SAMPLE = """# Market Scan Report — VALUE
**Undervalued quality stocks**

Date: 2026-07-09 18:16

## Pipeline Summary
- Universe: **947** stocks

## Sector Overview
| Rank | Sector | ETF |
|------|--------|-----|
| 1 | Energy | XLE |

## Top Picks
| # | Ticker | Name | Sector | Price | Chg% | Score | Grade | Signals |
|---|--------|------|--------|-------|------|-------|-------|---------|
| 1 | CALM | Cal-Maine Foods | Consumer Staples | $85.35 | +0.3% | 59 | A | momentum, rsi_momentum, low_pe |
| 2 | CVE.TO | Cenovus Energy | Energy | $1,036.85 | -2.4% | 50 | B | momentum |

## Detail Cards
### #1 CALM — Cal-Maine Foods
- **Price**: $85.35
"""


def test_parse_extracts_top_picks(tmp_path):
    p = tmp_path / "scan_value_20260709_1816.md"
    p.write_text(SAMPLE, encoding="utf-8")
    recs = parse_scan_markdown(str(p))

    assert len(recs) == 2
    calm = recs[0]
    assert calm.ticker == "CALM"
    assert calm.strategy == "value"
    assert calm.scan_date == "2026-07-09"
    assert calm.price == 85.35
    assert calm.score == 59.0
    assert calm.grade == "A"
    assert calm.signals == ["momentum", "rsi_momentum", "low_pe"]


def test_parse_strips_dollar_and_comma(tmp_path):
    p = tmp_path / "scan_value_20260709_1816.md"
    p.write_text(SAMPLE, encoding="utf-8")
    recs = parse_scan_markdown(str(p))
    cve = recs[1]
    assert cve.ticker == "CVE.TO"  # dotted symbols preserved
    assert cve.price == 1036.85    # "$1,036.85" cleaned


def test_parse_ignores_sector_table_and_detail_cards(tmp_path):
    p = tmp_path / "scan_value_20260709_1816.md"
    p.write_text(SAMPLE, encoding="utf-8")
    recs = parse_scan_markdown(str(p))
    # Only the two Top Picks rows — not "XLE" from Sector Overview.
    assert {r.ticker for r in recs} == {"CALM", "CVE.TO"}


def test_strategy_comes_from_filename(tmp_path):
    p = tmp_path / "scan_recovery_20260708_0802.md"
    p.write_text(SAMPLE, encoding="utf-8")
    recs = parse_scan_markdown(str(p))
    assert all(r.strategy == "recovery" for r in recs)
    assert all(r.scan_date == "2026-07-08" for r in recs)


def test_unrecognised_filename_returns_empty(tmp_path):
    p = tmp_path / "notascan.md"
    p.write_text(SAMPLE, encoding="utf-8")
    assert parse_scan_markdown(str(p)) == []


def test_run_backfill_writes_consolidated_json(tmp_path):
    (tmp_path / "scan_value_20260709_1816.md").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "scan_growth_20260709_1816.md").write_text(SAMPLE, encoding="utf-8")
    count = run_backfill(scans_dir=str(tmp_path))
    assert count == 4  # 2 files × 2 picks
    assert (tmp_path / "backfill_picks.json").exists()
