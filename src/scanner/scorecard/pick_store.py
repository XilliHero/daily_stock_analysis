# -*- coding: utf-8 -*-
"""Persistent store for market-scan picks.

At scan time the pipeline produces rich ``RankedStock`` objects but only the
markdown was ever saved. This module persists the structured picks as JSON so
the scorecard can measure how each recommendation performed over time.

Two sources feed :func:`load_all_picks`:
- ``picks_<strategy>_<YYYYMMDD_HHMM>.json`` — written going forward by the scan.
- ``backfill_picks.json`` — reconstructed once from the old markdown scans.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

# Picks live alongside the saved scan markdown.
SCANS_DIR = os.path.join("output", "scans")
BACKFILL_FILENAME = "backfill_picks.json"


@dataclass
class PickRecord:
    """One stock as it appeared in one scan on one day."""

    scan_date: str  # ISO date "YYYY-MM-DD"
    strategy: str
    ticker: str
    name: str = ""
    price: Optional[float] = None  # price at scan time
    score: Optional[float] = None
    grade: str = ""
    signals: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PickRecord":
        return cls(
            scan_date=str(d.get("scan_date", "")),
            strategy=str(d.get("strategy", "")),
            ticker=str(d.get("ticker", "")).upper(),
            name=str(d.get("name", "") or ""),
            price=_coerce_float(d.get("price")),
            score=_coerce_float(d.get("score")),
            grade=str(d.get("grade", "") or ""),
            signals=list(d.get("signals") or []),
        )


def _coerce_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _scans_dir() -> str:
    os.makedirs(SCANS_DIR, exist_ok=True)
    return SCANS_DIR


def save_picks(strategy: str, scan_dt: datetime, top_picks: Iterable) -> Optional[str]:
    """Persist a scan's picks as JSON. Returns the path written, or None if empty.

    ``top_picks`` is an iterable of ``RankedStock`` (or anything with the same
    attribute names). Attribute access is defensive so a schema tweak upstream
    never crashes the scan.
    """
    scan_date = scan_dt.strftime("%Y-%m-%d")
    records: List[dict] = []
    for p in top_picks or []:
        ticker = str(getattr(p, "ticker", "") or "").strip().upper()
        if not ticker:
            continue
        record = PickRecord(
            scan_date=scan_date,
            strategy=strategy,
            ticker=ticker,
            name=str(getattr(p, "name", "") or ""),
            price=_coerce_float(getattr(p, "current_price", None)),
            score=_coerce_float(getattr(p, "composite_score", None)),
            grade=str(getattr(p, "fundamental_grade", "") or ""),
            signals=list(getattr(p, "signal_names", []) or []),
        )
        records.append(record.to_dict())

    if not records:
        logger.warning("[pick_store] no picks to save for strategy=%s", strategy)
        return None

    filename = f"picks_{strategy}_{scan_dt.strftime('%Y%m%d_%H%M')}.json"
    path = os.path.join(_scans_dir(), filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info("[pick_store] saved %d picks → %s", len(records), path)
    return path


def _load_json_records(path: str) -> List[PickRecord]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[pick_store] could not read %s: %s", path, exc)
        return []
    if not isinstance(raw, list):
        logger.warning("[pick_store] %s is not a list, skipping", path)
        return []
    records = []
    for item in raw:
        if isinstance(item, dict):
            records.append(PickRecord.from_dict(item))
    return records


def load_all_picks(scans_dir: Optional[str] = None) -> List[PickRecord]:
    """Load every pick record: forward-saved ``picks_*.json`` + the backfill file."""
    base = scans_dir or SCANS_DIR
    records: List[PickRecord] = []

    backfill_path = os.path.join(base, BACKFILL_FILENAME)
    if os.path.exists(backfill_path):
        records.extend(_load_json_records(backfill_path))

    for path in sorted(glob.glob(os.path.join(base, "picks_*.json"))):
        records.extend(_load_json_records(path))

    logger.info("[pick_store] loaded %d pick records from %s", len(records), base)
    return records
