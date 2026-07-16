# -*- coding: utf-8 -*-
"""One-time backfill of scan picks from saved markdown reports.

The old scans were only ever saved as markdown (``output/scans/scan_*.md``).
This module reconstructs structured :class:`PickRecord` rows from the
"## Top Picks" table in each file so the scorecard has real history on day one.

Run once:  ``python -m src.scanner.scorecard.backfill``
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from typing import List, Optional

from src.scanner.scorecard.pick_store import (
    BACKFILL_FILENAME,
    SCANS_DIR,
    PickRecord,
    _coerce_float,
)

logger = logging.getLogger(__name__)

# scan_<strategy>_<YYYYMMDD>_<HHMM>.md
_FILENAME_RE = re.compile(
    r"scan_(?P<strategy>[a-z]+)_(?P<date>\d{8})_(?P<time>\d{4})\.md$",
    re.IGNORECASE,
)


def _parse_filename(path: str) -> Optional[tuple[str, str]]:
    """Return (strategy, iso_date) from a scan filename, or None if it doesn't match."""
    m = _FILENAME_RE.search(os.path.basename(path))
    if not m:
        return None
    strategy = m.group("strategy").lower()
    d = m.group("date")
    iso_date = f"{d[0:4]}-{d[4:6]}-{d[6:8]}"
    return strategy, iso_date


def _clean_price(cell: str) -> Optional[float]:
    """'$1,234.56' -> 1234.56 ; returns None if not numeric."""
    cleaned = cell.replace("$", "").replace(",", "").strip()
    return _coerce_float(cleaned)


def _parse_signals(cell: str) -> List[str]:
    return [s.strip() for s in cell.split(",") if s.strip()]


def parse_scan_markdown(path: str) -> List[PickRecord]:
    """Extract PickRecords from one saved scan markdown file.

    Reads rows of the ``## Top Picks`` table until the next ``##`` section.
    Row layout: ``| # | Ticker | Name | Sector | Price | Chg% | Score | Grade | Signals |``
    """
    meta = _parse_filename(path)
    if meta is None:
        logger.debug("[backfill] skipping unrecognised filename: %s", path)
        return []
    strategy, iso_date = meta

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as exc:
        logger.warning("[backfill] could not read %s: %s", path, exc)
        return []

    records: List[PickRecord] = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            # Enter on the Top Picks header; leave on the next section header.
            in_table = stripped.lower().startswith("## top picks")
            continue
        if not in_table or not stripped.startswith("|"):
            continue

        cells = [c.strip() for c in stripped.strip("|").split("|")]
        # Skip the header row and the |---|---| separator row.
        if len(cells) < 8:
            continue
        if not cells[0].isdigit():
            continue

        ticker = cells[1].upper()
        if not ticker:
            continue
        signals = _parse_signals(cells[8]) if len(cells) > 8 else []
        records.append(
            PickRecord(
                scan_date=iso_date,
                strategy=strategy,
                ticker=ticker,
                name=cells[2],
                price=_clean_price(cells[4]),
                score=_coerce_float(cells[6]),
                grade=cells[7],
                signals=signals,
            )
        )

    return records


def run_backfill(scans_dir: Optional[str] = None) -> int:
    """Parse every ``scan_*.md`` and write a consolidated ``backfill_picks.json``.

    Idempotent — overwrites the output file each run. Returns the record count.
    """
    base = scans_dir or SCANS_DIR
    paths = sorted(glob.glob(os.path.join(base, "scan_*.md")))
    all_records: List[dict] = []
    parsed_files = 0
    for path in paths:
        recs = parse_scan_markdown(path)
        if recs:
            parsed_files += 1
            all_records.extend(r.to_dict() for r in recs)

    out_path = os.path.join(base, BACKFILL_FILENAME)
    os.makedirs(base, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    logger.info(
        "[backfill] parsed %d/%d files → %d records → %s",
        parsed_files, len(paths), len(all_records), out_path,
    )
    return len(all_records)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    count = run_backfill()
    print(f"Backfill complete: {count} pick records written to "
          f"{os.path.join(SCANS_DIR, BACKFILL_FILENAME)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
