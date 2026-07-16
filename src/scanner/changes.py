# -*- coding: utf-8 -*-
"""What Changed Since Yesterday — day-over-day diff of the scan picks.

Compares the most recent scan day against the previous one and surfaces:
- 🆕 new entries / ❌ dropped names per strategy
- 🔥 rising conviction: stocks now appearing in more strategies than before

Reuses the same :class:`PickRecord` store as the scorecard. Pure set math —
no network, no LLM calls.

To keep the diff apples-to-apples, both days are normalised to their **top-N
by composite score** (the backfill kept 20 picks/strategy while live runs keep
50; comparing the top-N avoids spurious "new" entries from that difference).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.scanner.scorecard.pick_store import PickRecord, load_all_picks

logger = logging.getLogger(__name__)

DEFAULT_TOP_N = 20
DEFAULT_MIN_CONVICTION = 2


@dataclass
class Entry:
    ticker: str
    name: str = ""
    rank: Optional[int] = None  # rank on the latest day (1 = best)


@dataclass
class StrategyDiff:
    strategy: str
    new_entries: List[Entry] = field(default_factory=list)
    dropped: List[Entry] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.new_entries or self.dropped)


@dataclass
class ConvictionRise:
    ticker: str
    name: str
    prev_count: int
    latest_count: int
    strategies: List[str] = field(default_factory=list)


@dataclass
class ChangeReport:
    latest_date: str
    prev_date: str
    per_strategy: List[StrategyDiff] = field(default_factory=list)
    conviction_up: List[ConvictionRise] = field(default_factory=list)

    @property
    def has_any(self) -> bool:
        return any(s.has_changes for s in self.per_strategy) or bool(self.conviction_up)


def _group(records: List[PickRecord]) -> Dict[str, Dict[str, Dict[str, PickRecord]]]:
    """Return {strategy: {date: {ticker: record}}}.

    Later records overwrite earlier ones, so when a day has multiple runs the
    most recently loaded (latest) run wins for a given ticker.
    """
    out: Dict[str, Dict[str, Dict[str, PickRecord]]] = {}
    for r in records:
        if not r.strategy or not r.scan_date or not r.ticker:
            continue
        out.setdefault(r.strategy, {}).setdefault(r.scan_date, {})[r.ticker] = r
    return out


def _top_n(day: Dict[str, PickRecord], top_n: int) -> Dict[str, int]:
    """Return {ticker: rank} for the top-N tickers of a day, ranked by score desc."""
    ranked = sorted(
        day.values(),
        key=lambda r: (r.score if r.score is not None else float("-inf")),
        reverse=True,
    )
    return {r.ticker: i + 1 for i, r in enumerate(ranked[:top_n])}


def _all_dates(group: Dict[str, Dict[str, Dict[str, PickRecord]]]) -> List[str]:
    dates = {d for by_date in group.values() for d in by_date}
    return sorted(dates)


def _is_degenerate(day: Dict[str, PickRecord], top_n: int) -> bool:
    """True if the day's top-N picks have no score variation (all equal/None).

    Signals a degraded scan where the fundamental ranking failed and the top
    picks are all tied on a base score — the top-N is then just alphabetical
    noise, a useless baseline to diff against.
    """
    ranks = _top_n(day, top_n)
    scores = {day[t].score for t in ranks if day[t].score is not None}
    return len(scores) <= 1


def _healthy_prev_date(
    by_date: Dict[str, Dict[str, PickRecord]], latest: str, top_n: int
) -> Optional[str]:
    """Most recent prior date whose top-N isn't degenerate; falls back to the
    most recent prior date if every earlier scan is degenerate."""
    prior = sorted((d for d in by_date if d < latest), reverse=True)
    for d in prior:
        if not _is_degenerate(by_date[d], top_n):
            return d
    return prior[0] if prior else None


def _conviction_counts(
    group: Dict[str, Dict[str, Dict[str, PickRecord]]],
    date_by_strategy: Dict[str, str],
    top_n: int,
):
    """{ticker: [count, [strategies], name]} counting, per strategy, the top-N of
    that strategy's chosen date. ``date_by_strategy`` maps strategy -> scan date,
    which lets the 'previous' snapshot use each strategy's own prior scan day."""
    counts: Dict[str, list] = {}
    for strat, by_date in group.items():
        date = date_by_strategy.get(strat)
        if not date:
            continue
        day = by_date.get(date)
        if not day:
            continue
        for ticker in _top_n(day, top_n):
            rec = day[ticker]
            if ticker not in counts:
                counts[ticker] = [0, [], rec.name]
            counts[ticker][0] += 1
            counts[ticker][1].append(strat)
    return counts


def detect_changes(
    records: Optional[List[PickRecord]] = None,
    top_n: int = DEFAULT_TOP_N,
    min_conviction: int = DEFAULT_MIN_CONVICTION,
) -> Optional[ChangeReport]:
    """Diff the latest scan day against the previous one.

    Returns None when there aren't two distinct scan days to compare.
    """
    if records is None:
        records = load_all_picks()
    group = _group(records)
    dates = _all_dates(group)
    if len(dates) < 2:
        return None
    latest = dates[-1]

    # Each strategy compares against its OWN most recent prior scan day, so a
    # strategy that skipped a day (e.g. growth on 2026-07-14) still diffs
    # against real data rather than an empty set.
    prev_by_strategy: Dict[str, str] = {}
    for strat, by_date in group.items():
        prev_s = _healthy_prev_date(by_date, latest, top_n)
        if prev_s is not None:
            prev_by_strategy[strat] = prev_s
    if not prev_by_strategy:
        return None

    per_strategy: List[StrategyDiff] = []
    for strat in sorted(group):
        by_date = group[strat]
        latest_day = by_date.get(latest)
        prev_s = prev_by_strategy.get(strat)
        if not latest_day or prev_s is None:
            continue  # need both a latest snapshot and a baseline to diff
        latest_ranks = _top_n(latest_day, top_n)
        prev_day = by_date.get(prev_s, {})
        prev_ranks = _top_n(prev_day, top_n)

        diff = StrategyDiff(strategy=strat)
        for ticker, rank in sorted(latest_ranks.items(), key=lambda kv: kv[1]):
            if ticker not in prev_ranks:
                diff.new_entries.append(
                    Entry(ticker=ticker, name=latest_day[ticker].name, rank=rank)
                )
        for ticker in prev_ranks:
            if ticker not in latest_ranks:
                diff.dropped.append(
                    Entry(ticker=ticker, name=prev_day.get(ticker, PickRecord("", "", ticker)).name)
                )
        per_strategy.append(diff)

    # Rising conviction: latest snapshot uses the global latest date for every
    # strategy; the baseline uses each strategy's own prior scan day.
    latest_counts = _conviction_counts(group, {s: latest for s in group}, top_n)
    prev_counts = _conviction_counts(group, prev_by_strategy, top_n)
    conviction_up: List[ConvictionRise] = []
    for ticker, (count, strategies, name) in latest_counts.items():
        prev_count = prev_counts.get(ticker, [0])[0]
        if count >= min_conviction and count > prev_count:
            conviction_up.append(
                ConvictionRise(
                    ticker=ticker,
                    name=name,
                    prev_count=prev_count,
                    latest_count=count,
                    strategies=sorted(strategies),
                )
            )
    conviction_up.sort(key=lambda c: (c.latest_count, c.latest_count - c.prev_count), reverse=True)

    return ChangeReport(
        latest_date=latest,
        prev_date=max(prev_by_strategy.values()),
        per_strategy=per_strategy,
        conviction_up=conviction_up,
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

_MAX_LISTED = 10


def _fmt_entries(entries: List[Entry], with_rank: bool) -> str:
    shown = entries[:_MAX_LISTED]
    parts = []
    for e in shown:
        if with_rank and e.rank is not None:
            parts.append(f"{e.ticker} (#{e.rank})")
        else:
            parts.append(e.ticker)
    text = ", ".join(parts)
    extra = len(entries) - len(shown)
    if extra > 0:
        text += f", +{extra} more"
    return text


def render_changes_markdown(report: ChangeReport) -> str:
    """Return the ``## 🔄 What Changed`` markdown block."""
    lines: List[str] = []
    lines.append(f"## 🔄 What Changed Since {report.prev_date}")
    lines.append(
        f"Movement between the last two scans ({report.prev_date} → {report.latest_date})."
    )

    if report.conviction_up:
        lines.append("")
        lines.append("**🔥 Rising conviction (now in more strategies):**")
        for c in report.conviction_up[:_MAX_LISTED]:
            lines.append(
                f"- **{c.ticker}** — {c.latest_count} strategies "
                f"(was {c.prev_count}): {', '.join(c.strategies)}"
            )

    for diff in report.per_strategy:
        if not diff.has_changes:
            continue
        lines.append("")
        lines.append(f"### {diff.strategy.capitalize()}")
        if diff.new_entries:
            lines.append(f"🆕 **New:** {_fmt_entries(diff.new_entries, with_rank=True)}")
        if diff.dropped:
            lines.append(f"❌ **Dropped:** {_fmt_entries(diff.dropped, with_rank=False)}")

    if not report.has_any:
        lines.append("")
        lines.append("_No changes since the previous scan._")

    return "\n".join(lines)
