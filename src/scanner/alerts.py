# -*- coding: utf-8 -*-
"""Real-time alert digest — the actionable TL;DR of a scan.

The daily report is comprehensive but passive. This distills only the
high-priority, time-sensitive items into a short message that gets pushed to
every configured notification channel (email today; Telegram/Discord/etc. the
moment a push channel is configured — no code change needed).

Sources are already computed elsewhere: watchlist alerts (watchlist.py) and
cross-strategy overlaps (overlap.py). Pure text assembly — no network, no LLM.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

# Only these watchlist alert types are urgent enough to push (skip MA-cross and
# the low-signal "also in today's scan" note).
_HIGH_PRIORITY = ("oversold", "overbought", "52-week", "big move")

DEFAULT_MIN_STRATEGIES = 3  # overlap conviction threshold worth pinging about


def build_alert_digest(
    watchlist_report=None,
    overlap_board=None,
    health_line: Optional[str] = None,
    min_strategies: int = DEFAULT_MIN_STRATEGIES,
    as_of: Optional[date] = None,
) -> Optional[str]:
    """Compose a concise alert message, or None when nothing is worth pushing."""
    sections: List[str] = []

    if health_line:
        sections.append(health_line)

    if watchlist_report is not None:
        watch_lines: List[str] = []
        for item in getattr(watchlist_report, "items", []):
            urgent = [
                a for a in getattr(item, "alerts", [])
                if any(k in a.lower() for k in _HIGH_PRIORITY)
            ]
            if urgent:
                watch_lines.append(f"⭐ {item.ticker}: {' · '.join(urgent)}")
        if watch_lines:
            sections.append("**Watchlist:**\n" + "\n".join(watch_lines))

    if overlap_board is not None:
        high = [
            s for s in getattr(overlap_board, "stocks", [])
            if s.strategy_count >= min_strategies
        ]
        if high:
            conv_lines = [
                f"🔥 {s.ticker} — {s.strategy_count}/4 strategies ({', '.join(s.strategies)})"
                for s in high
            ]
            sections.append("**High-conviction:**\n" + "\n".join(conv_lines))

    if not sections:
        return None

    header = f"🔔 Scanner Alerts — {(as_of or date.today()).isoformat()}"
    return header + "\n\n" + "\n\n".join(sections)
