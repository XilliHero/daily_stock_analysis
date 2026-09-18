# -*- coding: utf-8 -*-
"""Turn a Gap into concrete buy/trim actions under the guardrail caps.

Rules:
- Trim only positions/sectors that are OVER cap (conviction + guardrails).
- Fill positive equity delta from ranked candidates, each new position <= per-
  position cap and its sector <= sector cap, spending no more than available.
- Non-equity positive deltas become a single dollar-target buy (symbol="").
- If equity delta can't be filled (no candidates), emit an 'unallocated' note.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List


@dataclass
class Action:
    side: str            # "buy" | "trim" | "unallocated"
    asset_class: str
    symbol: str = ""     # "" for a class-level dollar target
    amount: float = 0.0
    reason: str = ""


def size_positions(gap, candidates, current, target) -> List[Action]:
    base = current.base or 0.0
    pos_cap = target.max_position_pct * base
    sec_cap = target.max_sector_pct * base
    actions: List[Action] = []

    # 1) Trims — over-cap positions down to the cap.
    for symbol, over in sorted(gap.position_overage.items()):
        if over > 1e-9:
            actions.append(Action("trim", "equity", symbol, round(over, 2),
                                   f"{symbol} exceeds the {target.max_position_pct:.0%} per-position cap"))

    # 2) Buys per asset class from positive deltas.
    sector_running = defaultdict(float)
    for p in current.positions:
        if p.asset_class == "equity":
            sector_running[p.sector] += p.value

    for asset_class, delta in gap.class_delta.items():
        if delta <= 1e-6:
            continue
        if asset_class != "equity":
            actions.append(Action("buy", asset_class, "", round(delta, 2),
                                   f"increase {asset_class.replace('_', ' ')} by ${delta:,.0f} "
                                   f"(e.g. via a broad {asset_class.replace('_', ' ')} ETF)"))
            continue

        remaining = delta
        equity_cands = [c for c in candidates if c.strategy_count is not None]
        for c in equity_cands:
            if remaining <= 1e-6:
                break
            headroom_sector = sec_cap - sector_running[c.sector]
            room = min(pos_cap, headroom_sector, remaining)
            if room <= 1e-6:
                continue
            amount = round(room, 2)
            reason = f"fills equity target; {c.strategy_count}/4 strategies"
            if c.verdict:
                reason += f"; deep-dive: {c.verdict}"
            actions.append(Action("buy", "equity", c.ticker, amount, reason))
            sector_running[c.sector] += amount
            remaining -= amount

        if remaining > 1.0:  # meaningfully unfilled
            actions.append(Action("unallocated", "equity", "", round(remaining, 2),
                                   "no qualifying candidates under caps; consider a broad-market ETF"))

    return actions
