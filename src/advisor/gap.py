# -*- coding: utf-8 -*-
"""Deterministic gap: current vs target by asset class, plus cap overages.

class_delta[c] > 0 means "buy this class"; < 0 means "trim". Overages drive the
conviction+guardrails rule: only positions/sectors OVER their cap get trimmed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Gap:
    class_delta: Dict[str, float] = field(default_factory=dict)
    position_overage: Dict[str, float] = field(default_factory=dict)   # symbol -> $ over cap
    sector_overage: Dict[str, float] = field(default_factory=dict)     # sector -> $ over cap


def compute_gap(current, target) -> Gap:
    base = current.base or 0.0

    current_by_class: Dict[str, float] = defaultdict(float)
    current_by_class["cash"] += current.cash
    for p in current.positions:
        if p.asset_class == "cash":
            continue  # cash already counted via current.cash
        current_by_class[p.asset_class] += p.value

    classes = set(current_by_class) | set(target.weights)
    class_delta = {c: round(target.weights.get(c, 0.0) * base - current_by_class.get(c, 0.0), 6)
                   for c in classes}

    pos_cap = target.max_position_pct * base
    position_overage = {p.symbol: round(p.value - pos_cap, 6)
                        for p in current.positions if p.value - pos_cap > 1e-9}

    sector_totals: Dict[str, float] = defaultdict(float)
    for p in current.positions:
        # "Unknown" is missing sector data, not a real sector — don't cap on it.
        if p.asset_class == "equity" and p.sector and p.sector != "Unknown":
            sector_totals[p.sector] += p.value
    sec_cap = target.max_sector_pct * base
    sector_overage = {s: round(v - sec_cap, 6) for s, v in sector_totals.items() if v - sec_cap > 1e-9}

    return Gap(class_delta=class_delta, position_overage=position_overage, sector_overage=sector_overage)
