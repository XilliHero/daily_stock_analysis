# -*- coding: utf-8 -*-
"""Investor profile + target allocation — pure data with validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.advisor import ASSET_CLASSES, DEFAULT_OWNER_ID

RISK_LEVELS = ("conservative", "moderate", "aggressive")
_WEIGHT_TOLERANCE = 1e-6


@dataclass
class Target:
    """Target allocation the planner aims for. Weights sum to 1.0."""

    weights: Dict[str, float] = field(default_factory=dict)
    max_position_pct: float = 0.15  # guardrail: max % of portfolio in one stock
    max_sector_pct: float = 0.30    # guardrail: max % in one equity sector
    locked: bool = False

    def validate(self) -> None:
        for cls in self.weights:
            if cls not in ASSET_CLASSES:
                raise ValueError(f"unknown asset class: {cls}")
        for cls, w in self.weights.items():
            if w < 0:
                raise ValueError(f"negative weight for {cls}: {w}")
        total = sum(self.weights.values())
        if abs(total - 1.0) > _WEIGHT_TOLERANCE:
            raise ValueError(f"weights must sum to 1.0, got {total:.6f}")
        for name, cap in (("max_position_pct", self.max_position_pct),
                          ("max_sector_pct", self.max_sector_pct)):
            if not (0 < cap <= 1.0):
                raise ValueError(f"{name} must be in (0, 1], got {cap}")


@dataclass
class InvestorProfile:
    owner_id: str = DEFAULT_OWNER_ID
    risk_tolerance: str = "moderate"
    horizon_years: int = 10
    goals: List[str] = field(default_factory=list)
    constraints: Dict[str, bool] = field(default_factory=dict)  # e.g. {"exclude_crypto": True}
    investable_cash: float = 0.0
    base_currency: str = "USD"

    def validate(self) -> None:
        if self.risk_tolerance not in RISK_LEVELS:
            raise ValueError(f"risk_tolerance must be one of {RISK_LEVELS}")
        if self.horizon_years <= 0:
            raise ValueError("horizon_years must be positive")
        if self.investable_cash < 0:
            raise ValueError("investable_cash must be >= 0")
