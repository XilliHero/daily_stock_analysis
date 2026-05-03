# -*- coding: utf-8 -*-
"""
Market scanner strategy profiles.

Each profile defines screening criteria and decision weights
for the agent team. The active profile is selected via
MARKET_SCAN_STRATEGY in .env.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ScreeningCriteria:
    """Thresholds used by the Screener Agent."""

    volume_spike_min: float = 1.5
    breakout_lookback_days: int = 20
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_momentum_low: float = 55.0
    rsi_momentum_high: float = 75.0
    gap_up_pct: float = 2.0
    atr_min_pct: float = 3.0
    pe_max: Optional[float] = None
    pb_max: Optional[float] = None
    dividend_yield_min: Optional[float] = None
    dividend_years_min: Optional[int] = None
    payout_ratio_max: Optional[float] = None
    drawdown_from_high_min: Optional[float] = None
    revenue_growth_min: Optional[float] = None
    min_signals: int = 2


@dataclass
class DecisionWeights:
    """How the Decision Agent weights each agent's input."""

    screener: float = 0.0
    fundamental: float = 0.0
    intel: float = 0.0
    technical: float = 0.0
    risk: float = 0.0
    sector: float = 0.0


@dataclass
class StrategyProfile:
    """Complete strategy definition."""

    name: str
    description: str
    screening: ScreeningCriteria
    weights: DecisionWeights
    fundamental_depth: str = "full"  # "skip", "light", "full"
    preferred_sectors: List[str] = field(default_factory=list)
    avoided_sectors: List[str] = field(default_factory=list)


STRATEGY_PROFILES: Dict[str, StrategyProfile] = {
    "value": StrategyProfile(
        name="value",
        description="Undervalued quality stocks for medium-to-long-term holds",
        screening=ScreeningCriteria(
            pe_max=20.0,
            pb_max=1.5,
            drawdown_from_high_min=15.0,
            min_signals=2,
        ),
        weights=DecisionWeights(
            screener=0.20,
            fundamental=0.40,
            intel=0.20,
            technical=0.10,
            risk=0.10,
            sector=0.00,
        ),
        fundamental_depth="full",
    ),
    "growth": StrategyProfile(
        name="growth",
        description="Fast-growing companies with expanding revenue and margins",
        screening=ScreeningCriteria(
            revenue_growth_min=15.0,
            rsi_momentum_low=50.0,
            rsi_momentum_high=80.0,
            min_signals=2,
        ),
        weights=DecisionWeights(
            screener=0.15,
            fundamental=0.30,
            intel=0.25,
            technical=0.20,
            risk=0.00,
            sector=0.10,
        ),
        fundamental_depth="full",
        preferred_sectors=["Technology", "Healthcare", "Communication Services"],
    ),
    "dividend": StrategyProfile(
        name="dividend",
        description="Stable income stocks with reliable dividend history",
        screening=ScreeningCriteria(
            dividend_yield_min=2.5,
            payout_ratio_max=70.0,
            dividend_years_min=5,
            min_signals=2,
        ),
        weights=DecisionWeights(
            screener=0.25,
            fundamental=0.40,
            intel=0.10,
            technical=0.05,
            risk=0.20,
            sector=0.00,
        ),
        fundamental_depth="full",
        preferred_sectors=["Utilities", "Consumer Defensive", "Real Estate", "Financials"],
    ),
    "recovery": StrategyProfile(
        name="recovery",
        description="Beaten-down stocks with turnaround potential",
        screening=ScreeningCriteria(
            drawdown_from_high_min=25.0,
            rsi_oversold=35.0,
            volume_spike_min=1.3,
            min_signals=2,
        ),
        weights=DecisionWeights(
            screener=0.10,
            fundamental=0.25,
            intel=0.30,
            technical=0.25,
            risk=0.10,
            sector=0.00,
        ),
        fundamental_depth="full",
    ),
}


def get_strategy(name: str) -> StrategyProfile:
    """Retrieve a strategy profile by name, raising ValueError if unknown."""
    profile = STRATEGY_PROFILES.get(name.lower().strip())
    if profile is None:
        valid = ", ".join(sorted(STRATEGY_PROFILES.keys()))
        raise ValueError(f"Unknown scan strategy '{name}'. Valid: {valid}")
    return profile
