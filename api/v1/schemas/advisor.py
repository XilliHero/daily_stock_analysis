# -*- coding: utf-8 -*-
"""Advisor API schemas."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class TargetModel(BaseModel):
    weights: Dict[str, float] = Field(default_factory=dict)
    max_position_pct: float = 0.15
    max_sector_pct: float = 0.30
    locked: bool = False


class ProfileRequest(BaseModel):
    risk_tolerance: str = "moderate"
    horizon_years: int = 10
    goals: List[str] = Field(default_factory=list)
    constraints: Dict[str, bool] = Field(default_factory=dict)
    investable_cash: float = 0.0
    base_currency: str = "USD"
    target: TargetModel = Field(default_factory=TargetModel)


class ProfileResponse(ProfileRequest):
    owner_id: str


class SuggestResponse(BaseModel):
    target: TargetModel


class ActionModel(BaseModel):
    side: str
    asset_class: str
    symbol: str = ""
    amount: float = 0.0
    reason: str = ""


class PlanResponse(BaseModel):
    mode: str
    base: float
    target: Dict[str, float]
    actions: List[ActionModel]
    rationale: str
    markdown: str
    generated_at: str
    engine: str = "deterministic"
