# -*- coding: utf-8 -*-
"""Advisor endpoints: investor profile CRUD, allocation suggestion, plan generation.

Single-user v1: owner_id fixed to DEFAULT_OWNER_ID (mirrors portfolio's optional owner).
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from api.v1.schemas.advisor import (ActionModel, PlanResponse, ProfileRequest,
                                     ProfileResponse, SuggestResponse, TargetModel)
from src.advisor import DEFAULT_OWNER_ID
from src.advisor.allocation import suggest_allocation
from src.advisor.plan import generate_plan, render_markdown
from src.advisor.profile import InvestorProfile, Target
from src.advisor.repository import InvestorProfileRepository

logger = logging.getLogger(__name__)
router = APIRouter()


def _repo() -> InvestorProfileRepository:
    return InvestorProfileRepository()


@router.get("/profile", response_model=ProfileResponse, summary="Get the investor profile")
def get_profile() -> ProfileResponse:
    loaded = _repo().load(DEFAULT_OWNER_ID)
    if loaded is None:
        raise HTTPException(status_code=404, detail="No investor profile yet — create one.")
    profile, target = loaded
    return ProfileResponse(owner_id=profile.owner_id, risk_tolerance=profile.risk_tolerance,
                           horizon_years=profile.horizon_years, goals=profile.goals,
                           constraints=profile.constraints, investable_cash=profile.investable_cash,
                           base_currency=profile.base_currency, target=TargetModel(**vars(target)))


@router.put("/profile", response_model=ProfileResponse, summary="Create or update the profile")
def put_profile(body: ProfileRequest) -> ProfileResponse:
    profile = InvestorProfile(owner_id=DEFAULT_OWNER_ID, risk_tolerance=body.risk_tolerance,
                              horizon_years=body.horizon_years, goals=body.goals,
                              constraints=body.constraints, investable_cash=body.investable_cash,
                              base_currency=body.base_currency)
    target = Target(**body.target.model_dump())
    try:
        profile.validate()
        target.validate()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _repo().save(profile, target)
    return get_profile()


@router.post("/suggest", response_model=SuggestResponse, summary="Suggest a target from the profile")
def suggest() -> SuggestResponse:
    loaded = _repo().load(DEFAULT_OWNER_ID)
    if loaded is None:
        raise HTTPException(status_code=404, detail="No investor profile yet — create one.")
    profile, _ = loaded
    return SuggestResponse(target=TargetModel(**vars(suggest_allocation(profile))))


@router.post("/plan", response_model=PlanResponse, summary="Generate the whole-portfolio plan")
def create_plan(smart: bool = Query(True, description="Use the LLM to refine candidate selection")) -> PlanResponse:
    try:
        plan = generate_plan(owner_id=DEFAULT_OWNER_ID, smart=smart)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:  # target not locked
        raise HTTPException(status_code=400, detail=str(exc))
    return PlanResponse(mode=plan.mode, base=plan.base, target=plan.target,
                        actions=[ActionModel(**vars(a)) for a in plan.actions],
                        rationale=plan.rationale, markdown=render_markdown(plan),
                        generated_at=plan.generated_at, engine=plan.engine)
