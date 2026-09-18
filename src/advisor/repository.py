# -*- coding: utf-8 -*-
"""Persistence for the investor profile + locked target (one row per owner)."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional, Tuple

from src.advisor.profile import InvestorProfile, Target
from src.storage import DatabaseManager, InvestorProfileRecord


class InvestorProfileRepository:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager.get_instance()

    def load(self, owner_id: str) -> Optional[Tuple[InvestorProfile, Target]]:
        with self.db.session_scope() as session:
            row = session.get(InvestorProfileRecord, owner_id)
            if row is None:
                return None
            profile = InvestorProfile(**json.loads(row.profile_json))
            target = Target(**json.loads(row.target_json))
            return profile, target

    def save(self, profile: InvestorProfile, target: Target) -> None:
        with self.db.session_scope() as session:
            row = session.get(InvestorProfileRecord, profile.owner_id)
            payload = dict(
                profile_json=json.dumps(asdict(profile)),
                target_json=json.dumps(asdict(target)),
                locked=bool(target.locked),
            )
            if row is None:
                session.add(InvestorProfileRecord(owner_id=profile.owner_id, **payload))
            else:
                row.profile_json = payload["profile_json"]
                row.target_json = payload["target_json"]
                row.locked = payload["locked"]
