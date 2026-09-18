import os
import tempfile

from src.advisor.profile import InvestorProfile, Target
from src.advisor.repository import InvestorProfileRepository


def _fresh_repo():
    # Isolated DB per test via a temp file + a fresh DatabaseManager singleton.
    from src.storage import DatabaseManager
    DatabaseManager._instance = None  # reset singleton
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    dbm = DatabaseManager(db_url=f"sqlite:///{tmp.name}")
    return InvestorProfileRepository(db_manager=dbm), tmp.name


def test_load_returns_none_when_absent():
    repo, path = _fresh_repo()
    try:
        assert repo.load("nobody") is None
    finally:
        os.unlink(path)


def test_save_then_load_roundtrip():
    repo, path = _fresh_repo()
    try:
        profile = InvestorProfile(owner_id="default", risk_tolerance="aggressive",
                                  horizon_years=20, goals=["growth"], investable_cash=25000.0)
        target = Target(weights={"equity": 0.8, "cash": 0.2}, max_position_pct=0.2, locked=True)
        repo.save(profile, target)
        loaded_profile, loaded_target = repo.load("default")
        assert loaded_profile.risk_tolerance == "aggressive"
        assert loaded_profile.investable_cash == 25000.0
        assert loaded_target.weights == {"equity": 0.8, "cash": 0.2}
        assert loaded_target.locked is True
    finally:
        os.unlink(path)


def test_save_is_upsert():
    repo, path = _fresh_repo()
    try:
        repo.save(InvestorProfile(owner_id="default", horizon_years=5), Target(weights={"cash": 1.0}))
        repo.save(InvestorProfile(owner_id="default", horizon_years=30), Target(weights={"cash": 1.0}))
        loaded_profile, _ = repo.load("default")
        assert loaded_profile.horizon_years == 30  # updated, not duplicated
    finally:
        os.unlink(path)
