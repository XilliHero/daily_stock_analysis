import pytest
from src.advisor.profile import InvestorProfile, Target, ASSET_CLASSES, DEFAULT_OWNER_ID


def test_profile_defaults():
    p = InvestorProfile()
    assert p.owner_id == DEFAULT_OWNER_ID
    assert p.risk_tolerance == "moderate"
    assert p.horizon_years == 10
    assert p.base_currency == "USD"
    p.validate()  # defaults are valid


def test_profile_rejects_bad_risk():
    with pytest.raises(ValueError):
        InvestorProfile(risk_tolerance="yolo").validate()


def test_target_weights_must_sum_to_one():
    Target(weights={"equity": 0.7, "fixed_income": 0.2, "cash": 0.1}).validate()
    with pytest.raises(ValueError):
        Target(weights={"equity": 0.7, "cash": 0.1}).validate()  # sums to 0.8


def test_target_rejects_unknown_class_and_bad_caps():
    with pytest.raises(ValueError):
        Target(weights={"stonks": 1.0}).validate()
    with pytest.raises(ValueError):
        Target(weights={"equity": 1.0}, max_position_pct=1.5).validate()


def test_target_known_classes_subset():
    assert set(ASSET_CLASSES) == {"equity", "fixed_income", "cash", "crypto", "commodity"}
