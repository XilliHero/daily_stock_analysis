import tempfile

from fastapi.testclient import TestClient


def _client():
    from src.storage import DatabaseManager
    DatabaseManager._instance = None
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    DatabaseManager(db_url=f"sqlite:///{tmp.name}")  # prime singleton on temp DB
    from api.app import create_app
    return TestClient(create_app())


def test_get_profile_404_before_creation():
    client = _client()
    r = client.get("/api/v1/advisor/profile")
    assert r.status_code == 404


def test_put_then_get_profile_roundtrip():
    client = _client()
    body = {"risk_tolerance": "aggressive", "horizon_years": 20, "goals": ["growth"],
            "investable_cash": 25000.0, "base_currency": "USD",
            "target": {"weights": {"equity": 0.8, "cash": 0.2}, "max_position_pct": 0.2,
                       "max_sector_pct": 0.4, "locked": True}}
    r = client.put("/api/v1/advisor/profile", json=body)
    assert r.status_code == 200
    got = client.get("/api/v1/advisor/profile").json()
    assert got["risk_tolerance"] == "aggressive"
    assert got["target"]["locked"] is True


def test_generate_plan_requires_locked_target():
    client = _client()
    body = {"risk_tolerance": "moderate", "horizon_years": 10, "investable_cash": 1000.0,
            "target": {"weights": {"equity": 1.0}, "locked": False}}
    client.put("/api/v1/advisor/profile", json=body)
    r = client.post("/api/v1/advisor/plan")
    assert r.status_code == 400  # target not locked
