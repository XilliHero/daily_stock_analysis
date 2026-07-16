# -*- coding: utf-8 -*-
"""Contract tests for GET /api/v1/watchlist/validate (offline)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

try:
    from fastapi.testclient import TestClient

    from api.app import create_app
except Exception:  # pragma: no cover - fastapi not installed
    create_app = None


def _fake_frame():
    return pd.DataFrame({"Close": [10.0, 11.0, 12.5]})


class WatchlistValidateApiTest(unittest.TestCase):
    def _client(self, temp_dir):
        app = create_app(static_dir=Path(temp_dir))
        return TestClient(app)

    def test_valid_symbol_returns_true_with_price(self):
        if create_app is None:
            self.skipTest("fastapi not installed")
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "api.v1.endpoints.watchlist.fetch_price_history",
            return_value={"NVDA": _fake_frame()},
        ):
            resp = self._client(temp_dir).get("/api/v1/watchlist/validate", params={"symbol": "nvda"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["symbol"], "NVDA")  # normalized to upper
        self.assertTrue(body["valid"])
        self.assertEqual(body["price"], 12.5)

    def test_unknown_symbol_returns_false(self):
        if create_app is None:
            self.skipTest("fastapi not installed")
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "api.v1.endpoints.watchlist.fetch_price_history", return_value={}
        ):
            resp = self._client(temp_dir).get("/api/v1/watchlist/validate", params={"symbol": "ZZZZ"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["symbol"], "ZZZZ")
        self.assertFalse(body["valid"])
        self.assertIsNone(body["price"])

    def test_fetch_error_is_treated_as_invalid_not_500(self):
        if create_app is None:
            self.skipTest("fastapi not installed")
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "api.v1.endpoints.watchlist.fetch_price_history",
            side_effect=RuntimeError("network down"),
        ):
            resp = self._client(temp_dir).get("/api/v1/watchlist/validate", params={"symbol": "AAA"})
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["valid"])


if __name__ == "__main__":
    unittest.main()
