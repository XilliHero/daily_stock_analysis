# -*- coding: utf-8 -*-
"""Tests for pivot-point support/resistance (pure arithmetic, no network)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd  # noqa: E402

from src.agent.tools.support_resistance import (  # noqa: E402
    compute_pivot_levels,
    pivot_levels_from_history,
    _nearest,
)


class TestComputePivotLevels(unittest.TestCase):
    def test_known_values(self):
        # H=110, L=90, C=100 -> P=100, range=20.
        lv = compute_pivot_levels(110, 90, 100)
        self.assertEqual(lv["pivot"], 100.0)
        self.assertEqual(lv["r1"], 110.0)  # 2*100 - 90
        self.assertEqual(lv["s1"], 90.0)   # 2*100 - 110
        self.assertEqual(lv["r2"], 120.0)  # 100 + 20
        self.assertEqual(lv["s2"], 80.0)   # 100 - 20
        self.assertEqual(lv["r3"], 130.0)  # 110 + 2*(100-90)
        self.assertEqual(lv["s3"], 70.0)   # 90 - 2*(110-100)

    def test_ordering(self):
        lv = compute_pivot_levels(110, 90, 105)
        self.assertLess(lv["s3"], lv["s2"])
        self.assertLess(lv["s2"], lv["s1"])
        self.assertLess(lv["s1"], lv["pivot"])
        self.assertLess(lv["pivot"], lv["r1"])
        self.assertLess(lv["r1"], lv["r2"])
        self.assertLess(lv["r2"], lv["r3"])


class TestNearest(unittest.TestCase):
    def test_nearest_support_and_resistance(self):
        levels = [70, 80, 90, 100, 110, 120, 130]
        self.assertEqual(_nearest(levels, 105, below=True), 100)
        self.assertEqual(_nearest(levels, 105, below=False), 110)

    def test_none_when_price_beyond_all_levels(self):
        levels = [70, 80, 90, 100, 110, 120, 130]
        self.assertIsNone(_nearest(levels, 200, below=False))  # above everything
        self.assertIsNone(_nearest(levels, 50, below=True))    # below everything


class TestPivotLevelsFromHistory(unittest.TestCase):
    def _df(self, rows):
        return pd.DataFrame(rows)

    def test_from_history_uses_last_bar(self):
        df = self._df([
            {"date": "2026-01-01", "high": 50, "low": 40, "close": 45},
            {"date": "2026-01-02", "high": 110, "low": 90, "close": 100},
        ])
        res = pivot_levels_from_history(df)
        self.assertIsNotNone(res)
        self.assertEqual(res["pivot"], 100.0)
        self.assertEqual(res["current_price"], 100.0)
        self.assertEqual(res["basis_date"], "2026-01-02")
        self.assertEqual(res["period"], "daily")

    def test_nearest_levels_relative_to_close(self):
        # close 105 sits between pivot(100) and r1(110).
        df = self._df([{"high": 110, "low": 90, "close": 105}])
        res = pivot_levels_from_history(df)
        self.assertEqual(res["nearest_support"], res["pivot"])
        self.assertEqual(res["nearest_resistance"], res["r1"])

    def test_capitalized_columns_supported(self):
        df = self._df([{"High": 110, "Low": 90, "Close": 100}])
        res = pivot_levels_from_history(df)
        self.assertEqual(res["pivot"], 100.0)

    def test_none_on_empty(self):
        self.assertIsNone(pivot_levels_from_history(self._df([])))
        self.assertIsNone(pivot_levels_from_history(None))

    def test_none_on_missing_columns(self):
        df = self._df([{"open": 10, "volume": 100}])
        self.assertIsNone(pivot_levels_from_history(df))

    def test_none_when_high_below_low(self):
        df = self._df([{"high": 90, "low": 110, "close": 100}])
        self.assertIsNone(pivot_levels_from_history(df))

    def test_missing_date_is_tolerated(self):
        df = self._df([{"high": 110, "low": 90, "close": 100}])
        res = pivot_levels_from_history(df)
        self.assertIsNone(res["basis_date"])


if __name__ == "__main__":
    unittest.main()
