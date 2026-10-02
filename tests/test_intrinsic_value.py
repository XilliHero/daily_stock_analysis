# -*- coding: utf-8 -*-
"""Tests for the intrinsic-value estimator (DCF + Graham cross-check).

Pure-arithmetic unit tests — no network. Verify the DCF discounting, the
Graham formulas, conservative growth clamping, headline/fallback selection,
margin-of-safety verdicts and graceful handling of missing/unusable data.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.tools.intrinsic_value import (  # noqa: E402
    compute_intrinsic_value,
    _dcf_per_share,
    _graham_number,
    _graham_revised,
)


class TestDcfPerShare(unittest.TestCase):
    def test_known_dcf_value(self):
        # FCF 100, no growth, r=10%, tg=2.5%, 10y, no net debt, 100 shares.
        # PV(10y annuity of 100 @10%) + PV(terminal) / 100 shares.
        val = _dcf_per_share(
            fcf=100.0, shares=100.0, growth=0.0,
            discount_rate=0.10, terminal_growth=0.025, years=10, net_debt=0.0,
        )
        # Sum_{t=1..10} 100/1.1^t = 614.46; terminal = 100*1.025/0.075 = 1366.67,
        # discounted /1.1^10 (=2.5937) = 526.90; total 1141.36 / 100 = 11.41.
        self.assertAlmostEqual(val, 11.41, places=1)

    def test_net_debt_reduces_value(self):
        no_debt = _dcf_per_share(100.0, 100.0, 0.0, 0.10, 0.025, 10, 0.0)
        with_debt = _dcf_per_share(100.0, 100.0, 0.0, 0.10, 0.025, 10, 500.0)
        self.assertLess(with_debt, no_debt)
        self.assertAlmostEqual(no_debt - with_debt, 5.0, places=2)  # 500/100 shares

    def test_negative_fcf_returns_none(self):
        self.assertIsNone(_dcf_per_share(-50.0, 100.0, 0.0, 0.10, 0.025, 10, 0.0))

    def test_missing_shares_returns_none(self):
        self.assertIsNone(_dcf_per_share(100.0, None, 0.0, 0.10, 0.025, 10, 0.0))

    def test_discount_below_terminal_growth_returns_none(self):
        # r <= tg would make the terminal value diverge.
        self.assertIsNone(_dcf_per_share(100.0, 100.0, 0.0, 0.02, 0.025, 10, 0.0))

    def test_net_debt_exceeding_value_returns_none(self):
        self.assertIsNone(_dcf_per_share(100.0, 100.0, 0.0, 0.10, 0.025, 10, 1e9))


class TestGrahamFormulas(unittest.TestCase):
    def test_graham_number(self):
        # sqrt(22.5 * 5 * 20) = sqrt(2250) = 47.43
        self.assertAlmostEqual(_graham_number(5.0, 20.0), 47.43, places=2)

    def test_graham_number_requires_positive_inputs(self):
        self.assertIsNone(_graham_number(-1.0, 20.0))
        self.assertIsNone(_graham_number(5.0, 0.0))
        self.assertIsNone(_graham_number(None, 20.0))

    def test_graham_revised(self):
        # 5 * (8.5 + 2*10) = 5 * 28.5 = 142.5  (growth 0.10 -> 10 pts)
        self.assertAlmostEqual(_graham_revised(5.0, 0.10), 142.5, places=2)

    def test_graham_revised_requires_positive_eps(self):
        self.assertIsNone(_graham_revised(0.0, 0.10))
        self.assertIsNone(_graham_revised(None, 0.10))


class TestComputeIntrinsicValue(unittest.TestCase):
    def _base_info(self, **overrides):
        info = {
            "currentPrice": 10.0,
            "sharesOutstanding": 100.0,
            "freeCashflow": 100.0,
            "trailingEps": 5.0,
            "bookValue": 20.0,
            "earningsGrowth": 0.0,
            "totalDebt": 0.0,
            "totalCash": 0.0,
        }
        info.update(overrides)
        return info

    def test_headline_is_dcf_when_available(self):
        result = compute_intrinsic_value(self._base_info())
        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "dcf")
        self.assertAlmostEqual(result["fair_value"], 11.41, places=1)
        self.assertIsNotNone(result["graham"])  # cross-check still present

    def test_growth_is_clamped_to_cap(self):
        # earningsGrowth 0.80 must be capped to 0.10 (conservative).
        result = compute_intrinsic_value(self._base_info(earningsGrowth=0.80))
        self.assertEqual(result["assumptions"]["growth_rate_pct"], 10.0)

    def test_negative_growth_clamped_to_zero(self):
        result = compute_intrinsic_value(self._base_info(earningsGrowth=-0.20))
        self.assertEqual(result["assumptions"]["growth_rate_pct"], 0.0)

    def test_fallback_growth_when_missing(self):
        info = self._base_info()
        del info["earningsGrowth"]
        result = compute_intrinsic_value(info)
        self.assertEqual(result["assumptions"]["growth_rate_pct"], 5.0)

    def test_verdict_undervalued(self):
        # fair ~11.41 vs price 5 -> undervalued, positive upside.
        result = compute_intrinsic_value(self._base_info(currentPrice=5.0))
        self.assertEqual(result["verdict"], "undervalued")
        self.assertGreater(result["upside_pct"], 0)
        self.assertGreater(result["margin_of_safety_pct"], 0)

    def test_verdict_overvalued(self):
        result = compute_intrinsic_value(self._base_info(currentPrice=100.0))
        self.assertEqual(result["verdict"], "overvalued")
        self.assertLess(result["upside_pct"], 0)

    def test_verdict_fair_within_band(self):
        result = compute_intrinsic_value(self._base_info(currentPrice=11.41))
        self.assertEqual(result["verdict"], "fair")

    def test_graham_fallback_when_no_fcf(self):
        # No positive FCF -> headline falls back to Graham Number.
        result = compute_intrinsic_value(self._base_info(freeCashflow=None))
        self.assertEqual(result["method"], "graham")
        self.assertAlmostEqual(result["fair_value"], 47.43, places=2)

    def test_agreement_flag(self):
        # DCF ~11.41 vs Graham 47.43 -> diverge.
        diverge = compute_intrinsic_value(self._base_info())
        self.assertEqual(diverge["agreement"], "diverge")

    def test_none_when_nothing_computable(self):
        # No FCF, no EPS/book value -> nothing to estimate.
        self.assertIsNone(compute_intrinsic_value({"currentPrice": 10.0}))

    def test_empty_info_returns_none(self):
        self.assertIsNone(compute_intrinsic_value({}))
        self.assertIsNone(compute_intrinsic_value(None))

    def test_missing_price_still_returns_fair_value(self):
        info = self._base_info()
        del info["currentPrice"]
        result = compute_intrinsic_value(info)
        self.assertIsNotNone(result["fair_value"])
        self.assertIsNone(result["verdict"])
        self.assertIsNone(result["current_price"])

    def test_assumptions_reported(self):
        result = compute_intrinsic_value(self._base_info())
        a = result["assumptions"]
        self.assertEqual(a["discount_rate_pct"], 10.0)
        self.assertEqual(a["terminal_growth_pct"], 2.5)
        self.assertEqual(a["projection_years"], 10)


if __name__ == "__main__":
    unittest.main()
