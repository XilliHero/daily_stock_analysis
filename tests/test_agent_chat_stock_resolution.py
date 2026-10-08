# -*- coding: utf-8 -*-
"""Tests for resolving the ticker a chat answer is about (for metric cards)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.v1.endpoints.agent import _resolved_stock_code  # noqa: E402


class TestResolvedStockCode(unittest.TestCase):
    def test_prefers_get_stock_info_over_other_tools(self):
        log = [
            {"tool": "get_realtime_quote", "arguments": {"stock_code": "AAPL"}},
            {"tool": "analyze_trend", "arguments": {"stock_code": "AAPL"}},
            {"tool": "get_stock_info", "arguments": {"stock_code": "AAPL"}},
        ]
        self.assertEqual(_resolved_stock_code(log), "AAPL")

    def test_falls_back_to_analyze_trend(self):
        log = [{"tool": "analyze_trend", "arguments": {"stock_code": "CNQ.TO"}}]
        self.assertEqual(_resolved_stock_code(log), "CNQ.TO")

    def test_any_tool_stock_code_when_no_priority_tool(self):
        log = [{"tool": "some_other_tool", "arguments": {"stock_code": "600519"}}]
        self.assertEqual(_resolved_stock_code(log), "600519")

    def test_context_used_when_no_tool_calls(self):
        self.assertEqual(
            _resolved_stock_code([], context={"stock_code": "TSLA"}),
            "TSLA",
        )

    def test_tool_log_wins_over_context(self):
        log = [{"tool": "get_stock_info", "arguments": {"stock_code": "NKE"}}]
        self.assertEqual(
            _resolved_stock_code(log, context={"stock_code": "TSLA"}),
            "NKE",
        )

    def test_none_when_nothing_available(self):
        self.assertIsNone(_resolved_stock_code([]))
        self.assertIsNone(_resolved_stock_code(None))
        self.assertIsNone(_resolved_stock_code([{"tool": "get_skills", "arguments": {}}]))


if __name__ == "__main__":
    unittest.main()
