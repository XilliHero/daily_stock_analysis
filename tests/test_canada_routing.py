# -*- coding: utf-8 -*-
"""Canadian (.TO / .V) ticker routing — deterministic, offline.

Regression guard for the fix that stopped .TO tickers being mis-classified as
Chinese A-shares (and mangled to `.SZ`).
"""

from data_provider.base import _is_canada_market, _is_hk_market
from data_provider.us_index_mapping import is_us_stock_code
from data_provider.yfinance_fetcher import YfinanceFetcher

_f = YfinanceFetcher()


def test_canada_market_detection():
    assert _is_canada_market("BNS.TO") is True
    assert _is_canada_market("MDA.TO") is True
    assert _is_canada_market("shop.v") is True  # TSX Venture, case-insensitive
    assert _is_canada_market("AAPL") is False
    assert _is_canada_market("600519") is False
    assert _is_canada_market("00700.HK") is False


def test_canada_symbol_passes_through_unmangled():
    # The bug: these became BNS.TO.SZ etc. Now they must pass through.
    assert _f._convert_stock_code("BNS.TO") == "BNS.TO"
    assert _f._convert_stock_code("MDA.TO") == "MDA.TO"
    assert _f._convert_stock_code("SHOP.TO") == "SHOP.TO"


def test_other_markets_unaffected():
    assert _f._convert_stock_code("AAPL") == "AAPL"            # US unchanged
    assert _f._convert_stock_code("600519") == "600519.SS"     # Shanghai unchanged
    assert _f._convert_stock_code("000001") == "000001.SZ"     # Shenzhen unchanged
    # Canada must not be mistaken for HK
    assert _is_hk_market("BNS.TO") is False
    # US detection must not swallow Canadian tickers
    assert is_us_stock_code("BNS.TO") is False
