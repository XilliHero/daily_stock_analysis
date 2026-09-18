# -*- coding: utf-8 -*-
"""Deterministic ticker → asset class, and sector lookup with a safe default.

Known non-equity tickers are enumerated; everything else defaults to equity.
The lists are intentionally small and easy to extend.
"""

from __future__ import annotations

from typing import Dict, Optional

_CASH = {"CASH", "USD", "$CASH"}
_BOND_ETFS = {"BND", "AGG", "TLT", "IEF", "LQD", "BNDX", "SHY", "TIP", "VGSH", "GOVT"}
_COMMODITY_ETFS = {"CORN", "WEAT", "SOYB", "GLD", "SLV", "USO", "DBC", "UNG", "PDBC"}


def classify_asset(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    if t in _CASH:
        return "cash"
    if t in _BOND_ETFS:
        return "fixed_income"
    if t in _COMMODITY_ETFS:
        return "commodity"
    if t.endswith("-USD"):
        return "crypto"
    return "equity"


def sector_of(ticker: str, lookup: Optional[Dict[str, str]]) -> str:
    if not lookup:
        return "Unknown"
    return lookup.get((ticker or "").strip().upper(), "Unknown")
