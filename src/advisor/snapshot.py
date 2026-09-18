# -*- coding: utf-8 -*-
"""Current portfolio picture — the single place that decides rebalance vs deploy.

Money formula (both modes): base = holdings_value + snapshot_cash + investable_cash.
Deploy-from-cash is the degenerate case where holdings and snapshot cash are 0.
Reads existing PortfolioService.get_portfolio_snapshot(); a fake is injectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.advisor.classify import classify_asset, sector_of


@dataclass
class Position:
    symbol: str
    asset_class: str
    sector: str
    value: float


@dataclass
class CurrentPicture:
    mode: str                      # "rebalance" | "deploy"
    base: float                    # total investable base in profile currency
    cash: float                    # snapshot cash + new investable cash
    positions: List[Position] = field(default_factory=list)


def get_current(profile, portfolio_service=None,
                sector_lookup: Optional[Dict[str, str]] = None) -> CurrentPicture:
    svc = portfolio_service
    if svc is None:
        from src.services.portfolio_service import PortfolioService
        svc = PortfolioService()

    snap = svc.get_portfolio_snapshot() or {}
    snapshot_cash = float(snap.get("total_cash", 0.0) or 0.0)

    positions: List[Position] = []
    holdings_value = 0.0
    for account in snap.get("accounts", []) or []:
        for row in account.get("positions", []) or []:
            symbol = str(row.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            value = float(row.get("market_value_base", 0.0) or 0.0)
            positions.append(Position(
                symbol=symbol,
                asset_class=classify_asset(symbol),
                sector=sector_of(symbol, sector_lookup),
                value=value,
            ))
            holdings_value += value

    new_cash = float(profile.investable_cash or 0.0)
    base = holdings_value + snapshot_cash + new_cash
    mode = "rebalance" if positions else "deploy"
    return CurrentPicture(mode=mode, base=base, cash=snapshot_cash + new_cash, positions=positions)
