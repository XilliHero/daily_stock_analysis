# -*- coding: utf-8 -*-
"""
===================================
自选股（Watchlist）接口
===================================

职责：
1. GET /api/v1/watchlist/validate 校验股票代码是否有效

校验使用与扫描器一致的 yfinance 路径（fetch_price_history），
因此对美股 / 加股（.TO）/ 加密货币（BTC-USD）等格式均可正确识别。
"""

import logging

from fastapi import APIRouter, Query

from api.v1.schemas.watchlist import ValidateSymbolResponse
from src.scanner.screener_agent import fetch_price_history

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/validate",
    response_model=ValidateSymbolResponse,
    summary="校验自选股代码",
    description="通过 yfinance 检查代码是否返回有效行情数据（与扫描器一致）。",
)
def validate_symbol(
    symbol: str = Query(..., description="Ticker symbol, e.g. NVDA, BTC-USD, MDA.TO"),
) -> ValidateSymbolResponse:
    """Return whether ``symbol`` resolves to usable price data.

    Uses a ``def`` handler so FastAPI runs the blocking yfinance call in its
    threadpool. Any failure is treated as "not valid" rather than an error, so
    the client always gets a clean yes/no.
    """
    normalized = (symbol or "").strip().upper()
    if not normalized:
        return ValidateSymbolResponse(symbol="", valid=False, price=None)

    try:
        history = fetch_price_history([normalized])
        df = history.get(normalized)
        if df is None or df.empty:
            return ValidateSymbolResponse(symbol=normalized, valid=False, price=None)
        close = df["Close"].dropna()
        price = round(float(close.iloc[-1]), 2) if not close.empty else None
        return ValidateSymbolResponse(symbol=normalized, valid=price is not None, price=price)
    except Exception as exc:  # never surface as a 500 — just "not valid"
        logger.warning("[watchlist] validation failed for %s: %s", normalized, exc)
        return ValidateSymbolResponse(symbol=normalized, valid=False, price=None)
