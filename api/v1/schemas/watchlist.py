# -*- coding: utf-8 -*-
"""
===================================
Watchlist 相关模型
===================================

职责：
1. 定义自选股代码校验结果模型
"""

from typing import Optional

from pydantic import BaseModel, Field


class ValidateSymbolResponse(BaseModel):
    """Result of validating a ticker symbol against market data."""

    symbol: str = Field(..., description="Normalized ticker symbol")
    valid: bool = Field(..., description="Whether the symbol returned usable price data")
    price: Optional[float] = Field(None, description="Latest close, when available")

    class Config:
        json_schema_extra = {
            "example": {"symbol": "NVDA", "valid": True, "price": 132.45}
        }
