"""
Portfolio Risk Module
=====================
Computes:
  - Beta           : each stock's volatility relative to ^GSPC (90-day window)
  - Alpha (Jensen's): excess return vs. benchmark over 90 days
  - Correlation matrix : pairwise daily-return correlations across all holdings
  - Portfolio-level risk summary : annualized vol, weighted beta, diversification flags

All calculations use equal weighting across the watchlist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from stock_analysis.config import (
    HIGH_BETA_THRESHOLD,
    HIGH_CORRELATION_THRESHOLD,
    HIGH_VOLATILITY_THRESHOLD,
    RISK_FREE_RATE_ANNUAL,
    SECTOR_MAP,
    WATCHLIST,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TickerRisk:
    ticker: str
    beta: float
    alpha: float          # Jensen's alpha (period, not annualised)
    alpha_annualised: float
    return_90d: float     # cumulative return over the window
    volatility: float     # annualised individual volatility
    n_days: int           # actual number of trading days used


@dataclass
class CorrelationFlag:
    ticker_a: str
    ticker_b: str
    correlation: float


@dataclass
class PortfolioRiskSummary:
    portfolio_volatility: float          # annualised
    portfolio_beta: float                # equal-weighted
    portfolio_return_90d: float          # equal-weighted cumulative return
    high_corr_pairs: list[CorrelationFlag] = field(default_factory=list)
    sector_concentration_flag: bool = False
    high_volatility_flag: bool = False
    high_beta_flag: bool = False
    flags: list[str] = field(default_factory=list)


@dataclass
class RiskReport:
    ticker_risks: dict[str, TickerRisk]
    correlation_matrix: pd.DataFrame
    portfolio: PortfolioRiskSummary
    lookback_days: int


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

def _daily_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return prices.pct_change().dropna()


def _compute_beta_alpha(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    rf_annual: float = RISK_FREE_RATE_ANNUAL,
) -> tuple[float, float, float, float, int]:
    """
    Returns (beta, alpha_period, alpha_annual, cumulative_stock_return, n_days).

    Beta   = Cov(R_i, R_m) / Var(R_m)
    Alpha  = R_i_period - [Rf_period + Beta * (R_m_period - Rf_period)]
    """
    combined = pd.concat(
        [stock_returns.rename("stock"), market_returns.rename("market")], axis=1
    ).dropna()

    if len(combined) < 20:
        logger.warning("Too few data points (%d) for reliable beta estimation.", len(combined))

    s = combined["stock"].values
    m = combined["market"].values
    n = len(combined)

    # Beta via covariance matrix
    cov_matrix = np.cov(s, m, ddof=1)
    var_market = cov_matrix[1, 1]
    beta = cov_matrix[0, 1] / var_market if var_market != 0 else 1.0

    # Cumulative returns over the actual period
    r_stock = float(np.prod(1 + s) - 1)
    r_market = float(np.prod(1 + m) - 1)

    # Risk-free rate scaled to the period length (in trading days)
    rf_period = float((1 + rf_annual) ** (n / 252) - 1)

    # Jensen's Alpha
    alpha_period = r_stock - (rf_period + beta * (r_market - rf_period))

    # Annualise alpha for comparability
    alpha_annual = float((1 + alpha_period) ** (252 / n) - 1)

    return beta, alpha_period, alpha_annual, r_stock, n


def _compute_ticker_volatility(returns: pd.Series) -> float:
    """Annualised standard deviation of daily returns."""
    return float(returns.std(ddof=1) * np.sqrt(252))


def _compute_correlation_matrix(stock_returns: pd.DataFrame) -> pd.DataFrame:
    return stock_returns.corr(method="pearson")


def _find_high_corr_pairs(
    corr: pd.DataFrame,
    threshold: float = HIGH_CORRELATION_THRESHOLD,
) -> list[CorrelationFlag]:
    pairs: list[CorrelationFlag] = []
    tickers = list(corr.columns)
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            c = corr.iloc[i, j]
            if abs(c) >= threshold:
                pairs.append(CorrelationFlag(tickers[i], tickers[j], float(c)))
    return sorted(pairs, key=lambda x: abs(x.correlation), reverse=True)


def _check_sector_concentration(tickers: list[str]) -> bool:
    """Flag if any single sector has more than 2 holdings."""
    counts: dict[str, int] = {}
    for t in tickers:
        sector = SECTOR_MAP.get(t, "Unknown")
        counts[sector] = counts.get(sector, 0) + 1
    return any(v > 2 for v in counts.values())


def _compute_portfolio_metrics(
    stock_returns: pd.DataFrame,
    ticker_risks: dict[str, TickerRisk],
) -> PortfolioRiskSummary:
    """Equal-weighted portfolio-level risk metrics."""
    tickers = list(stock_returns.columns)
    n = len(tickers)
    weights = np.full(n, 1.0 / n)

    # Equal-weighted daily returns
    port_returns = stock_returns[tickers].dot(weights)

    # Annualised portfolio volatility
    port_vol = float(port_returns.std(ddof=1) * np.sqrt(252))

    # Equal-weighted beta
    port_beta = float(np.mean([ticker_risks[t].beta for t in tickers if t in ticker_risks]))

    # Equal-weighted 90-day cumulative return
    port_return_90d = float(np.prod(1 + port_returns) - 1)

    # Flags
    corr = _compute_correlation_matrix(stock_returns)
    high_corr_pairs = _find_high_corr_pairs(corr)
    sector_conc = _check_sector_concentration(tickers)
    high_vol = port_vol >= HIGH_VOLATILITY_THRESHOLD
    high_beta_flag = port_beta >= HIGH_BETA_THRESHOLD

    flags: list[str] = []
    if high_corr_pairs:
        pair_strs = ", ".join(f"{p.ticker_a}/{p.ticker_b} ({p.correlation:+.2f})" for p in high_corr_pairs)
        flags.append(f"High correlation detected: {pair_strs}")
    if sector_conc:
        flags.append("Sector concentration: more than 2 holdings in the same sector")
    if high_vol:
        flags.append(f"High portfolio volatility: {port_vol:.1%} annualised (threshold {HIGH_VOLATILITY_THRESHOLD:.0%})")
    if high_beta_flag:
        flags.append(f"High portfolio beta: {port_beta:.2f} (threshold {HIGH_BETA_THRESHOLD})")

    return PortfolioRiskSummary(
        portfolio_volatility=port_vol,
        portfolio_beta=port_beta,
        portfolio_return_90d=port_return_90d,
        high_corr_pairs=high_corr_pairs,
        sector_concentration_flag=sector_conc,
        high_volatility_flag=high_vol,
        high_beta_flag=high_beta_flag,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_risk_analysis(
    stock_prices: pd.DataFrame,
    market_prices: pd.Series,
    rf_annual: float = RISK_FREE_RATE_ANNUAL,
) -> RiskReport:
    """
    Main entry point.

    Parameters
    ----------
    stock_prices  : DataFrame  — adjusted close prices, one column per ticker
    market_prices : Series     — ^GSPC adjusted close prices (same date range)
    rf_annual     : float      — annual risk-free rate

    Returns
    -------
    RiskReport dataclass with all computed metrics.
    """
    stock_returns = _daily_returns(stock_prices)
    market_returns = _daily_returns(market_prices)

    ticker_risks: dict[str, TickerRisk] = {}

    for ticker in stock_prices.columns:
        if ticker not in stock_returns.columns:
            continue
        try:
            beta, alpha_p, alpha_a, r90, n = _compute_beta_alpha(
                stock_returns[ticker], market_returns, rf_annual
            )
            vol = _compute_ticker_volatility(stock_returns[ticker].dropna())
            ticker_risks[ticker] = TickerRisk(
                ticker=ticker,
                beta=beta,
                alpha=alpha_p,
                alpha_annualised=alpha_a,
                return_90d=r90,
                volatility=vol,
                n_days=n,
            )
        except Exception as exc:
            logger.error("Risk calculation failed for %s: %s", ticker, exc)

    # Restrict returns to tickers that succeeded
    valid_tickers = [t for t in stock_prices.columns if t in ticker_risks]
    stock_returns_valid = stock_returns[valid_tickers].dropna()

    corr_matrix = _compute_correlation_matrix(stock_returns_valid)
    portfolio = _compute_portfolio_metrics(stock_returns_valid, ticker_risks)

    logger.info(
        "Risk analysis complete. Portfolio vol=%.1f%% beta=%.2f",
        portfolio.portfolio_volatility * 100,
        portfolio.portfolio_beta,
    )

    return RiskReport(
        ticker_risks=ticker_risks,
        correlation_matrix=corr_matrix,
        portfolio=portfolio,
        lookback_days=len(stock_returns_valid),
    )
