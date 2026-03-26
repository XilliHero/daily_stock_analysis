"""
Portfolio Risk Module
=====================
Computes 90-day rolling risk metrics for an equal-weighted watchlist:

  - Beta          : each stock's volatility relative to ^GSPC
  - Alpha         : Jensen's alpha (excess return vs. benchmark)
  - Correlation   : pairwise daily-return correlations
  - Portfolio     : equal-weighted vol, beta, return, and risk flags

Called by morning_briefing.py; no external dependencies beyond
numpy, pandas, and yfinance (already in requirements.txt).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger("portfolio_risk")

# ── Thresholds ──────────────────────────────────────────────────────────────
RISK_FREE_RATE_ANNUAL: float = 0.05   # approximate 3-month T-bill
HIGH_CORR_THRESHOLD:   float = 0.75   # flag pairs above this
HIGH_VOL_THRESHOLD:    float = 0.40   # flag portfolio annualised vol above this
HIGH_BETA_THRESHOLD:   float = 1.5    # flag portfolio beta above this

# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class TickerRisk:
    ticker:        str
    beta:          float   # vs. ^GSPC
    alpha_annual:  float   # Jensen's alpha, annualised
    return_90d:    float   # cumulative return over the window
    volatility:    float   # annualised individual volatility
    n_days:        int     # actual trading days used


@dataclass
class HighCorrPair:
    ticker_a:    str
    ticker_b:    str
    correlation: float


@dataclass
class PortfolioRisk:
    portfolio_volatility: float   # annualised equal-weighted
    portfolio_beta:       float   # equal-weighted
    portfolio_return_90d: float   # equal-weighted cumulative return
    high_corr_pairs:      list[HighCorrPair] = field(default_factory=list)
    flags:                list[str]          = field(default_factory=list)


@dataclass
class RiskData:
    ticker_risks:       dict[str, TickerRisk]
    correlation_matrix: pd.DataFrame
    portfolio:          PortfolioRisk
    watchlist:          list[str]
    error:              Optional[str] = None


# ── Internal helpers ─────────────────────────────────────────────────────────

def _daily_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return prices.pct_change().dropna()


def _beta_alpha(
    stock_ret: pd.Series,
    market_ret: pd.Series,
    rf_annual: float = RISK_FREE_RATE_ANNUAL,
) -> tuple[float, float, float, int]:
    """Return (beta, alpha_annual, cumulative_stock_return, n_days)."""
    combined = pd.concat(
        [stock_ret.rename("s"), market_ret.rename("m")], axis=1
    ).dropna()
    n = len(combined)
    if n < 10:
        return 1.0, 0.0, 0.0, n

    s = combined["s"].values
    m = combined["m"].values

    cov  = np.cov(s, m, ddof=1)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0

    r_s  = float(np.prod(1 + s) - 1)
    r_m  = float(np.prod(1 + m) - 1)
    rf_p = float((1 + rf_annual) ** (n / 252) - 1)

    alpha_period = r_s - (rf_p + beta * (r_m - rf_p))
    # Annualise: (1 + alpha_period)^(252/n) - 1
    alpha_annual = float((1 + alpha_period) ** (252 / max(n, 1)) - 1)

    return float(beta), alpha_annual, r_s, n


def _annualised_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(252))


def _high_corr_pairs(corr: pd.DataFrame, threshold: float) -> list[HighCorrPair]:
    tickers = list(corr.columns)
    pairs: list[HighCorrPair] = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            c = float(corr.iloc[i, j])
            if abs(c) >= threshold:
                pairs.append(HighCorrPair(tickers[i], tickers[j], c))
    return sorted(pairs, key=lambda x: abs(x.correlation), reverse=True)


# ── Public entry point ───────────────────────────────────────────────────────

def compute_risk(
    watchlist:  list[str],
    benchmark:  str   = "^GSPC",
    lookback:   int   = 90,
    rf_annual:  float = RISK_FREE_RATE_ANNUAL,
) -> RiskData:
    """
    Fetch price history and compute portfolio risk metrics.

    Parameters
    ----------
    watchlist : stock tickers (Yahoo Finance format)
    benchmark : benchmark ticker for beta/alpha, default ^GSPC
    lookback  : calendar days to fetch (yields ~63 trading days for 90 cal days)
    rf_annual : annual risk-free rate

    Returns
    -------
    RiskData  — fully populated, or with .error set on failure.
    """
    all_tickers = watchlist + [benchmark]
    try:
        raw = yf.download(
            all_tickers,
            period=f"{lookback}d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        log.error("yfinance download failed: %s", exc)
        return RiskData(
            ticker_risks={},
            correlation_matrix=pd.DataFrame(),
            portfolio=PortfolioRisk(0.0, 0.0, 0.0),
            watchlist=watchlist,
            error=str(exc),
        )

    # Normalise to a Close-price DataFrame
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, :len(all_tickers)]
    else:
        close = raw[["Close"]] if "Close" in raw.columns else raw

    if close.empty:
        return RiskData(
            ticker_risks={},
            correlation_matrix=pd.DataFrame(),
            portfolio=PortfolioRisk(0.0, 0.0, 0.0),
            watchlist=watchlist,
            error="No price data returned",
        )

    # Split benchmark from watchlist prices
    market_prices = close[benchmark].dropna() if benchmark in close.columns else pd.Series(dtype=float)
    stock_prices  = close[[t for t in watchlist if t in close.columns]].dropna(how="all")

    if market_prices.empty or stock_prices.empty:
        return RiskData(
            ticker_risks={},
            correlation_matrix=pd.DataFrame(),
            portfolio=PortfolioRisk(0.0, 0.0, 0.0),
            watchlist=watchlist,
            error="Insufficient price data",
        )

    market_ret = _daily_returns(market_prices)
    stock_ret  = _daily_returns(stock_prices)

    # ── Per-ticker metrics ────────────────────────────────────────────────
    ticker_risks: dict[str, TickerRisk] = {}
    for ticker in stock_prices.columns:
        if ticker not in stock_ret.columns:
            continue
        try:
            beta, alpha_a, r90, n = _beta_alpha(stock_ret[ticker], market_ret, rf_annual)
            vol = _annualised_vol(stock_ret[ticker].dropna())
            ticker_risks[ticker] = TickerRisk(
                ticker=ticker,
                beta=beta,
                alpha_annual=alpha_a,
                return_90d=r90,
                volatility=vol,
                n_days=n,
            )
        except Exception as exc:
            log.warning("Risk calc failed for %s: %s", ticker, exc)

    valid = [t for t in stock_prices.columns if t in ticker_risks]
    if not valid:
        return RiskData(
            ticker_risks={},
            correlation_matrix=pd.DataFrame(),
            portfolio=PortfolioRisk(0.0, 0.0, 0.0),
            watchlist=watchlist,
            error="Risk calculation failed for all tickers",
        )

    valid_ret  = stock_ret[valid].dropna()
    corr       = valid_ret.corr(method="pearson")

    # ── Portfolio-level metrics ───────────────────────────────────────────
    n_stocks    = len(valid)
    weights     = np.full(n_stocks, 1.0 / n_stocks)
    port_ret    = valid_ret[valid].dot(weights)

    port_vol    = float(port_ret.std(ddof=1) * np.sqrt(252))
    port_beta   = float(np.mean([ticker_risks[t].beta for t in valid]))
    port_r90    = float(np.prod(1 + port_ret) - 1)

    high_pairs  = _high_corr_pairs(corr, HIGH_CORR_THRESHOLD)

    flags: list[str] = []
    if high_pairs:
        pairs_str = ", ".join(
            f"{p.ticker_a}/{p.ticker_b} ({p.correlation:+.2f})" for p in high_pairs
        )
        flags.append(f"High correlation: {pairs_str}")
    if port_vol >= HIGH_VOL_THRESHOLD:
        flags.append(f"High portfolio volatility: {port_vol:.1%} annualised")
    if port_beta >= HIGH_BETA_THRESHOLD:
        flags.append(f"High portfolio beta: {port_beta:.2f}")

    portfolio = PortfolioRisk(
        portfolio_volatility=port_vol,
        portfolio_beta=port_beta,
        portfolio_return_90d=port_r90,
        high_corr_pairs=high_pairs,
        flags=flags,
    )

    log.info(
        "Risk computed. vol=%.1f%% beta=%.2f 90d_ret=%+.1f%% flags=%d",
        port_vol * 100, port_beta, port_r90 * 100, len(flags),
    )

    return RiskData(
        ticker_risks=ticker_risks,
        correlation_matrix=corr,
        portfolio=portfolio,
        watchlist=valid,
    )
