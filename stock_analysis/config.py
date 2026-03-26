"""
Central configuration for the stock analysis system.
"""

# Watchlist of tickers to track
WATCHLIST = ["TSLA", "OKLO", "XLE", "BTC-USD", "MDA.TO"]

# Benchmark index for beta/alpha calculations
BENCHMARK = "^GSPC"

# Annual risk-free rate (approximate 3-month T-bill)
RISK_FREE_RATE_ANNUAL = 0.05

# Rolling window for risk calculations (trading days)
LOOKBACK_DAYS = 90

# Risk thresholds
HIGH_CORRELATION_THRESHOLD = 0.75  # flag pairs above this
HIGH_VOLATILITY_THRESHOLD = 0.40   # 40% annualized — flag portfolio above this
HIGH_BETA_THRESHOLD = 1.5          # flag individual stocks above this

# Sector classification for concentration analysis
SECTOR_MAP = {
    "TSLA":    "Consumer Discretionary",
    "OKLO":    "Utilities / Nuclear",
    "XLE":     "Energy",
    "BTC-USD": "Crypto",
    "MDA.TO":  "Industrials / Space",
}
