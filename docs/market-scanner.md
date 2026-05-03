# Market Scanner — Instructions & Architecture Guide

## What Is It?

The Market Scanner is a team of AI-powered agents that automatically scans **~1,730 stocks** across the US and Canadian markets, filters them through technical and fundamental analysis, and produces a ranked list of top picks based on your chosen investment strategy.

No LLM calls are made during scanning — it is pure data processing using live market data from Yahoo Finance.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [How It Works](#2-how-it-works)
3. [The Agent Team](#3-the-agent-team)
4. [Strategy Profiles](#4-strategy-profiles)
5. [Screening Signals Explained](#5-screening-signals-explained)
6. [Fundamental Analysis](#6-fundamental-analysis)
7. [Sector Analysis](#7-sector-analysis)
8. [Scoring & Ranking](#8-scoring--ranking)
9. [Output & Reports](#9-output--reports)
10. [Configuration](#10-configuration)
11. [Architecture Diagram](#11-architecture-diagram)
12. [Files & Structure](#12-files--structure)
13. [Testing](#13-testing)
14. [FAQ](#14-faq)

---

## 1. Quick Start

### Run all 4 strategies on US + Canadian stocks:
```bash
python main.py --market-scan
```

### Run a single strategy:
```bash
python main.py --market-scan --scan-strategy value
```

### Run on US stocks only, top 20 picks:
```bash
python main.py --market-scan --scan-region us --scan-top-n 20
```

### Run multiple specific strategies:
```bash
python main.py --market-scan --scan-strategy growth,dividend
```

### All CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--market-scan` | off | Enable the market scanner |
| `--scan-strategy` | all four | Comma-separated: `value`, `growth`, `dividend`, `recovery` |
| `--scan-region` | `us_ca` | `us` (US only), `ca` (Canada only), `us_ca` (both) |
| `--scan-top-n` | `50` | Maximum number of top picks per strategy |

Reports are saved to `output/scans/scan_{strategy}_{timestamp}.md`.

---

## 2. How It Works

The scanner runs a **4-stage pipeline**, where each stage is handled by a specialized agent:

```
Stage 1: UNIVERSE         Build the stock list
              |
Stage 2: SCREENER         Fast technical screening (eliminates ~97% of stocks)
              |
Stage 3: FUNDAMENTAL      Deep financial analysis    (runs in parallel)
         + SECTOR          Sector momentum analysis   (runs in parallel)
              |
Stage 4: REPORT           Combine scores, rank, generate Markdown report
```

### Timing

- **Stage 1** (Universe): ~5-15 seconds — fetches index constituents from Wikipedia, cached for 24 hours
- **Stage 2** (Screener): ~2-5 minutes — downloads 40 days of price data for all stocks in batches of 50
- **Stage 3** (Fundamental + Sector): ~3-10 minutes — deep dives on ~50 shortlisted stocks + 11 sector ETFs
- **Stage 4** (Report): <1 second — pure computation and rendering

Total: typically **5-15 minutes** per strategy, depending on network speed and number of stocks.

---

## 3. The Agent Team

### 3.1 Universe Agent (`universe_agent.py`)

**Job**: Build the complete list of stocks to scan.

**Data sources**:
- **S&P 500** (large-cap US) — ~500 stocks
- **S&P 400** (mid-cap US) — ~400 stocks
- **S&P 600** (small-cap US) — ~600 stocks
- **TSX Composite** (Canadian large-cap) — ~230 stocks

**How it works**:
1. Fetches constituent tables from Wikipedia for each index
2. Extracts ticker, company name, sector, and industry
3. Assigns cap tiers: large ($10B+), mid ($2B-$10B), small ($300M-$2B)
4. Applies your filters (regions, sectors, cap tiers)
5. Caches results for 24 hours to avoid repeated Wikipedia requests

**Output**: A list of `StockEntry` objects (~1,730 stocks with no filters).

---

### 3.2 Screener Agent (`screener_agent.py`)

**Job**: Fast first-pass to reduce the universe to ~50 candidates.

**How it works**:
1. Downloads 40 days of daily OHLCV (Open, High, Low, Close, Volume) data via `yf.download()` in batches of 50 tickers
2. For each stock, computes up to **8 technical/valuation signals**
3. Stocks must trigger at least `min_signals` (default: 2) to qualify
4. Qualified stocks are scored and ranked; the top N are passed forward

**Output**: A ranked shortlist of `StockSignals` objects.

---

### 3.3 Fundamental Agent (`fundamental_agent.py`)

**Job**: Deep financial analysis for each shortlisted candidate.

**How it works**:
1. For each candidate, pulls detailed financials from `yf.Ticker().info`
2. Evaluates 5 dimensions: profitability, balance sheet, cash flow, growth, dividends
3. Applies strategy-specific scoring (value stocks scored differently than growth stocks)
4. Assigns a letter grade (A through F)

**Output**: A list of `FundamentalData` objects with scores and flags.

---

### 3.4 Sector Agent (`sector_agent.py`)

**Job**: Provide sector-level context — which sectors are hot, which are cold.

**How it works**:
1. Downloads 35 days of price data for 11 sector ETFs (XLK, XLV, XLF, etc.) plus SPY as benchmark
2. Computes weekly and monthly returns for each sector
3. Calculates relative strength vs SPY
4. Computes momentum score (0-100) based on moving average alignment
5. Ranks all 11 sectors and assigns each stock a sector context score

**Sector ETF mapping**:

| Sector | ETF |
|--------|-----|
| Technology | XLK |
| Healthcare | XLV |
| Financials | XLF |
| Energy | XLE |
| Consumer Cyclical | XLY |
| Consumer Defensive | XLP |
| Industrials | XLI |
| Basic Materials | XLB |
| Communication Services | XLC |
| Real Estate | XLRE |
| Utilities | XLU |

**Output**: Sector rankings + per-stock `SectorAnalysis` objects.

---

### 3.5 Report Agent (`report_agent.py`)

**Job**: Combine all agent outputs into a final ranked report.

**How it works**:
1. Takes outputs from Screener, Fundamental, and Sector agents
2. Computes a **composite score** for each stock using strategy-specific weights
3. Ranks stocks by composite score
4. Renders a Markdown report with summary tables and detail cards

**Output**: A `ScanReport` with Markdown text saved to disk.

---

## 4. Strategy Profiles

Each strategy defines **what signals matter**, **what thresholds to use**, and **how to weight each agent's contribution**.

### Value Strategy
> *Undervalued quality stocks for medium-to-long-term holds*

| Parameter | Value |
|-----------|-------|
| P/E max | 20.0 |
| P/B max | 1.5 |
| Drawdown from high | 15%+ |
| Min signals | 2 |
| **Weights**: Fundamental 40%, Screener 20%, Intel 20% | |

### Growth Strategy
> *Fast-growing companies with expanding revenue and margins*

| Parameter | Value |
|-----------|-------|
| Revenue growth min | 15% |
| RSI momentum zone | 50-80 |
| Min signals | 2 |
| Preferred sectors | Technology, Healthcare, Communication Services |
| **Weights**: Fundamental 30%, Intel 25%, Technical 20% | |

### Dividend Strategy
> *Stable income stocks with reliable dividend history*

| Parameter | Value |
|-----------|-------|
| Dividend yield min | 2.5% |
| Payout ratio max | 70% |
| Dividend years min | 5 |
| Min signals | 2 |
| Preferred sectors | Utilities, Consumer Defensive, Real Estate, Financials |
| **Weights**: Fundamental 40%, Screener 25%, Risk 20% | |

### Recovery Strategy
> *Beaten-down stocks with turnaround potential*

| Parameter | Value |
|-----------|-------|
| Drawdown from high | 25%+ |
| RSI oversold | 35 |
| Volume spike min | 1.3x |
| Min signals | 2 |
| **Weights**: Intel 30%, Technical 25%, Fundamental 25% | |

---

## 5. Screening Signals Explained

The Screener Agent checks each stock for these 8 signals:

### Volume Spike
- **What**: Today's volume is significantly higher than the 20-day average
- **Threshold**: volume_ratio >= 1.5x (configurable)
- **Why it matters**: Unusual volume often precedes major price moves

### Price Breakout
- **What**: Current price exceeds the highest high of the past N days
- **Threshold**: Lookback period defaults to 20 days
- **Why it matters**: Breaking above resistance can signal the start of a new trend

### Momentum Crossover
- **What**: 5-day moving average crosses above the 10-day moving average, and both are rising
- **Why it matters**: A classic trend-following signal indicating upward momentum

### Gap Up
- **What**: Today's open is significantly higher than yesterday's close
- **Threshold**: >= 2.0% gap (configurable)
- **Why it matters**: Gaps often indicate overnight news, earnings surprises, or institutional buying

### RSI (Relative Strength Index)
- **What**: 14-period RSI measures momentum on a 0-100 scale
- **Triggers**: Oversold (RSI < 30) OR in the momentum zone (55-75, configurable)
- **Why it matters**: Oversold = potential bounce; momentum zone = confirmed trend

### Drawdown from 52-Week High
- **What**: How far the stock has fallen from its highest point in the data window
- **Threshold**: Varies by strategy (15% for value, 25% for recovery)
- **Why it matters**: Large drawdowns may represent buying opportunities for contrarian strategies

### Low P/E and Low P/B
- **What**: Price-to-Earnings and Price-to-Book below thresholds
- **Threshold**: P/E <= 20, P/B <= 1.5 (for value strategy)
- **Why it matters**: Classic value metrics — low ratios may indicate undervaluation

### High Dividend Yield
- **What**: Annual dividend yield above threshold
- **Threshold**: >= 2.5% (for dividend strategy)
- **Why it matters**: High yield with sustainable payout = income opportunity

---

## 6. Fundamental Analysis

The Fundamental Agent evaluates each stock across 5 dimensions:

### Profitability
| Metric | Flag Threshold |
|--------|---------------|
| ROE (Return on Equity) | >20% = `high_roe` |
| Net Margin | >15% = `high_margin`, <0% = `negative_margin` |
| Gross Margin, Operating Margin, ROA | Tracked but no flags |

### Balance Sheet
| Metric | Flag Threshold |
|--------|---------------|
| Debt-to-Equity | >2.0 = `high_debt` |
| Current Ratio | <1.0 = `low_liquidity` |
| Net Cash Position | Cash > Debt = `net_cash` |

### Cash Flow
| Metric | Flag Threshold |
|--------|---------------|
| Free Cash Flow | >0 = `positive_fcf`, <0 = `negative_fcf` |
| Operating Cash Flow | Tracked |

### Growth
| Metric | Flag Threshold |
|--------|---------------|
| Revenue Growth (YoY) | >15% = `strong_revenue_growth` |
| Earnings Growth (YoY) | >20% = `strong_earnings_growth` |

### Dividend History
| Metric | Flag Threshold |
|--------|---------------|
| Consecutive dividend years | >=10 = `dividend_veteran`, >=5 = `dividend_consistent` |

### Grading Scale

| Grade | Score Range |
|-------|------------|
| A | 80-100 |
| B | 65-79 |
| C | 50-64 |
| D | 35-49 |
| F | 0-34 |

---

## 7. Sector Analysis

The Sector Agent ranks all 11 GICS sectors by momentum and assigns context to each stock.

### Momentum Score (0-100)

Five checks, 20 points each:
1. Current price > 5-day moving average
2. Current price > 10-day moving average
3. Current price > 20-day moving average
4. 5-day MA > 10-day MA
5. 10-day MA > 20-day MA

### Trend Classification
- **Bullish**: Momentum >= 70 AND positive weekly return
- **Bearish**: Momentum <= 30 AND negative weekly return
- **Neutral**: Everything else

### Stock Sector Score (base = 50)
- Bullish sector: +15 / Bearish: -15
- Top 3 ranked sector: +10 / Bottom 3: -10
- Relative strength vs SPY: +0.5 per percentage point
- In preferred sector (per strategy): +10
- In avoided sector: -10

---

## 8. Scoring & Ranking

The final composite score combines all agent outputs using **strategy-specific weights**.

### Formula
```
composite_score = screener_normalized x W_screener
                + fundamental_score   x W_fundamental
                + sector_score        x W_sector
```

### Normalization
- **Screener score**: Divided by 60, capped at 1.0, then scaled to 0-100
- **Fundamental score**: Already 0-100
- **Sector score**: Already 0-100

### Example (Value Strategy)

A stock with:
- Screener score: 45 -> normalized: min(45/60, 1.0) x 100 = 75
- Fundamental score: 82 (Grade A)
- Sector score: 65

Composite = 75 x 0.20 + 82 x 0.40 + 65 x 0.00 = 15 + 32.8 + 0 = **47.8**

(Note: The value strategy weights sector at 0.0, so sector score has no impact for value. Growth strategy weights sector at 0.10.)

---

## 9. Output & Reports

Each strategy produces a Markdown report saved to `output/scans/`.

### Report Structure

1. **Header**: Strategy name, description, date
2. **Pipeline Summary**: Universe size -> Screened -> Qualified -> Analyzed -> Top picks
3. **Sector Overview Table**: All 11 sectors ranked with weekly/monthly returns, relative strength, and trend
4. **Top Picks Table** (up to 20): Ticker, name, sector, price, change%, composite score, grade, top signals
5. **Detail Cards** (up to 10): Full breakdown per stock including all valuation metrics, flags, and component scores
6. **Errors Section**: Any failed tickers or data issues

### Example filename
```
output/scans/scan_value_20260503_1430.md
output/scans/scan_growth_20260503_1445.md
```

---

## 10. Configuration

### .env.example options
```bash
# Strategies: value, growth, dividend, recovery (comma-separated)
# MARKET_SCAN_STRATEGY=value,growth,dividend,recovery

# Region: us, ca, or us_ca
# MARKET_SCAN_REGION=us_ca

# Max picks per strategy
# MARKET_SCAN_TOP_N=50
```

These are currently used via CLI flags only. The `.env` entries are reserved for future scheduled scanning.

---

## 11. Architecture Diagram

```
                    main.py --market-scan
                           |
                    PipelineConfig
                    (strategy, region, top_n)
                           |
                  MarketScannerPipeline
                           |
          +----------------+----------------+
          |                                 |
     get_strategy()                   UniverseAgent
     (loads profile)                       |
          |                    Wikipedia S&P 500/400/600
          |                    + TSX Composite
          |                           |
          |                    ~1,730 StockEntry objects
          |                           |
          |                     ScreenerAgent
          |                    (yf.download batches)
          |                           |
          |                    8 signals computed
          |                    min_signals >= 2 filter
          |                           |
          |                    ~50 StockSignals (shortlist)
          |                           |
          |              +------------+------------+
          |              |                         |
          |       FundamentalAgent            SectorAgent
          |       (yf.Ticker.info)         (11 sector ETFs)
          |       (parallel thread)        (parallel thread)
          |              |                         |
          |       FundamentalData           SectorAnalysis
          |       (score + grade)           (trend + rank)
          |              |                         |
          |              +------------+------------+
          |                           |
          +-----> weights -----> ReportAgent
                                      |
                               composite_score
                               = w.screener * screener_norm
                               + w.fundamental * fund_score
                               + w.sector * sector_score
                                      |
                               Ranked top picks
                                      |
                               Markdown Report
                                      |
                            output/scans/scan_*.md
```

---

## 12. Files & Structure

```
src/scanner/
  __init__.py              Package exports
  strategy_profiles.py     4 strategy definitions + ScreeningCriteria + DecisionWeights
  universe_agent.py        Builds stock universe from index constituents
  screener_agent.py        Fast technical + valuation screening pass
  fundamental_agent.py     Deep financial analysis per candidate
  sector_agent.py          Sector ETF momentum + rotation analysis
  report_agent.py          Final ranking + Markdown report generation
  pipeline.py              Orchestrator (stages 1-4, parallel execution)

tests/
  test_market_scanner.py   16 offline unit tests

main.py                    CLI integration (--market-scan flags)
.env.example               Scanner config entries
```

---

## 13. Testing

### Run all scanner tests (offline, no network needed):
```bash
python -m pytest tests/test_market_scanner.py -v
```

### What is tested:
- All 4 strategy profiles exist and weights sum to 1.0
- RSI computation: bullish (>70), bearish (<30), short series (=50)
- ATR computation: basic case and short series
- Fundamental grading: A/B/C/D/F boundaries
- Sector momentum computation: perfect uptrend = 100
- Pipeline config defaults and custom values
- Report rendering with empty data

### What is NOT tested (requires live data):
- Actual yfinance downloads
- Wikipedia table parsing
- End-to-end pipeline execution

To run a live integration test:
```bash
python main.py --market-scan --scan-strategy value --scan-region us --scan-top-n 5
```

---

## 14. FAQ

**Q: Does this use AI/LLM?**
A: No. The scanner is pure data processing — it uses Yahoo Finance data and mathematical computations. No API keys or LLM calls are needed.

**Q: How much data does it download?**
A: ~1,730 stocks x 40 days of OHLCV in batches of 50, plus detailed info for ~50 shortlisted stocks, plus 12 ETFs. Total: a few hundred API calls to Yahoo Finance.

**Q: Can I add my own strategy?**
A: Yes. Add a new entry to `STRATEGY_PROFILES` in `src/scanner/strategy_profiles.py` with your own `ScreeningCriteria` and `DecisionWeights`.

**Q: Can I scan only specific sectors?**
A: Yes, via `--scan-region` for geography, and programmatically via `include_sectors`/`exclude_sectors` in `PipelineConfig`. CLI sector filtering can be added as a future enhancement.

**Q: What if Yahoo Finance rate-limits me?**
A: The scanner batches downloads (50 at a time) and handles failures gracefully — individual ticker failures are logged but don't stop the pipeline.

**Q: Where are the reports saved?**
A: `output/scans/scan_{strategy}_{YYYYMMDD_HHMM}.md`

**Q: Can I run this on a schedule?**
A: Not yet via the built-in scheduler, but you can use cron or Task Scheduler:
```bash
# Daily at 5 PM EST (after market close)
0 17 * * 1-5 cd /path/to/daily_stock_analysis && python main.py --market-scan
```
