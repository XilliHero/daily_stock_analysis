"""
Daily Stock Analysis — Main Orchestrator
=========================================
Entry point for the daily workflow.

Run manually:
    python main.py

Or triggered by GitHub Actions at 9:30 PM UTC (Mon–Fri).

Optional flags:
    --dry-run    Build and print the report but do not send the email.
    --save-html  Save the HTML report to ./report_output.html for inspection.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

# Load .env file (local dev only; GitHub Actions uses repository secrets)
load_dotenv()

from stock_analysis.config import BENCHMARK, LOOKBACK_DAYS, WATCHLIST
from stock_analysis.data_fetcher import fetch_current_quotes, fetch_news, fetch_price_history
from stock_analysis.email_sender import send_report
from stock_analysis.portfolio_risk import run_risk_analysis
from stock_analysis.report_builder import build_html_report
from stock_analysis.technicals import compute_technicals

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False, save_html: bool = False) -> None:
    now = datetime.now(timezone.utc)
    logger.info("=== Daily stock analysis starting — %s ===", now.strftime("%Y-%m-%d %H:%M UTC"))

    # 1. Fetch data -----------------------------------------------------------
    logger.info("Fetching price history (%d-day window) for %s + %s …", LOOKBACK_DAYS, WATCHLIST, BENCHMARK)
    stock_prices, market_prices = fetch_price_history(WATCHLIST, BENCHMARK, LOOKBACK_DAYS)
    logger.info("Price history fetched. Shape: stocks=%s, market=%s", stock_prices.shape, market_prices.shape)

    logger.info("Fetching current quotes …")
    quotes = fetch_current_quotes(WATCHLIST)
    for ticker, q in quotes.items():
        price = q.get("price")
        chg = q.get("change_pct")
        logger.info("  %-10s  $%s  (%+.2f%%)", ticker, f"{price or 0:,.2f}", chg or 0)

    # 2. Technical analysis ---------------------------------------------------
    logger.info("Computing technical indicators (MA5/MA10/MA20, RSI, bias) …")
    techs = compute_technicals(stock_prices, quotes)
    for ticker, t in sorted(techs.items(), key=lambda x: x[1].score, reverse=True):
        logger.info(
            "  %-10s  %-12s  score=%-3d  RSI=%-5s  bias=%s",
            ticker, t.signal, t.score,
            f"{t.rsi:.1f}" if t.rsi is not None else "N/A",
            f"{t.bias_pct:+.2f}%" if t.bias_pct is not None else "N/A",
        )

    # 3. News -----------------------------------------------------------------
    logger.info("Fetching latest news (48 h window) …")
    news_map = fetch_news(WATCHLIST)
    total_news = sum(len(v) for v in news_map.values())
    logger.info("  %d news items fetched across %d tickers.", total_news, len(news_map))

    # 4. Risk analysis --------------------------------------------------------
    logger.info("Running portfolio risk analysis …")
    risk_report = run_risk_analysis(stock_prices, market_prices)

    logger.info("  Portfolio volatility : %.1f%%", risk_report.portfolio.portfolio_volatility * 100)
    logger.info("  Portfolio beta       : %.2f",  risk_report.portfolio.portfolio_beta)
    logger.info("  Portfolio 90d return : %+.1f%%", risk_report.portfolio.portfolio_return_90d * 100)
    if risk_report.portfolio.flags:
        for flag in risk_report.portfolio.flags:
            logger.warning("  FLAG: %s", flag)

    # 5. Build HTML report ----------------------------------------------------
    logger.info("Building HTML report …")
    html = build_html_report(quotes, risk_report, techs=techs, news_map=news_map, report_date=now)

    if save_html:
        output_path = "report_output.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("HTML saved to %s", output_path)

    # 4. Send email -----------------------------------------------------------
    subject = f"Daily Market Report — {now.strftime('%A, %B %-d, %Y')}"

    if dry_run:
        logger.info("--dry-run: skipping email send. Report size: %d bytes.", len(html))
    else:
        logger.info("Sending email …")
        send_report(html, subject)

    logger.info("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily stock analysis report")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the report but do not send the email.",
    )
    parser.add_argument(
        "--save-html",
        action="store_true",
        help="Save the rendered HTML to report_output.html.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, save_html=args.save_html)
