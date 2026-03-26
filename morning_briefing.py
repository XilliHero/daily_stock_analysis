"""
Morning Pre-Market Briefing — Main Orchestrator
================================================
Sends a short, scannable pre-market email before markets open.

Run manually:
    python morning_briefing.py

Or triggered by GitHub Actions at 13:00 UTC (8:00 AM Eastern, Mon–Fri).

Optional flags:
    --dry-run    Build and print the report but do not send the email.
    --save-html  Save the HTML briefing to ./briefing_output.html for inspection.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

# Load .env file (local dev only; GitHub Actions uses repository secrets)
load_dotenv()

from stock_analysis.briefing_builder import build_briefing_html
from stock_analysis.briefing_data import fetch_briefing_data
from stock_analysis.config import WATCHLIST
from stock_analysis.email_sender import send_report

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
    logger.info("=== Morning briefing starting — %s ===", now.strftime("%Y-%m-%d %H:%M UTC"))

    # 1. Fetch data -----------------------------------------------------------
    logger.info("Fetching briefing data for watchlist: %s …", WATCHLIST)
    data = fetch_briefing_data(WATCHLIST)

    for fq in data.futures:
        logger.info(
            "  %-10s  %s  %s",
            fq.name,
            f"{fq.price:,.2f}" if fq.price else "N/A",
            f"({fq.change_pct:+.2f}%)" if fq.change_pct is not None else "",
        )
    for ticker, headlines in data.news.items():
        logger.info("  %-10s  %d headline(s)", ticker, len(headlines))

    # 2. Build HTML -----------------------------------------------------------
    logger.info("Building HTML briefing …")
    html = build_briefing_html(data, report_date=now)

    if save_html:
        output_path = "briefing_output.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("HTML saved to %s", output_path)

    # 3. Send email -----------------------------------------------------------
    subject = f"Pre-Market Briefing — {now.strftime('%A, %B %-d, %Y')}"

    if dry_run:
        logger.info("--dry-run: skipping email send. Briefing size: %d bytes.", len(html))
    else:
        logger.info("Sending briefing email …")
        send_report(html, subject)

    logger.info("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Morning pre-market briefing")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the briefing but do not send the email.",
    )
    parser.add_argument(
        "--save-html",
        action="store_true",
        help="Save the rendered HTML to briefing_output.html.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, save_html=args.save_html)
