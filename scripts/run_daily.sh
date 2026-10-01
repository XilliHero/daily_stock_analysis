#!/bin/bash
# Daily stock analysis runner — called by cron Mon-Fri at 4:30 PM ET

REPO="/Users/josevargasvega/AI/daily_stock_analysis"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
LOG="$REPO/logs/cron_$(date +%Y%m%d).log"

cd "$REPO" || { echo "$(date): ERROR: cannot cd to $REPO" >> "$LOG"; exit 1; }

echo "=== $(date): Starting daily analysis ===" >> "$LOG"
"$PYTHON" main.py >> "$LOG" 2>&1
echo "=== $(date): Finished (exit $?) ===" >> "$LOG"
