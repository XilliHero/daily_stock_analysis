# Market Scanner — Quick Reference

## Run a Scan

Run from the project directory (`~/scanner-test`).

```bash
cd ~/scanner-test
```

### All 4 strategies (value, growth, dividend, recovery)

```bash
python3 main.py --market-scan
```

### Single strategy

```bash
python3 main.py --market-scan --scan-strategy value
python3 main.py --market-scan --scan-strategy growth
python3 main.py --market-scan --scan-strategy dividend
python3 main.py --market-scan --scan-strategy recovery
```

### Limit number of top picks

```bash
python3 main.py --market-scan --scan-top-n 20
```

### Scan only US or Canadian stocks

```bash
python3 main.py --market-scan --scan-region us
python3 main.py --market-scan --scan-region ca
python3 main.py --market-scan --scan-region us_ca   # both (default)
```

### Combine options

```bash
python3 main.py --market-scan --scan-strategy value --scan-region us --scan-top-n 30
```

---

## Reports

Reports are saved automatically to:

```
~/scanner-test/output/scans/
```

Each file is named like `scan_value_2026-05-08.md`.

Reports are also emailed to the address configured in your `.env` file.

---

## Scheduled Daily Scan (launchd)

The scanner is set to run every weekday at 5 PM automatically.

### Check if the schedule is active

```bash
launchctl list | grep marketscan
```

If you see a line with `com.dailystockanalysis.marketscan`, it's active.

### Stop the daily schedule

```bash
launchctl unload ~/Library/LaunchAgents/com.dailystockanalysis.marketscan.plist
```

### Restart the daily schedule

```bash
launchctl load ~/Library/LaunchAgents/com.dailystockanalysis.marketscan.plist
```

### Change the schedule time

1. Open the plist file:
   ```bash
   code ~/Library/LaunchAgents/com.dailystockanalysis.marketscan.plist
   ```
2. Change the `<integer>` value next to `Hour` (uses 24-hour format, e.g. 17 = 5 PM, 9 = 9 AM)
3. Save the file
4. Reload:
   ```bash
   launchctl unload ~/Library/LaunchAgents/com.dailystockanalysis.marketscan.plist
   launchctl load ~/Library/LaunchAgents/com.dailystockanalysis.marketscan.plist
   ```

---

## Update the Stock Universe

The scanner uses static CSV files with ~950 stocks. To refresh them from Wikipedia:

```bash
python3 main.py --refresh-universe
```

This updates the files in `data/scanner/` with the latest S&P 500, S&P 400, S&P 600, and TSX Composite constituents.

---

## Email Settings

Email configuration is in `~/scanner-test/.env`:

```
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
```

To change the recipient or credentials, edit this file:

```bash
code ~/scanner-test/.env
```

---

## Troubleshooting

### Scan runs but no email received

- Check your `.env` file has the correct `EMAIL_SENDER` and `EMAIL_PASSWORD`
- For Gmail, you need an App Password (not your regular password)
- Check your spam folder

### launchd not running at 5 PM

- Your Mac must be awake (not shut down). If asleep, it will run when it wakes up
- Verify the schedule is loaded: `launchctl list | grep marketscan`

### Check launchd logs

```bash
cat ~/scanner-test/output/scans/launchd.log
```

### "command not found: python3"

Use the full path:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 main.py --market-scan
```

---

## Strategy Cheat Sheet

| Strategy | What it looks for |
|----------|-------------------|
| **Value** | Low P/E, low P/B, strong balance sheet, undervalued stocks |
| **Growth** | Revenue growth, momentum, breakouts, earnings acceleration |
| **Dividend** | High yield, consistent payout history, financial stability |
| **Recovery** | Beaten-down stocks with drawdown from highs showing early reversal signals |
