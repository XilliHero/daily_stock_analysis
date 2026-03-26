#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Morning Pre-Market Briefing
───────────────────────────
Sends a concise, scannable email at 8 AM Eastern every weekday.

Sections
  1. Overnight Futures  — S&P, Nasdaq, Dow, Russell 2000, VIX, Gold, Oil
  2. Watchlist          — pre-market price vs. prior close for every ticker
  3. Key Headlines      — up to 2 recent stories per ticker (last 24 h)
  4. What to Watch      — auto-generated bullets derived from the data

No AI / LLM required.
Requires: EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVERS, STOCK_LIST
"""

import os
import sys
import smtplib
import logging
import time
from datetime import datetime, timedelta, timezone
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from typing import Optional

log = logging.getLogger("briefing")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ──────────────────────────────────────────────────
# STATIC CONFIG
# ──────────────────────────────────────────────────

# Futures / macro instruments always shown (symbol, friendly label)
FUTURES_WATCH = [
    ("ES=F",  "S&P 500 Futures"),
    ("NQ=F",  "Nasdaq 100 Futures"),
    ("YM=F",  "Dow Futures"),
    ("RTY=F", "Russell 2000 Futures"),
    ("^VIX",  "VIX (Fear Index)"),
    ("GC=F",  "Gold"),
    ("CL=F",  "Crude Oil (WTI)"),
]

SMTP_MAP: dict = {
    "gmail.com":   ("smtp.gmail.com",        587, False),
    "outlook.com": ("smtp-mail.outlook.com", 587, False),
    "hotmail.com": ("smtp-mail.outlook.com", 587, False),
    "live.com":    ("smtp-mail.outlook.com", 587, False),
    "qq.com":      ("smtp.qq.com",           465, True),
    "foxmail.com": ("smtp.qq.com",           465, True),
    "163.com":     ("smtp.163.com",          465, True),
    "126.com":     ("smtp.126.com",          465, True),
}

# ──────────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────────

def fetch_quote(sym: str, retries: int = 2) -> dict:
    """
    Return {sym, name, price, prev_close, pct_chg} for one ticker.
    Uses yfinance fast_info first; falls back to history() on failure.
    Returns {} on total failure (handled gracefully in rendering).
    """
    import yfinance as yf
    for attempt in range(retries + 1):
        try:
            t = yf.Ticker(sym)
            fi = t.fast_info

            price      = getattr(fi, "last_price",     None) or getattr(fi, "lastPrice",     None)
            prev_close = getattr(fi, "previous_close", None) or getattr(fi, "previousClose", None)

            # Fallback to history for futures / indices that may not populate fast_info
            if price is None or prev_close is None:
                hist = t.history(period="2d", prepost=True, auto_adjust=True)
                if not hist.empty:
                    price      = float(hist["Close"].iloc[-1])
                    prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
                else:
                    log.warning(f"  {sym}: no data returned")
                    return {}

            pct_chg = ((float(price) - float(prev_close)) / float(prev_close) * 100) if prev_close else 0.0

            # Best-effort display name (skip for speed if not needed)
            name = sym
            try:
                info_dict = t.info
                raw_name = info_dict.get("shortName") or info_dict.get("longName") or sym
                name = raw_name[:32] + "…" if len(raw_name) > 32 else raw_name
            except Exception:
                pass

            log.info(f"  {sym}: ${price:.4f}  ({pct_chg:+.2f}%)")
            return {
                "sym":        sym,
                "name":       name,
                "price":      float(price),
                "prev_close": float(prev_close),
                "pct_chg":    pct_chg,
            }

        except Exception as exc:
            if attempt < retries:
                log.warning(f"  {sym}: attempt {attempt+1} failed ({exc}), retrying...")
                time.sleep(2)
            else:
                log.warning(f"  {sym}: all attempts failed — {exc}")
                return {}
    return {}


def fetch_news(sym: str, hours: int = 24, max_items: int = 2) -> list:
    """
    Return up to `max_items` recent news dicts for sym.
    Each dict: {title, link, source, ts (datetime)}
    Filters to the last `hours` hours; silently returns [] on error.
    """
    import yfinance as yf
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        raw_news = yf.Ticker(sym).news or []
        results = []
        for item in raw_news[:15]:
            title  = (item.get("title") or "").strip()
            link   = item.get("link") or item.get("url") or "#"
            pub_ts = item.get("providerPublishTime") or 0
            source = item.get("publisher") or item.get("source") or ""
            pub_dt = datetime.fromtimestamp(pub_ts, tz=timezone.utc) if pub_ts else None
            if title and (pub_dt is None or pub_dt >= cutoff):
                results.append({"title": title, "link": link, "source": source, "ts": pub_dt})
            if len(results) >= max_items:
                break
        return results
    except Exception as exc:
        log.warning(f"  news({sym}): {exc}")
        return []


# ──────────────────────────────────────────────────
# FORMATTING HELPERS
# ──────────────────────────────────────────────────

def arrow(pct: float) -> str:
    return "▲" if pct >= 0 else "▼"

def clr(pct: float) -> str:
    """Return green or red hex for a % change."""
    return "#16a34a" if pct >= 0 else "#dc2626"

def fmt_pct(pct: float) -> str:
    return f"{arrow(pct)}&nbsp;{abs(pct):.2f}%"

def fmt_price(price: float, sym: str) -> str:
    """Format price with appropriate precision."""
    if sym in ("^VIX",):
        return f"{price:.2f}"
    if price >= 10_000:
        return f"${price:,.0f}"
    if price >= 100:
        return f"${price:,.2f}"
    if price >= 1:
        return f"${price:.4f}"
    return f"${price:.6f}"

def minutes_to_open(now_et: datetime) -> str:
    """Return human-friendly time until 9:30 AM ET on the same day."""
    open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    diff_secs = (open_et - now_et).total_seconds()
    if diff_secs <= 0:
        return "now open"
    total_mins = int(diff_secs / 60)
    h, m = divmod(total_mins, 60)
    if h == 0:
        return f"{m}m"
    return f"{h}h {m}m" if m else f"{h}h"


# ──────────────────────────────────────────────────
# WHAT TO WATCH GENERATOR
# ──────────────────────────────────────────────────

def build_watch_bullets(futures_data: list, watchlist_data: list) -> list:
    """
    Auto-generate actionable bullets from live data.
    Returns a list of HTML strings (may include <strong> tags).
    """
    bullets = []

    sp500  = next((r for r in futures_data if r.get("sym") == "ES=F"),  None)
    nasdaq = next((r for r in futures_data if r.get("sym") == "NQ=F"),  None)
    vix    = next((r for r in futures_data if r.get("sym") == "^VIX"),  None)
    gold   = next((r for r in futures_data if r.get("sym") == "GC=F"),  None)
    oil    = next((r for r in futures_data if r.get("sym") == "CL=F"),  None)

    # ── Market open sentiment ──────────────────────────────────
    if sp500 and nasdaq:
        sp_pct = sp500["pct_chg"]
        nq_pct = nasdaq["pct_chg"]
        avg    = (sp_pct + nq_pct) / 2
        if avg <= -1.0:
            bullets.append(
                f"<strong>Gap-down open expected</strong> — S&P futures {fmt_pct(sp_pct)}, "
                f"Nasdaq {fmt_pct(nq_pct)}. Watch key support levels; avoid chasing shorts at open."
            )
        elif avg >= 1.0:
            bullets.append(
                f"<strong>Gap-up open expected</strong> — S&P futures {fmt_pct(sp_pct)}, "
                f"Nasdaq {fmt_pct(nq_pct)}. Watch for gap-fill vs. continuation in the first 30 min."
            )
        else:
            bullets.append(
                f"Futures near flat (S&P {fmt_pct(sp_pct)}, Nasdaq {fmt_pct(nq_pct)}) — "
                f"expect a <strong>balanced open</strong>. Wait for price action to develop before committing."
            )
    elif sp500:
        pct = sp500["pct_chg"]
        tone = "bearish" if pct < -0.5 else ("bullish" if pct > 0.5 else "flat")
        bullets.append(f"S&P 500 futures {fmt_pct(pct)} — <strong>{tone} bias</strong> into the open.")

    # ── VIX / volatility ──────────────────────────────────────
    if vix:
        v = vix["price"]
        pct = vix["pct_chg"]
        if v >= 30:
            bullets.append(
                f"<strong>VIX at {v:.1f} — extreme fear.</strong> Widen stops, reduce size. "
                f"Expect intraday swings of 1–2%."
            )
        elif v >= 20:
            bullets.append(
                f"VIX elevated at {v:.1f} ({fmt_pct(pct)}) — <strong>volatility is high.</strong> "
                f"Use limit orders and avoid over-leveraging."
            )
        elif v <= 13:
            bullets.append(
                f"VIX at {v:.1f} — low fear, complacency risk. "
                f"Risk-on environment but watch for sudden reversals."
            )
        else:
            bullets.append(
                f"VIX at {v:.1f} ({fmt_pct(pct)}) — moderate uncertainty. Standard risk management applies."
            )

    # ── Watchlist big movers ───────────────────────────────────
    big_movers = sorted(
        [r for r in watchlist_data if r and abs(r["pct_chg"]) >= 2.0],
        key=lambda r: abs(r["pct_chg"]),
        reverse=True,
    )
    for row in big_movers[:3]:
        sym = row["sym"]
        pct = row["pct_chg"]
        direction = "higher" if pct >= 0 else "lower"
        bullets.append(
            f"<strong>{sym}</strong> is pre-market {direction} by "
            f"<strong>{abs(pct):.1f}%</strong> — "
            f"check the headlines section for the catalyst before trading."
        )

    # ── BTC as risk sentiment proxy ────────────────────────────
    btc = next((r for r in watchlist_data if r and "BTC" in r.get("sym", "").upper()), None)
    if btc and abs(btc["pct_chg"]) >= 3.0:
        pct = btc["pct_chg"]
        signal = "risk-on appetite" if pct > 0 else "risk-off pressure"
        bullets.append(
            f"BTC {fmt_pct(pct)} overnight — crypto is signaling "
            f"<strong>{signal}</strong> heading into the session."
        )

    # ── Gold / safe haven ─────────────────────────────────────
    if gold and abs(gold["pct_chg"]) >= 0.8:
        pct = gold["pct_chg"]
        if pct > 0:
            bullets.append(f"Gold {fmt_pct(pct)} — <strong>safe-haven demand rising.</strong> Watch defensive sectors.")
        else:
            bullets.append(f"Gold {fmt_pct(pct)} — safe-haven selling; risk appetite is improving.")

    # ── Oil / energy sector ───────────────────────────────────
    if oil and abs(oil["pct_chg"]) >= 1.5:
        pct = oil["pct_chg"]
        direction = "higher" if pct >= 0 else "lower"
        bullets.append(
            f"Crude oil {fmt_pct(pct)} — energy sector (XLE, MDA.TO) may "
            f"see outsized moves at open."
        )

    if not bullets:
        bullets.append(
            "No significant pre-market catalysts detected. "
            "Standard market-hours rules apply — wait for the first 15–30 min "
            "before entering new positions."
        )

    return bullets


# ──────────────────────────────────────────────────
# HTML EMAIL BUILDER
# ──────────────────────────────────────────────────

def build_html(
    now_et: datetime,
    futures_data: list,
    watchlist_data: list,
    news_data: dict,
    watch_bullets: list,
) -> str:
    date_str   = now_et.strftime("%A, %B %-d, %Y")
    time_str   = now_et.strftime("%-I:%M %p ET")
    opens_in   = minutes_to_open(now_et)

    # ── Futures table rows ────────────────────────
    futures_rows = ""
    for i, (sym, label) in enumerate(FUTURES_WATCH):
        row = futures_data[i] if i < len(futures_data) else {}
        if not row:
            futures_rows += f"""
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:7px 14px;color:#1e293b;font-weight:600;">{label}</td>
          <td style="padding:7px 14px;text-align:right;color:#94a3b8;">—</td>
          <td style="padding:7px 14px;text-align:right;color:#94a3b8;">—</td>
        </tr>"""
            continue
        pct = row["pct_chg"]
        futures_rows += f"""
        <tr style="border-bottom:1px solid #f1f5f9;">
          <td style="padding:7px 14px;color:#1e293b;font-weight:600;">{label}</td>
          <td style="padding:7px 14px;text-align:right;color:#334155;">{fmt_price(row['price'], sym)}</td>
          <td style="padding:7px 14px;text-align:right;font-weight:700;color:{clr(pct)};">{fmt_pct(pct)}</td>
        </tr>"""

    # ── Watchlist table rows ──────────────────────
    watch_rows = ""
    for row in watchlist_data:
        if not row:
            continue
        pct = row["pct_chg"]
        sym = row["sym"]
        bg  = "#fff7ed" if abs(pct) >= 2 else "transparent"
        watch_rows += f"""
        <tr style="border-bottom:1px solid #f1f5f9;background:{bg};">
          <td style="padding:7px 14px;font-weight:700;color:#0f172a;font-family:monospace;font-size:14px;">{sym}</td>
          <td style="padding:7px 14px;color:#64748b;font-size:13px;">{row['name']}</td>
          <td style="padding:7px 14px;text-align:right;color:#334155;">{fmt_price(row['price'], sym)}</td>
          <td style="padding:7px 14px;text-align:right;font-weight:700;color:{clr(pct)};">{fmt_pct(pct)}</td>
        </tr>"""

    if not watch_rows:
        watch_rows = '<tr><td colspan="4" style="padding:12px 14px;color:#94a3b8;text-align:center;">No data available</td></tr>'

    # ── News section ──────────────────────────────
    news_blocks = ""
    for sym, items in news_data.items():
        if not items:
            continue
        news_blocks += f"""
      <div style="margin-bottom:14px;">
        <span style="display:inline-block;background:#e2e8f0;color:#475569;font-size:11px;
                     font-weight:700;padding:2px 8px;border-radius:4px;font-family:monospace;
                     letter-spacing:.03em;">{sym}</span>"""
        for item in items:
            ts_str  = item["ts"].strftime("%-I:%M %p") if item.get("ts") else ""
            src_str = f' · {item["source"]}' if item.get("source") else ""
            meta    = f'<span style="color:#94a3b8;font-size:11px;">{ts_str}{src_str}</span>' if (ts_str or src_str) else ""
            link    = item.get("link", "#")
            title   = item["title"].replace("<", "&lt;").replace(">", "&gt;")
            news_blocks += f"""
        <div style="margin-top:6px;">
          <a href="{link}" style="color:#1d4ed8;text-decoration:none;font-size:14px;line-height:1.4;">{title}</a>
          {"<br>" + meta if meta else ""}
        </div>"""
        news_blocks += "\n      </div>"

    if not news_blocks:
        news_blocks = '<p style="color:#94a3b8;font-size:13px;margin:0;">No major headlines in the last 24 hours.</p>'

    # ── What to watch bullets ─────────────────────
    bullets_html = "\n".join(
        f'      <li style="margin:6px 0;color:#1c1917;line-height:1.6;font-size:14px;">{b}</li>'
        for b in watch_bullets
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Pre-Market Briefing — {date_str}</title>
</head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<div style="max-width:620px;margin:0 auto;padding:20px 12px 32px;">

  <!-- ── Header ── -->
  <div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);border-radius:12px 12px 0 0;padding:22px 28px;">
    <p style="margin:0;color:#94a3b8;font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;">
      📊 &nbsp;Pre-Market Briefing
    </p>
    <h1 style="margin:6px 0 4px;color:#f8fafc;font-size:20px;font-weight:700;">{date_str}</h1>
    <p style="margin:0;color:#64748b;font-size:13px;">
      Generated {time_str} &nbsp;·&nbsp; Market opens in <strong style="color:#93c5fd;">{opens_in}</strong>
    </p>
  </div>

  <!-- ── Overnight Futures ── -->
  <div style="background:#ffffff;border-left:1px solid #e2e8f0;border-right:1px solid #e2e8f0;padding:0;">
    <div style="padding:16px 28px 8px;">
      <h2 style="margin:0;font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.08em;">
        🔮 &nbsp;Overnight Futures &amp; Macro
      </h2>
    </div>
    <table style="width:100%;border-collapse:collapse;">
      <thead>
        <tr style="background:#f8fafc;">
          <th style="padding:6px 14px;text-align:left;font-size:11px;color:#94a3b8;font-weight:600;text-transform:uppercase;">Contract</th>
          <th style="padding:6px 14px;text-align:right;font-size:11px;color:#94a3b8;font-weight:600;text-transform:uppercase;">Last</th>
          <th style="padding:6px 14px;text-align:right;font-size:11px;color:#94a3b8;font-weight:600;text-transform:uppercase;">Change</th>
        </tr>
      </thead>
      <tbody>{futures_rows}
      </tbody>
    </table>
    <p style="margin:0;padding:8px 14px 14px;font-size:11px;color:#94a3b8;text-align:right;">
      vs. prior session close
    </p>
  </div>

  <!-- ── Watchlist Pre-Market ── -->
  <div style="background:#ffffff;border:1px solid #e2e8f0;border-top:none;padding:0;">
    <div style="padding:16px 28px 8px;">
      <h2 style="margin:0;font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.08em;">
        📋 &nbsp;Watchlist — Pre-Market
      </h2>
    </div>
    <table style="width:100%;border-collapse:collapse;">
      <thead>
        <tr style="background:#f8fafc;">
          <th style="padding:6px 14px;text-align:left;font-size:11px;color:#94a3b8;font-weight:600;text-transform:uppercase;">Ticker</th>
          <th style="padding:6px 14px;text-align:left;font-size:11px;color:#94a3b8;font-weight:600;text-transform:uppercase;">Name</th>
          <th style="padding:6px 14px;text-align:right;font-size:11px;color:#94a3b8;font-weight:600;text-transform:uppercase;">Price</th>
          <th style="padding:6px 14px;text-align:right;font-size:11px;color:#94a3b8;font-weight:600;text-transform:uppercase;">vs. Close</th>
        </tr>
      </thead>
      <tbody>{watch_rows}
      </tbody>
    </table>
    <p style="margin:0;padding:8px 14px 14px;font-size:11px;color:#94a3b8;text-align:right;">
      ⚡ Highlighted rows = pre-market move ≥ 2%
    </p>
  </div>

  <!-- ── Key Headlines ── -->
  <div style="background:#ffffff;border:1px solid #e2e8f0;border-top:none;padding:16px 28px 18px;">
    <h2 style="margin:0 0 14px;font-size:12px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.08em;">
      📰 &nbsp;Key Headlines (last 24 h)
    </h2>
    {news_blocks}
  </div>

  <!-- ── What to Watch ── -->
  <div style="background:#fffbeb;border:1px solid #fde68a;border-top:none;border-radius:0 0 12px 12px;padding:16px 28px 20px;">
    <h2 style="margin:0 0 10px;font-size:12px;font-weight:700;color:#92400e;text-transform:uppercase;letter-spacing:.08em;">
      🎯 &nbsp;What to Watch Today
    </h2>
    <ul style="margin:0;padding-left:18px;line-height:1.5;">
{bullets_html}
    </ul>
  </div>

  <!-- ── Footer ── -->
  <p style="text-align:center;color:#94a3b8;font-size:11px;margin-top:18px;line-height:1.6;">
    Daily Stock Analysis &nbsp;·&nbsp; Pre-Market Briefing<br>
    Data via Yahoo Finance &nbsp;·&nbsp; Not financial advice
  </p>

</div>
</body>
</html>"""


# ──────────────────────────────────────────────────
# EMAIL SEND
# ──────────────────────────────────────────────────

def send_email(subject: str, html_body: str, sender: str, password: str, receivers: list) -> bool:
    domain = sender.split("@")[-1].lower()
    smtp_server, smtp_port, use_ssl = SMTP_MAP.get(domain, (f"smtp.{domain}", 465, True))

    msg = MIMEMultipart("alternative")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"]    = formataddr(("Daily Stock Analysis", sender))
    msg["To"]      = ", ".join(receivers)
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    server = None
    try:
        log.info(f"Connecting to {smtp_server}:{smtp_port} (ssl={use_ssl})...")
        if use_ssl:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
            server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        log.info(f"✅ Briefing sent to: {receivers}")
        return True
    except smtplib.SMTPAuthenticationError:
        log.error("Email auth failed — check EMAIL_SENDER and EMAIL_PASSWORD.")
        return False
    except Exception as exc:
        log.error(f"Email send failed: {exc}")
        return False
    finally:
        if server:
            try:
                server.quit()
            except Exception:
                pass


# ──────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────

def main() -> None:
    # ── Read config from environment ──────────────
    sender   = os.getenv("EMAIL_SENDER", "").strip()
    password = os.getenv("EMAIL_PASSWORD", "").strip()
    rcv_raw  = os.getenv("EMAIL_RECEIVERS", "").strip()
    receivers = [r.strip() for r in rcv_raw.split(",") if r.strip()] or (
        [sender] if sender else []
    )
    stock_raw = os.getenv(
        "STOCK_LIST", "TSLA,OKLO,XLE,CORN,MDA.TO,BTC-USD,SPOT"
    ).strip()
    watchlist = [s.strip().upper() for s in stock_raw.split(",") if s.strip()]
    force_run = os.getenv("FORCE_RUN", "false").lower() in ("true", "1", "yes")

    if not sender or not password:
        log.error("EMAIL_SENDER and EMAIL_PASSWORD must be set.")
        sys.exit(1)

    # ── Current time in Eastern ───────────────────
    # Cron runs at 13:00 UTC = 8:00 AM EST (UTC-5) / 9:00 AM EDT (UTC-4).
    # We detect DST roughly by month: EDT (UTC-4) from March–November.
    now_utc   = datetime.now(timezone.utc)
    month     = now_utc.month
    et_offset = timedelta(hours=-4) if 3 <= month <= 11 else timedelta(hours=-5)
    now_et    = now_utc.astimezone(timezone(et_offset))

    log.info("=" * 52)
    log.info(f"📊 Morning Pre-Market Briefing")
    log.info(f"   Time    : {now_et.strftime('%Y-%m-%d %H:%M ET')}")
    log.info(f"   Watchlist: {watchlist}")
    log.info(f"   Receivers: {receivers}")
    log.info("=" * 52)

    # ── Fetch futures ─────────────────────────────
    log.info("Fetching overnight futures...")
    futures_data = []
    for sym, label in FUTURES_WATCH:
        row = fetch_quote(sym)
        if row:
            row["name"] = label          # override with our friendly label
        futures_data.append(row)

    # ── Fetch watchlist quotes ────────────────────
    log.info("Fetching watchlist pre-market quotes...")
    watchlist_data = [fetch_quote(sym) for sym in watchlist]

    # ── Fetch news ────────────────────────────────
    log.info("Fetching news headlines...")
    news_data: dict = {}
    for sym in watchlist:
        items = fetch_news(sym)
        if items:
            news_data[sym] = items

    # ── Build "What to Watch" ─────────────────────
    watch_bullets = build_watch_bullets(futures_data, watchlist_data)

    # ── Build HTML ────────────────────────────────
    html_body = build_html(now_et, futures_data, watchlist_data, news_data, watch_bullets)

    # ── Send email ────────────────────────────────
    date_label = now_et.strftime("%b %-d")
    day_label  = now_et.strftime("%a")
    subject    = f"📊 Pre-Market Briefing — {day_label} {date_label}"

    ok = send_email(subject, html_body, sender, password, receivers)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
