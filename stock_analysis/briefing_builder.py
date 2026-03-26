"""
Morning Briefing HTML Builder
==============================
Builds a compact, scannable pre-market HTML email.
Shares the same dark colour palette and inline-CSS approach as report_builder.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

from stock_analysis.briefing_data import BriefingData, FutureQuote
from stock_analysis.report_builder import COLORS, BASE_CELL, HEADER_CELL, _section_header, _fmt_pct


# ---------------------------------------------------------------------------
# Section: Overnight Futures
# ---------------------------------------------------------------------------

def _futures_row(fq: FutureQuote) -> str:
    if fq.price is None:
        price_str = "N/A"
        chg_str   = "N/A"
        chg_color = COLORS["text_muted"]
        arrow     = ""
    else:
        price_str = f"{fq.price:,.2f}"
        chg_color = COLORS["green"] if (fq.change_pct or 0) >= 0 else COLORS["red"]
        arrow     = "▲" if (fq.change_pct or 0) >= 0 else "▼"
        chg_str   = (
            f"{arrow} {_fmt_pct(fq.change_pct)}"
            if fq.change_pct is not None
            else "N/A"
        )

    return f"""
        <tr>
          <td style="{BASE_CELL} font-weight:600; color:{COLORS['accent']};">{fq.name}</td>
          <td style="{BASE_CELL} font-size:11px; color:{COLORS['text_muted']};">{fq.ticker}</td>
          <td style="{BASE_CELL} font-weight:700; font-size:14px;">{price_str}</td>
          <td style="{BASE_CELL} color:{chg_color}; font-weight:600;">{chg_str}</td>
        </tr>"""


def _build_futures_section(futures: list[FutureQuote]) -> str:
    rows = "".join(_futures_row(fq) for fq in futures)
    return f"""
    <div style="margin:0 0 28px;">
      {_section_header("Overnight Futures", "Latest quotes vs. previous close")}
      <table style="width:100%; border-collapse:collapse;">
        <thead>
          <tr>
            <th style="{HEADER_CELL}">Index</th>
            <th style="{HEADER_CELL}">Ticker</th>
            <th style="{HEADER_CELL}">Last Price</th>
            <th style="{HEADER_CELL}">Overnight Change</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


# ---------------------------------------------------------------------------
# Section: News Headlines
# ---------------------------------------------------------------------------

def _build_news_section(news: dict[str, list]) -> str:
    ticker_blocks = ""
    for ticker, items in news.items():
        if not items:
            headlines_html = (
                f'<p style="margin:0; font-size:12px; color:{COLORS["text_muted"]}; '
                f'font-style:italic;">No recent headlines found.</p>'
            )
        else:
            headline_items = ""
            for item in items:
                link_open  = f'<a href="{item.url}" style="color:{COLORS["accent"]}; text-decoration:none;">' if item.url else ""
                link_close = "</a>" if item.url else ""
                pub = (
                    f'<span style="font-size:11px; color:{COLORS["text_muted"]}; margin-left:6px;">— {item.publisher}</span>'
                    if item.publisher
                    else ""
                )
                headline_items += (
                    f'<li style="margin:0 0 6px; font-size:13px; color:{COLORS["text"]}; line-height:1.45;">'
                    f"{link_open}{item.title}{link_close}{pub}</li>"
                )
            headlines_html = f'<ul style="margin:0; padding-left:18px;">{headline_items}</ul>'

        ticker_blocks += f"""
        <div style="margin:0 0 14px; padding:14px 16px;
                    background:{COLORS['card_alt']}; border:1px solid {COLORS['border']};
                    border-radius:8px;">
          <div style="font-size:12px; font-weight:700; color:{COLORS['accent']};
                      text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;">
            {ticker}
          </div>
          {headlines_html}
        </div>"""

    return f"""
    <div style="margin:0 0 28px;">
      {_section_header("Pre-Market Headlines", "Top recent news for each watchlist ticker")}
      {ticker_blocks}
    </div>"""


# ---------------------------------------------------------------------------
# Section: Key Watchpoints
# ---------------------------------------------------------------------------

def _build_watchpoints_section(watchpoints: list[str]) -> str:
    if not watchpoints:
        return ""

    items_html = "".join(
        f'<li style="margin:0 0 8px; font-size:13px; color:{COLORS["text"]}; line-height:1.5;">'
        f'<span style="color:{COLORS["yellow"]}; margin-right:6px;">›</span>{point}</li>'
        for point in watchpoints
    )

    return f"""
    <div style="margin:0 0 8px;">
      {_section_header("Key Things to Watch Today", "Data-driven focus points for the session")}
      <div style="background:#1c1a08; border:1px solid #78350f; border-radius:8px; padding:16px 18px;">
        <ul style="margin:0; padding-left:8px; list-style:none;">
          {items_html}
        </ul>
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# Full briefing assembly
# ---------------------------------------------------------------------------

def build_briefing_html(
    data: BriefingData,
    report_date: datetime | None = None,
) -> str:
    """Return the complete pre-market briefing HTML email string."""
    if report_date is None:
        report_date = datetime.now(timezone.utc)

    date_str = report_date.strftime("%A, %B %-d, %Y")
    time_str = report_date.strftime("%H:%M UTC")

    futures_section     = _build_futures_section(data.futures)
    news_section        = _build_news_section(data.news)
    watchpoints_section = _build_watchpoints_section(data.watchpoints)

    footer_note = (
        f'<p style="margin:0; font-size:11px; color:{COLORS["text_muted"]}; text-align:center;">'
        f'Data sourced from Yahoo Finance via yfinance · Generated {time_str} · '
        f'Futures: ES=F, NQ=F, ^GSPTSX'
        f'</p>'
        f'<p style="margin:8px 0 0; font-size:10px; color:{COLORS["text_muted"]}; text-align:center;">'
        f'This briefing is for informational purposes only and does not constitute financial advice.'
        f'</p>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pre-Market Briefing — {date_str}</title>
</head>
<body style="margin:0; padding:0; background:{COLORS['bg']}; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div style="max-width:860px; margin:0 auto; padding:24px 16px;">

    <!-- ─── Header ────────────────────────────────────────────────── -->
    <div style="background:linear-gradient(135deg,{COLORS['header_grad1']},{COLORS['header_grad2']});
                border:1px solid {COLORS['border']}; border-radius:12px;
                padding:28px 32px; margin-bottom:28px; text-align:center;">
      <div style="font-size:11px; color:{COLORS['accent']}; text-transform:uppercase;
                  letter-spacing:0.12em; font-weight:700; margin-bottom:8px;">
        Morning Pre-Market Briefing
      </div>
      <h1 style="margin:0; font-size:26px; font-weight:800; color:{COLORS['text']};">
        {date_str}
      </h1>
      <p style="margin:8px 0 0; font-size:13px; color:{COLORS['text_muted']};">
        Before the open · {time_str}
      </p>
    </div>

    <!-- ─── Main content card ──────────────────────────────────────── -->
    <div style="background:{COLORS['card']}; border:1px solid {COLORS['border']};
                border-radius:12px; padding:28px 32px; margin-bottom:20px;">

      {futures_section}

      <div style="border-top:1px solid {COLORS['border']}; margin:4px 0 28px; opacity:0.5;"></div>

      {news_section}

      <div style="border-top:1px solid {COLORS['border']}; margin:4px 0 28px; opacity:0.5;"></div>

      {watchpoints_section}

    </div>

    <!-- ─── Footer ────────────────────────────────────────────────── -->
    <div style="padding:0 8px;">
      {footer_note}
    </div>

  </div>
</body>
</html>"""
