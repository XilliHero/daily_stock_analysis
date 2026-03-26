"""
HTML Email Report Builder
=========================
Assembles the full daily HTML email from:
  - Current quotes (price snapshot)
  - Portfolio risk metrics (beta, alpha, correlation, summary)

All styles are inline for maximum email-client compatibility.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from stock_analysis.config import (
    HIGH_BETA_THRESHOLD,
    HIGH_CORRELATION_THRESHOLD,
    HIGH_VOLATILITY_THRESHOLD,
    SECTOR_MAP,
    WATCHLIST,
)
from stock_analysis.portfolio_risk import RiskReport


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

COLORS = {
    "bg":           "#0f1117",
    "card":         "#1a1d27",
    "card_alt":     "#1e2130",
    "border":       "#2a2d3e",
    "text":         "#e2e8f0",
    "text_muted":   "#94a3b8",
    "accent":       "#6366f1",   # indigo
    "green":        "#22c55e",
    "red":          "#ef4444",
    "yellow":       "#f59e0b",
    "blue":         "#3b82f6",
    "header_grad1": "#1e1b4b",
    "header_grad2": "#0f172a",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_price(v: float | None, currency: str = "") -> str:
    if v is None:
        return "N/A"
    prefix = currency or ""
    return f"{prefix}{v:,.2f}"


def _fmt_pct(v: float | None, decimals: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{v:+.{decimals}f}%"


def _fmt_float(v: float | None, decimals: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def _change_color(v: float | None) -> str:
    if v is None:
        return COLORS["text_muted"]
    return COLORS["green"] if v >= 0 else COLORS["red"]


def _pct_color(v: float | None, good_positive: bool = True) -> str:
    if v is None:
        return COLORS["text_muted"]
    positive_is_good = good_positive
    if positive_is_good:
        return COLORS["green"] if v >= 0 else COLORS["red"]
    return COLORS["red"] if v >= 0 else COLORS["green"]


def _beta_color(beta: float) -> str:
    if beta < 0:
        return COLORS["blue"]
    if beta > HIGH_BETA_THRESHOLD:
        return COLORS["red"]
    if beta > 1.0:
        return COLORS["yellow"]
    return COLORS["green"]


def _corr_cell_style(corr: float) -> str:
    """Return background and text color based on correlation strength."""
    abs_c = abs(corr)
    if abs_c >= 0.9:
        bg = "#7f1d1d" if corr > 0 else "#1e3a5f"
    elif abs_c >= HIGH_CORRELATION_THRESHOLD:
        bg = "#991b1b" if corr > 0 else "#1e40af"
    elif abs_c >= 0.5:
        bg = "#78350f" if corr > 0 else "#1e3a5f"
    elif abs_c >= 0.25:
        bg = "#292524"
    else:
        bg = COLORS["card"]

    return (
        f"background:{bg}; color:{COLORS['text']}; "
        f"padding:8px 6px; text-align:center; font-size:13px; "
        f"border:1px solid {COLORS['border']}; font-weight:{'600' if abs_c >= HIGH_CORRELATION_THRESHOLD else '400'};"
    )


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

BASE_CELL = (
    f"padding:10px 14px; border:1px solid {COLORS['border']}; "
    f"font-size:13px; color:{COLORS['text']}; vertical-align:middle;"
)
HEADER_CELL = (
    f"padding:10px 14px; border:1px solid {COLORS['border']}; "
    f"font-size:11px; text-transform:uppercase; letter-spacing:0.06em; "
    f"color:{COLORS['text_muted']}; font-weight:600; background:{COLORS['card_alt']};"
)


def _section_header(title: str, subtitle: str = "") -> str:
    sub_html = (
        f'<p style="margin:4px 0 0; font-size:12px; color:{COLORS["text_muted"]};">'
        f"{subtitle}</p>"
        if subtitle
        else ""
    )
    return (
        f'<div style="margin:0 0 12px;">'
        f'<h2 style="margin:0; font-size:16px; font-weight:700; color:{COLORS["text"]}; '
        f'letter-spacing:0.02em;">'
        f'<span style="color:{COLORS["accent"]};">▌</span> {title}</h2>'
        f"{sub_html}"
        f"</div>"
    )


def _build_price_snapshot(quotes: dict[str, dict]) -> str:
    rows_html = ""
    for ticker in WATCHLIST:
        q = quotes.get(ticker, {})
        price = q.get("price")
        chg = q.get("change_pct")
        high = q.get("day_high")
        low = q.get("day_low")
        vol = q.get("volume")
        wk_high = q.get("fifty_two_week_high")
        wk_low = q.get("fifty_two_week_low")

        chg_color = _change_color(chg)
        arrow = "▲" if (chg or 0) >= 0 else "▼"

        vol_str = f"{int(vol):,}" if vol else "N/A"

        rows_html += f"""
        <tr>
          <td style="{BASE_CELL} font-weight:600; color:{COLORS['accent']};">{ticker}</td>
          <td style="{BASE_CELL} font-weight:700; font-size:14px;">{_fmt_price(price)}</td>
          <td style="{BASE_CELL} color:{chg_color}; font-weight:600;">{arrow} {_fmt_pct(chg)}</td>
          <td style="{BASE_CELL}">{_fmt_price(high)} / {_fmt_price(low)}</td>
          <td style="{BASE_CELL}">{_fmt_price(wk_high)} / {_fmt_price(wk_low)}</td>
          <td style="{BASE_CELL} color:{COLORS['text_muted']};">{vol_str}</td>
        </tr>"""

    return f"""
    <div style="margin:0 0 32px;">
      {_section_header("Market Snapshot", "Closing prices as of today's session")}
      <table style="width:100%; border-collapse:collapse;">
        <thead>
          <tr>
            <th style="{HEADER_CELL}">Ticker</th>
            <th style="{HEADER_CELL}">Last Price</th>
            <th style="{HEADER_CELL}">Day Change</th>
            <th style="{HEADER_CELL}">Day Hi / Lo</th>
            <th style="{HEADER_CELL}">52-Wk Hi / Lo</th>
            <th style="{HEADER_CELL}">Volume</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""


def _build_beta_alpha_table(risk: RiskReport) -> str:
    rows_html = ""
    for ticker in WATCHLIST:
        tr = risk.ticker_risks.get(ticker)
        if tr is None:
            rows_html += f"""
            <tr>
              <td style="{BASE_CELL} font-weight:600; color:{COLORS['accent']};">{ticker}</td>
              <td style="{BASE_CELL} color:{COLORS['text_muted']};" colspan="5">Data unavailable</td>
            </tr>"""
            continue

        beta_col = _beta_color(tr.beta)
        alpha_col = _pct_color(tr.alpha_annualised)
        ret_col = _pct_color(tr.return_90d)
        vol_col = COLORS["red"] if tr.volatility >= HIGH_VOLATILITY_THRESHOLD else COLORS["text"]

        # Beta interpretation label
        if tr.beta < 0:
            beta_label = "inverse"
        elif tr.beta < 0.5:
            beta_label = "low"
        elif tr.beta < 1.0:
            beta_label = "moderate"
        elif tr.beta < HIGH_BETA_THRESHOLD:
            beta_label = "high"
        else:
            beta_label = "very high"

        rows_html += f"""
        <tr>
          <td style="{BASE_CELL} font-weight:600; color:{COLORS['accent']};">{ticker}</td>
          <td style="{BASE_CELL}">
            <span style="color:{beta_col}; font-weight:700; font-size:14px;">{_fmt_float(tr.beta)}</span>
            <span style="font-size:11px; color:{COLORS['text_muted']}; margin-left:5px;">({beta_label})</span>
          </td>
          <td style="{BASE_CELL} color:{alpha_col}; font-weight:600;">{_fmt_pct(tr.alpha_annualised * 100, 2)}</td>
          <td style="{BASE_CELL} color:{ret_col}; font-weight:600;">{_fmt_pct(tr.return_90d * 100, 2)}</td>
          <td style="{BASE_CELL} color:{vol_col};">{_fmt_pct(tr.volatility * 100, 1)}</td>
          <td style="{BASE_CELL} color:{COLORS['text_muted']}; font-size:12px;">{SECTOR_MAP.get(ticker, '—')}</td>
        </tr>"""

    return f"""
    <div style="margin:0 0 28px;">
      {_section_header(
          "Beta &amp; Alpha (Jensen's)",
          f"90-day rolling window vs. S&amp;P 500 (^GSPC) · Equal weighting · Rf = {5:.0f}%"
      )}
      <table style="width:100%; border-collapse:collapse;">
        <thead>
          <tr>
            <th style="{HEADER_CELL}">Ticker</th>
            <th style="{HEADER_CELL}">Beta</th>
            <th style="{HEADER_CELL}">Alpha (Annualised)</th>
            <th style="{HEADER_CELL}">90-Day Return</th>
            <th style="{HEADER_CELL}">Ann. Volatility</th>
            <th style="{HEADER_CELL}">Sector</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
      <p style="margin:8px 0 0; font-size:11px; color:{COLORS['text_muted']};">
        Beta &lt; 1: less volatile than S&amp;P 500 · Beta &gt; 1: more volatile ·
        Negative beta: moves opposite to market ·
        Alpha: return above what beta alone predicts (annualised)
      </p>
    </div>"""


def _build_correlation_matrix(risk: RiskReport) -> str:
    corr = risk.correlation_matrix
    tickers = list(corr.columns)

    # Header row
    header_cells = f'<th style="{HEADER_CELL}"></th>'
    for t in tickers:
        header_cells += f'<th style="{HEADER_CELL} text-align:center;">{t}</th>'

    # Data rows
    rows_html = ""
    for row_t in tickers:
        cells = f'<td style="{BASE_CELL} font-weight:600; color:{COLORS["accent"]};">{row_t}</td>'
        for col_t in tickers:
            c = corr.loc[row_t, col_t]
            if row_t == col_t:
                cells += (
                    f'<td style="background:{COLORS["card_alt"]}; color:{COLORS["text_muted"]}; '
                    f'padding:8px 6px; text-align:center; font-size:13px; '
                    f'border:1px solid {COLORS["border"]};">—</td>'
                )
            else:
                flag = " ⚠" if abs(c) >= HIGH_CORRELATION_THRESHOLD else ""
                cells += f'<td style="{_corr_cell_style(c)}">{c:+.2f}{flag}</td>'
        rows_html += f"<tr>{cells}</tr>"

    legend = (
        f'<p style="margin:10px 0 0; font-size:11px; color:{COLORS["text_muted"]};">'
        f"Color scale: "
        f'<span style="background:#991b1b; color:white; padding:1px 6px; border-radius:3px;">strong +corr</span> &nbsp;'
        f'<span style="background:#1e40af; color:white; padding:1px 6px; border-radius:3px;">strong −corr</span> &nbsp;'
        f'<span style="background:#78350f; color:white; padding:1px 6px; border-radius:3px;">moderate +corr</span> &nbsp;'
        f'<span style="background:{COLORS["card"]}; color:{COLORS["text"]}; padding:1px 6px; border-radius:3px; '
        f'border:1px solid {COLORS["border"]};">low corr</span> &nbsp;'
        f'⚠ = above {HIGH_CORRELATION_THRESHOLD:.0%} threshold'
        f"</p>"
    )

    return f"""
    <div style="margin:0 0 28px;">
      {_section_header(
          "Correlation Matrix",
          "Pairwise Pearson correlations of daily returns over the 90-day window"
      )}
      <div style="overflow-x:auto;">
        <table style="border-collapse:collapse; min-width:100%;">
          <thead><tr>{header_cells}</tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
      {legend}
    </div>"""


def _build_portfolio_summary(risk: RiskReport) -> str:
    p = risk.portfolio

    vol_color = COLORS["red"] if p.high_volatility_flag else COLORS["green"]
    beta_color = COLORS["red"] if p.high_beta_flag else (
        COLORS["yellow"] if p.portfolio_beta > 1.0 else COLORS["green"]
    )
    ret_color = _pct_color(p.portfolio_return_90d)

    # KPI cards
    kpi_style = (
        f"display:inline-block; background:{COLORS['card_alt']}; "
        f"border:1px solid {COLORS['border']}; border-radius:8px; "
        f"padding:14px 20px; margin:6px; min-width:160px; text-align:center;"
    )

    kpis = f"""
    <div style="margin:0 0 16px; text-align:center;">
      <div style="{kpi_style}">
        <div style="font-size:11px; color:{COLORS['text_muted']}; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">Portfolio Volatility</div>
        <div style="font-size:24px; font-weight:700; color:{vol_color};">{p.portfolio_volatility:.1%}</div>
        <div style="font-size:11px; color:{COLORS['text_muted']}; margin-top:3px;">annualised</div>
      </div>
      <div style="{kpi_style}">
        <div style="font-size:11px; color:{COLORS['text_muted']}; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">Portfolio Beta</div>
        <div style="font-size:24px; font-weight:700; color:{beta_color};">{p.portfolio_beta:.2f}</div>
        <div style="font-size:11px; color:{COLORS['text_muted']}; margin-top:3px;">vs. S&amp;P 500</div>
      </div>
      <div style="{kpi_style}">
        <div style="font-size:11px; color:{COLORS['text_muted']}; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">90-Day Return</div>
        <div style="font-size:24px; font-weight:700; color:{ret_color};">{p.portfolio_return_90d:+.1%}</div>
        <div style="font-size:11px; color:{COLORS['text_muted']}; margin-top:3px;">equal-weighted</div>
      </div>
      <div style="{kpi_style}">
        <div style="font-size:11px; color:{COLORS['text_muted']}; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">Holdings</div>
        <div style="font-size:24px; font-weight:700; color:{COLORS['text']};">{len(risk.ticker_risks)}</div>
        <div style="font-size:11px; color:{COLORS['text_muted']}; margin-top:3px;">equal-weighted</div>
      </div>
    </div>"""

    # Flags section
    if p.flags:
        flag_items = "".join(
            f'<li style="margin:4px 0; font-size:13px;">{flag}</li>'
            for flag in p.flags
        )
        flags_html = f"""
        <div style="background:#451a03; border:1px solid #92400e; border-radius:8px; padding:14px 18px; margin-top:12px;">
          <div style="font-size:12px; font-weight:700; color:{COLORS['yellow']}; margin-bottom:8px;">
            ⚠ Risk Flags ({len(p.flags)})
          </div>
          <ul style="margin:0; padding-left:18px; color:{COLORS['text']};">
            {flag_items}
          </ul>
        </div>"""
    else:
        flags_html = f"""
        <div style="background:#052e16; border:1px solid #16a34a; border-radius:8px; padding:14px 18px; margin-top:12px;">
          <div style="font-size:13px; color:{COLORS['green']}; font-weight:600;">
            ✓ No risk flags — portfolio appears well-diversified within thresholds.
          </div>
        </div>"""

    # High-correlation pair detail
    if p.high_corr_pairs:
        pair_rows = ""
        for cp in p.high_corr_pairs:
            c_col = COLORS["red"] if cp.correlation > 0 else COLORS["blue"]
            pair_rows += f"""
            <tr>
              <td style="{BASE_CELL} font-weight:600;">{cp.ticker_a}</td>
              <td style="{BASE_CELL} font-weight:600;">{cp.ticker_b}</td>
              <td style="{BASE_CELL} color:{c_col}; font-weight:700;">{cp.correlation:+.3f}</td>
            </tr>"""

        pairs_section = f"""
        <div style="margin-top:16px;">
          <div style="font-size:13px; font-weight:600; color:{COLORS['text']}; margin-bottom:8px;">
            High-Correlation Pairs (|r| ≥ {HIGH_CORRELATION_THRESHOLD:.0%})
          </div>
          <table style="border-collapse:collapse; width:auto;">
            <thead>
              <tr>
                <th style="{HEADER_CELL}">Ticker A</th>
                <th style="{HEADER_CELL}">Ticker B</th>
                <th style="{HEADER_CELL}">Correlation</th>
              </tr>
            </thead>
            <tbody>{pair_rows}</tbody>
          </table>
          <p style="margin:6px 0 0; font-size:11px; color:{COLORS['text_muted']};">
            Highly correlated holdings provide limited diversification benefit.
          </p>
        </div>"""
    else:
        pairs_section = ""

    return f"""
    <div style="margin:0 0 28px;">
      {_section_header(
          "Portfolio Risk Summary",
          "Equal-weighted across all holdings · 90-day window"
      )}
      {kpis}
      {flags_html}
      {pairs_section}
    </div>"""


def _build_portfolio_risk_section(risk: RiskReport) -> str:
    """Full portfolio risk section, including divider and all sub-sections."""
    divider = (
        f'<div style="border-top:2px solid {COLORS["accent"]}; margin:32px 0 28px; '
        f'opacity:0.4;"></div>'
    )

    section_title = (
        f'<div style="margin:0 0 24px;">'
        f'<h1 style="margin:0 0 4px; font-size:20px; font-weight:800; color:{COLORS["text"]};">'
        f'<span style="color:{COLORS["accent"]};">⬡</span> Portfolio Risk Analysis</h1>'
        f'<p style="margin:0; font-size:12px; color:{COLORS["text_muted"]};">'
        f'Beta · Jensen\'s Alpha · Correlation Matrix · Risk Flags'
        f'</p></div>'
    )

    return (
        divider
        + section_title
        + _build_beta_alpha_table(risk)
        + _build_correlation_matrix(risk)
        + _build_portfolio_summary(risk)
    )


# ---------------------------------------------------------------------------
# Full report assembly
# ---------------------------------------------------------------------------

def build_html_report(
    quotes: dict[str, dict],
    risk: RiskReport,
    report_date: datetime | None = None,
) -> str:
    """Return the complete HTML email string."""
    if report_date is None:
        report_date = datetime.now(timezone.utc)

    date_str = report_date.strftime("%A, %B %-d, %Y")
    time_str = report_date.strftime("%H:%M UTC")

    price_section = _build_price_snapshot(quotes)
    risk_section = _build_portfolio_risk_section(risk)

    footer_note = (
        f'<p style="margin:0; font-size:11px; color:{COLORS["text_muted"]}; text-align:center;">'
        f'Data sourced from Yahoo Finance via yfinance · Generated {time_str} · '
        f'Watchlist: {", ".join(WATCHLIST)} · Benchmark: ^GSPC'
        f'</p>'
        f'<p style="margin:8px 0 0; font-size:10px; color:{COLORS["text_muted"]}; text-align:center;">'
        f'This report is for informational purposes only and does not constitute financial advice.'
        f'</p>'
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Daily Stock Report — {date_str}</title>
</head>
<body style="margin:0; padding:0; background:{COLORS['bg']}; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div style="max-width:860px; margin:0 auto; padding:24px 16px;">

    <!-- ─── Header ─────────────────────────────────────────────── -->
    <div style="background:linear-gradient(135deg,{COLORS['header_grad1']},{COLORS['header_grad2']});
                border:1px solid {COLORS['border']}; border-radius:12px;
                padding:28px 32px; margin-bottom:28px; text-align:center;">
      <div style="font-size:11px; color:{COLORS['accent']}; text-transform:uppercase;
                  letter-spacing:0.12em; font-weight:700; margin-bottom:8px;">
        Daily Market Report
      </div>
      <h1 style="margin:0; font-size:26px; font-weight:800; color:{COLORS['text']};">
        {date_str}
      </h1>
      <p style="margin:8px 0 0; font-size:13px; color:{COLORS['text_muted']};">
        After-market analysis · {time_str}
      </p>
    </div>

    <!-- ─── Main content card ───────────────────────────────────── -->
    <div style="background:{COLORS['card']}; border:1px solid {COLORS['border']};
                border-radius:12px; padding:28px 32px; margin-bottom:20px;">

      {price_section}
      {risk_section}

    </div>

    <!-- ─── Footer ──────────────────────────────────────────────── -->
    <div style="padding:0 8px;">
      {footer_note}
    </div>

  </div>
</body>
</html>"""

    return html
