"""
HTML Email Report Builder
=========================
Assembles the full daily HTML email from:
  - Decision dashboard   (Buy/Watch/Sell summary, sorted by score)
  - Market snapshot      (live prices, day change, hi/lo, volume)
  - Per-stock analysis   (MA5/10/20, RSI, bias, battle plan, news)
  - Portfolio risk       (beta, alpha, correlation matrix, risk flags)

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
from stock_analysis.technicals import TechnicalSignal


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

C = {
    "bg":           "#0f1117",
    "card":         "#1a1d27",
    "card_alt":     "#1e2130",
    "border":       "#2a2d3e",
    "text":         "#e2e8f0",
    "muted":        "#94a3b8",
    "accent":       "#6366f1",
    "green":        "#22c55e",
    "green_bg":     "#052e16",
    "green_border": "#16a34a",
    "red":          "#ef4444",
    "red_bg":       "#450a0a",
    "red_border":   "#b91c1c",
    "yellow":       "#f59e0b",
    "yellow_bg":    "#451a03",
    "yellow_border":"#92400e",
    "blue":         "#3b82f6",
    "purple":       "#a78bfa",
    "header1":      "#1e1b4b",
    "header2":      "#0f172a",
}

BASE_CELL = (
    f"padding:10px 14px; border:1px solid {C['border']}; "
    f"font-size:13px; color:{C['text']}; vertical-align:middle;"
)
HDR_CELL = (
    f"padding:10px 14px; border:1px solid {C['border']}; "
    f"font-size:11px; text-transform:uppercase; letter-spacing:0.06em; "
    f"color:{C['muted']}; font-weight:600; background:{C['card_alt']};"
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _p(v: float | None, prefix: str = "") -> str:
    return "N/A" if v is None else f"{prefix}{v:,.2f}"


def _pct(v: float | None, dec: int = 2) -> str:
    return "N/A" if v is None else f"{v:+.{dec}f}%"


def _chg_color(v: float | None) -> str:
    if v is None:
        return C["muted"]
    return C["green"] if v >= 0 else C["red"]


def _pct_color(v: float | None, good_pos: bool = True) -> str:
    if v is None:
        return C["muted"]
    return (C["green"] if v >= 0 else C["red"]) if good_pos else (C["red"] if v >= 0 else C["green"])


def _section_header(title: str, subtitle: str = "") -> str:
    sub = (
        f'<p style="margin:4px 0 0; font-size:12px; color:{C["muted"]};">{subtitle}</p>'
        if subtitle else ""
    )
    return (
        f'<div style="margin:0 0 14px;">'
        f'<h2 style="margin:0; font-size:16px; font-weight:700; color:{C["text"]}; '
        f'letter-spacing:0.02em;">'
        f'<span style="color:{C["accent"]};">&#9646;</span> {title}</h2>'
        f"{sub}</div>"
    )


def _signal_badge(signal: str) -> str:
    cfg = {
        "Strong Buy": (C["green_bg"],  C["green"],  C["green_border"],  "&#128154; Strong Buy"),
        "Buy":        (C["green_bg"],  C["green"],  C["green_border"],  "&#129001; Buy"),
        "Watch":      (C["card_alt"], C["yellow"], C["yellow_border"], "&#9898; Watch"),
        "Sell":       (C["red_bg"],    C["red"],    C["red_border"],    "&#128308; Sell"),
    }
    bg, fg, border, label = cfg.get(
        signal,
        (C["card_alt"], C["muted"], C["border"], signal)
    )
    return (
        f'<span style="background:{bg}; color:{fg}; border:1px solid {border}; '
        f'border-radius:999px; padding:3px 10px; font-size:12px; font-weight:700; '
        f'white-space:nowrap;">{label}</span>'
    )


def _checklist_icon(passed: bool | None) -> str:
    if passed is True:
        return f'<span style="color:{C["green"]};">&#10003;</span>'
    if passed is False:
        return f'<span style="color:{C["red"]};">&#10007;</span>'
    return f'<span style="color:{C["yellow"]};">&#9888;</span>'


# ---------------------------------------------------------------------------
# 1 – Decision Dashboard
# ---------------------------------------------------------------------------

def _build_decision_dashboard(
    techs: dict[str, TechnicalSignal],
    quotes: dict[str, dict],
) -> str:
    buys    = [t for t in techs.values() if t.signal in ("Strong Buy", "Buy")]
    watches = [t for t in techs.values() if t.signal == "Watch"]
    sells   = [t for t in techs.values() if t.signal == "Sell"]

    n = len(techs)
    summary_bar = (
        f'<div style="margin:0 0 16px; font-size:13px; color:{C["muted"]};">'
        f'Analyzed <strong style="color:{C["text"]};">{n}</strong> stocks &nbsp;|&nbsp; '
        f'<span style="color:{C["green"]};">Buy: {len(buys)}</span> &nbsp;'
        f'<span style="color:{C["yellow"]};">Watch: {len(watches)}</span> &nbsp;'
        f'<span style="color:{C["red"]};">Sell: {len(sells)}</span>'
        f'</div>'
    )

    sorted_techs = sorted(techs.values(), key=lambda t: t.score, reverse=True)
    rows = ""
    for t in sorted_techs:
        q     = quotes.get(t.ticker, {})
        chg   = q.get("change_pct")
        chg_c = _chg_color(chg)
        arrow = "&#9650;" if (chg or 0) >= 0 else "&#9660;"
        score_color = (
            C["green"] if t.score >= 70 else C["yellow"] if t.score >= 45 else C["red"]
        )
        rows += f"""
        <tr>
          <td style="{BASE_CELL} font-weight:700; color:{C['accent']}; font-size:14px;">{t.ticker}</td>
          <td style="{BASE_CELL}">{_signal_badge(t.signal)}</td>
          <td style="{BASE_CELL}">
            <span style="color:{score_color}; font-weight:700; font-size:15px;">{t.score}</span>
            <span style="color:{C['muted']}; font-size:11px;">/100</span>
          </td>
          <td style="{BASE_CELL} color:{C['muted']}; font-size:12px;">{t.trend}</td>
          <td style="{BASE_CELL} font-weight:600;">{_p(t.price)}</td>
          <td style="{BASE_CELL} color:{chg_c}; font-weight:600;">{arrow} {_pct(chg)}</td>
          <td style="{BASE_CELL} color:{C['muted']}; font-size:11px;">{t.action_urgency}</td>
        </tr>"""

    return f"""
    <div style="margin:0 0 32px;">
      {_section_header("Decision Dashboard", "Rule-based signals from MA alignment, RSI, and bias ratio")}
      {summary_bar}
      <table style="width:100%; border-collapse:collapse;">
        <thead><tr>
          <th style="{HDR_CELL}">Ticker</th>
          <th style="{HDR_CELL}">Signal</th>
          <th style="{HDR_CELL}">Score</th>
          <th style="{HDR_CELL}">Trend</th>
          <th style="{HDR_CELL}">Last Price</th>
          <th style="{HDR_CELL}">Day Chg</th>
          <th style="{HDR_CELL}">Urgency</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


# ---------------------------------------------------------------------------
# 2 – Market Snapshot
# ---------------------------------------------------------------------------

def _build_price_snapshot(quotes: dict[str, dict]) -> str:
    rows = ""
    for ticker in WATCHLIST:
        q   = quotes.get(ticker, {})
        p   = q.get("price")
        chg = q.get("change_pct")
        hi  = q.get("day_high")
        lo  = q.get("day_low")
        vol = q.get("volume")
        wh  = q.get("fifty_two_week_high")
        wl  = q.get("fifty_two_week_low")

        chg_c = _chg_color(chg)
        arrow = "&#9650;" if (chg or 0) >= 0 else "&#9660;"
        vol_s = f"{int(vol):,}" if vol else "N/A"

        rows += f"""
        <tr>
          <td style="{BASE_CELL} font-weight:600; color:{C['accent']};">{ticker}</td>
          <td style="{BASE_CELL} font-weight:700; font-size:14px;">{_p(p)}</td>
          <td style="{BASE_CELL} color:{chg_c}; font-weight:600;">{arrow} {_pct(chg)}</td>
          <td style="{BASE_CELL}">{_p(hi)} / {_p(lo)}</td>
          <td style="{BASE_CELL}">{_p(wh)} / {_p(wl)}</td>
          <td style="{BASE_CELL} color:{C['muted']};">{vol_s}</td>
        </tr>"""

    return f"""
    <div style="margin:0 0 32px;">
      {_section_header("Market Snapshot", "Closing prices as of today's session")}
      <table style="width:100%; border-collapse:collapse;">
        <thead><tr>
          <th style="{HDR_CELL}">Ticker</th>
          <th style="{HDR_CELL}">Last Price</th>
          <th style="{HDR_CELL}">Day Change</th>
          <th style="{HDR_CELL}">Day Hi / Lo</th>
          <th style="{HDR_CELL}">52-Wk Hi / Lo</th>
          <th style="{HDR_CELL}">Volume</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


# ---------------------------------------------------------------------------
# 3 – Per-stock analysis cards
# ---------------------------------------------------------------------------

def _build_stock_card(t: TechnicalSignal, news: list[dict]) -> str:
    sig_color = {
        "Strong Buy": C["green"],
        "Buy":        C["green"],
        "Watch":      C["yellow"],
        "Sell":       C["red"],
    }.get(t.signal, C["muted"])

    trend_color = (
        C["green"]  if "Bullish" in t.trend
        else C["red"]   if "Bearish" in t.trend
        else C["yellow"]
    )

    # ── Signal header ──────────────────────────────────────────────────────
    header_html = f"""
    <div style="margin-bottom:14px;">
      <div style="display:flex; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:10px;">
        {_signal_badge(t.signal)}
        <span style="font-size:12px; color:{trend_color}; font-weight:600;">{t.trend}</span>
        <span style="font-size:12px; color:{C['muted']};">Score:
          <strong style="color:{sig_color};">{t.score}/100</strong>
        </span>
        <span style="font-size:12px; color:{C['muted']};">&#9201; {t.action_urgency}</span>
      </div>
      <p style="margin:0; font-size:13px; color:{C['text']}; font-style:italic;">{t.summary}</p>
    </div>"""

    # ── Position guidance table ────────────────────────────────────────────
    if t.signal in ("Strong Buy", "Buy"):
        g_no  = (
            f"Consider initiating a {t.position_size_pct}% position near "
            f"${_p(t.ideal_entry)} (MA5) or on a dip to ${_p(t.secondary_entry)} (MA10)."
        )
        g_yes = (
            f"Maintain position. Add on confirmed pullback to MA5. "
            f"Protect with stop at ${_p(t.stop_loss)}."
        )
    else:
        g_no  = "Do not initiate new long positions. Wait for MA alignment reversal."
        g_yes = (
            f"Consider reducing exposure. Set stop below MA20. "
            f"Watch for reversal above ${_p(t.ma20)}."
        )

    guidance_html = f"""
    <table style="width:100%; border-collapse:collapse; margin-bottom:16px;">
      <thead><tr>
        <th style="{HDR_CELL}">Position</th>
        <th style="{HDR_CELL}">Recommended Action</th>
      </tr></thead>
      <tbody>
        <tr>
          <td style="{BASE_CELL} font-weight:600; color:{C['accent']}; white-space:nowrap;">No Position</td>
          <td style="{BASE_CELL}">{g_no}</td>
        </tr>
        <tr>
          <td style="{BASE_CELL} font-weight:600; color:{C['purple']}; white-space:nowrap;">Holding</td>
          <td style="{BASE_CELL}">{g_yes}</td>
        </tr>
      </tbody>
    </table>"""

    # ── Risk flags ─────────────────────────────────────────────────────────
    flags_html = ""
    if t.flags:
        flag_items = "".join(
            f'<li style="margin:3px 0; font-size:12px;">{f}</li>' for f in t.flags
        )
        flags_html = f"""
        <div style="background:{C['yellow_bg']}; border:1px solid {C['yellow_border']};
                    border-radius:6px; padding:10px 14px; margin-bottom:16px;">
          <div style="font-size:11px; font-weight:700; color:{C['yellow']}; margin-bottom:6px;">
            &#9888; Risk Flags
          </div>
          <ul style="margin:0; padding-left:16px; color:{C['text']};">{flag_items}</ul>
        </div>"""

    # ── Technical data ─────────────────────────────────────────────────────
    bias_color = (
        C["green"] if t.bias_pct is not None and -5 < t.bias_pct < 5
        else C["red"]
    )
    rsi_color = (
        C["red"]  if t.rsi is not None and t.rsi >= 70
        else C["blue"] if t.rsi is not None and t.rsi <= 30
        else C["green"]
    )
    ma_str = (
        "Bullish — MA5 &gt; MA10 &gt; MA20"
        if t.ma5 and t.ma10 and t.ma20 and t.ma5 > t.ma10 > t.ma20
        else "Bearish — MA5 &lt; MA10 &lt; MA20"
        if t.ma5 and t.ma10 and t.ma20 and t.ma5 < t.ma10 < t.ma20
        else "Mixed" if t.ma5 and t.ma10 and t.ma20
        else "N/A"
    )
    ma_color = (
        C["green"] if "Bullish" in ma_str
        else C["red"] if "Bearish" in ma_str
        else C["yellow"]
    )

    rsi_val = f"{t.rsi:.1f}" if t.rsi is not None else "N/A"

    tech_html = f"""
    <div style="margin-bottom:16px;">
      <div style="font-size:12px; font-weight:700; color:{C['text']}; margin-bottom:8px;">
        &#128202; Technical Data
      </div>
      <div style="font-size:12px; margin-bottom:8px;">
        MA Alignment: <strong style="color:{ma_color};">{ma_str}</strong>
      </div>
      <table style="width:100%; border-collapse:collapse; margin-bottom:10px;">
        <thead><tr>
          <th style="{HDR_CELL}">Metric</th><th style="{HDR_CELL}">Value</th>
          <th style="{HDR_CELL}">Metric</th><th style="{HDR_CELL}">Value</th>
        </tr></thead>
        <tbody>
          <tr>
            <td style="{BASE_CELL} color:{C['muted']};">Price</td>
            <td style="{BASE_CELL} font-weight:700;">${_p(t.price)}</td>
            <td style="{BASE_CELL} color:{C['muted']};">RSI (14)</td>
            <td style="{BASE_CELL} color:{rsi_color}; font-weight:700;">{rsi_val}</td>
          </tr>
          <tr>
            <td style="{BASE_CELL} color:{C['muted']};">MA5</td>
            <td style="{BASE_CELL}">${_p(t.ma5)}</td>
            <td style="{BASE_CELL} color:{C['muted']};">Bias vs MA5</td>
            <td style="{BASE_CELL} color:{bias_color}; font-weight:600;">{_pct(t.bias_pct)}</td>
          </tr>
          <tr>
            <td style="{BASE_CELL} color:{C['muted']};">MA10</td>
            <td style="{BASE_CELL}">${_p(t.ma10)}</td>
            <td style="{BASE_CELL} color:{C['muted']};">Support</td>
            <td style="{BASE_CELL} color:{C['blue']};">${_p(t.support)}</td>
          </tr>
          <tr>
            <td style="{BASE_CELL} color:{C['muted']};">MA20</td>
            <td style="{BASE_CELL}">${_p(t.ma20)}</td>
            <td style="{BASE_CELL} color:{C['muted']};">Resistance</td>
            <td style="{BASE_CELL} color:{C['purple']};">${_p(t.resistance)}</td>
          </tr>
        </tbody>
      </table>
    </div>"""

    # ── Battle plan ────────────────────────────────────────────────────────
    if t.signal in ("Strong Buy", "Buy"):
        plan_rows = f"""
        <tr>
          <td style="{BASE_CELL} color:{C['muted']};">&#127919; Ideal Entry</td>
          <td style="{BASE_CELL} color:{C['green']}; font-weight:700;">${_p(t.ideal_entry)} (near MA5)</td>
        </tr>
        <tr>
          <td style="{BASE_CELL} color:{C['muted']};">&#128309; Secondary Entry</td>
          <td style="{BASE_CELL} color:{C['blue']};">${_p(t.secondary_entry)} (near MA10)</td>
        </tr>
        <tr>
          <td style="{BASE_CELL} color:{C['muted']};">&#128683; Stop Loss</td>
          <td style="{BASE_CELL} color:{C['red']};">${_p(t.stop_loss)} (2% below MA20)</td>
        </tr>
        <tr>
          <td style="{BASE_CELL} color:{C['muted']};">&#127881; Target</td>
          <td style="{BASE_CELL} color:{C['purple']}; font-weight:700;">${_p(t.target)}</td>
        </tr>
        <tr>
          <td style="{BASE_CELL} color:{C['muted']};">&#128176; Position Size</td>
          <td style="{BASE_CELL} font-weight:600;">{t.position_size_pct}% of portfolio</td>
        </tr>"""
    else:
        plan_rows = f"""
        <tr>
          <td style="{BASE_CELL} color:{C['muted']};">&#127919; Ideal Entry</td>
          <td style="{BASE_CELL} color:{C['muted']};">Not recommended — bearish trend</td>
        </tr>
        <tr>
          <td style="{BASE_CELL} color:{C['muted']};">&#128683; Protective Stop</td>
          <td style="{BASE_CELL} color:{C['red']};">${_p(t.stop_loss)} (for existing positions)</td>
        </tr>
        <tr>
          <td style="{BASE_CELL} color:{C['muted']};">&#128260; Reversal Signal</td>
          <td style="{BASE_CELL} color:{C['yellow']};">Sustained close above MA20 (${_p(t.ma20)}) with volume</td>
        </tr>
        <tr>
          <td style="{BASE_CELL} color:{C['muted']};">&#128176; Position Size</td>
          <td style="{BASE_CELL} color:{C['muted']};">0% — no new long positions</td>
        </tr>"""

    plan_html = f"""
    <div style="margin-bottom:16px;">
      <div style="font-size:12px; font-weight:700; color:{C['text']}; margin-bottom:8px;">
        &#127919; Battle Plan
      </div>
      <table style="width:100%; border-collapse:collapse;">
        <tbody>{plan_rows}</tbody>
      </table>
    </div>"""

    # ── Checklist ──────────────────────────────────────────────────────────
    cl_items = "".join(
        f'<li style="margin:4px 0; font-size:12px; list-style:none;">'
        f'{_checklist_icon(passed)} {label}</li>'
        for label, passed in t.checklist
    )
    checklist_html = f"""
    <div style="margin-bottom:16px;">
      <div style="font-size:12px; font-weight:700; color:{C['text']}; margin-bottom:6px;">
        Entry Checklist
      </div>
      <ul style="margin:0; padding:0;">{cl_items}</ul>
    </div>"""

    # ── News ───────────────────────────────────────────────────────────────
    if news:
        news_items = ""
        for n in news:
            pub  = n.get("published_at")
            pub_s = pub.strftime("%b %-d, %H:%M UTC") if pub else ""
            pub_s += f' · {n["publisher"]}' if n.get("publisher") else ""
            url  = n.get("url", "#")
            ttl  = n.get("title", "")
            news_items += f"""
            <div style="padding:8px 0; border-bottom:1px solid {C['border']};">
              <a href="{url}" style="color:{C['accent']}; text-decoration:none;
                 font-size:13px; font-weight:500;" target="_blank">{ttl}</a>
              <div style="font-size:11px; color:{C['muted']}; margin-top:2px;">{pub_s}</div>
            </div>"""
        news_html = f"""
        <div style="margin-bottom:8px;">
          <div style="font-size:12px; font-weight:700; color:{C['text']}; margin-bottom:6px;">
            &#128240; Latest News (last 48 h)
          </div>{news_items}
        </div>"""
    else:
        news_html = f"""
        <div style="margin-bottom:8px;">
          <div style="font-size:12px; font-weight:700; color:{C['text']}; margin-bottom:6px;">
            &#128240; Latest News
          </div>
          <p style="margin:0; font-size:12px; color:{C['muted']};">
            No relevant news in the last 48 hours.
          </p>
        </div>"""

    sector = SECTOR_MAP.get(t.ticker, "")
    sector_tag = (
        f' <span style="font-size:11px; color:{C["muted"]}; '
        f'border:1px solid {C["border"]}; border-radius:4px; '
        f'padding:1px 6px;">{sector}</span>'
        if sector else ""
    )

    return f"""
    <div style="background:{C['card_alt']}; border:1px solid {C['border']};
                border-radius:10px; padding:22px 24px; margin-bottom:20px;">
      <div style="border-bottom:1px solid {C['border']}; padding-bottom:12px; margin-bottom:16px;">
        <span style="font-size:18px; font-weight:800; color:{C['text']};">{t.ticker}</span>
        {sector_tag}
      </div>
      {header_html}
      {flags_html}
      {guidance_html}
      {tech_html}
      {plan_html}
      {checklist_html}
      {news_html}
    </div>"""


def _build_all_stock_cards(
    techs: dict[str, TechnicalSignal],
    news_map: dict[str, list[dict]],
) -> str:
    sorted_techs = sorted(techs.values(), key=lambda t: t.score, reverse=True)
    cards = "".join(
        _build_stock_card(t, news_map.get(t.ticker, []))
        for t in sorted_techs
    )
    divider = (
        f'<div style="border-top:2px solid {C["accent"]}; margin:32px 0 28px; opacity:0.35;"></div>'
    )
    title = (
        f'<div style="margin:0 0 20px;">'
        f'<h1 style="margin:0 0 4px; font-size:20px; font-weight:800; color:{C["text"]};">'
        f'<span style="color:{C["accent"]};">&#9672;</span> Individual Stock Analysis</h1>'
        f'<p style="margin:0; font-size:12px; color:{C["muted"]};">'
        f'MA5 · MA10 · MA20 · RSI-14 · Bias · Battle Plan · News'
        f'</p></div>'
    )
    return divider + title + cards


# ---------------------------------------------------------------------------
# 4 – Portfolio risk
# ---------------------------------------------------------------------------

def _beta_color(beta: float) -> str:
    if beta < 0:        return C["blue"]
    if beta > HIGH_BETA_THRESHOLD: return C["red"]
    if beta > 1.0:      return C["yellow"]
    return C["green"]


def _corr_cell_style(corr: float) -> str:
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
        bg = C["card"]
    return (
        f"background:{bg}; color:{C['text']}; padding:8px 6px; text-align:center; "
        f"font-size:13px; border:1px solid {C['border']}; "
        f"font-weight:{'600' if abs_c >= HIGH_CORRELATION_THRESHOLD else '400'};"
    )


def _build_beta_alpha_table(risk: RiskReport) -> str:
    rows = ""
    for ticker in WATCHLIST:
        tr = risk.ticker_risks.get(ticker)
        if tr is None:
            rows += (
                f'<tr><td style="{BASE_CELL} font-weight:600; color:{C["accent"]};">{ticker}</td>'
                f'<td style="{BASE_CELL} color:{C["muted"]};" colspan="5">Data unavailable</td></tr>'
            )
            continue
        bc   = _beta_color(tr.beta)
        ac   = _pct_color(tr.alpha_annualised)
        rc   = _pct_color(tr.return_90d)
        vc   = C["red"] if tr.volatility >= HIGH_VOLATILITY_THRESHOLD else C["text"]
        blbl = (
            "inverse"  if tr.beta < 0    else "low"      if tr.beta < 0.5
            else "moderate" if tr.beta < 1.0 else "high"     if tr.beta < HIGH_BETA_THRESHOLD
            else "very high"
        )
        rows += f"""
        <tr>
          <td style="{BASE_CELL} font-weight:600; color:{C['accent']};">{ticker}</td>
          <td style="{BASE_CELL}">
            <span style="color:{bc}; font-weight:700; font-size:14px;">{tr.beta:.2f}</span>
            <span style="font-size:11px; color:{C['muted']}; margin-left:5px;">({blbl})</span>
          </td>
          <td style="{BASE_CELL} color:{ac}; font-weight:600;">{_pct(tr.alpha_annualised * 100)}</td>
          <td style="{BASE_CELL} color:{rc}; font-weight:600;">{_pct(tr.return_90d * 100)}</td>
          <td style="{BASE_CELL} color:{vc};">{_pct(tr.volatility * 100, 1)}</td>
          <td style="{BASE_CELL} color:{C['muted']}; font-size:12px;">{SECTOR_MAP.get(ticker, '&#8212;')}</td>
        </tr>"""

    return f"""
    <div style="margin:0 0 28px;">
      {_section_header("Beta &amp; Alpha (Jensen's)",
          "90-day rolling window vs. S&amp;P 500 · Equal weighting · Rf = 5%")}
      <table style="width:100%; border-collapse:collapse;">
        <thead><tr>
          <th style="{HDR_CELL}">Ticker</th>
          <th style="{HDR_CELL}">Beta</th>
          <th style="{HDR_CELL}">Alpha (Ann.)</th>
          <th style="{HDR_CELL}">90-Day Return</th>
          <th style="{HDR_CELL}">Ann. Volatility</th>
          <th style="{HDR_CELL}">Sector</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <p style="margin:8px 0 0; font-size:11px; color:{C['muted']};">
        Beta &lt;1: less volatile · Beta &gt;1: more volatile ·
        Negative beta: inverse · Alpha: excess return above beta prediction
      </p>
    </div>"""


def _build_correlation_matrix(risk: RiskReport) -> str:
    corr    = risk.correlation_matrix
    tickers = list(corr.columns)
    hdrs    = f'<th style="{HDR_CELL}"></th>'
    for t in tickers:
        hdrs += f'<th style="{HDR_CELL} text-align:center;">{t}</th>'
    rows = ""
    for rt in tickers:
        cells = f'<td style="{BASE_CELL} font-weight:600; color:{C["accent"]};">{rt}</td>'
        for ct in tickers:
            c = corr.loc[rt, ct]
            if rt == ct:
                cells += (
                    f'<td style="background:{C["card_alt"]}; color:{C["muted"]}; '
                    f'padding:8px 6px; text-align:center; font-size:13px; '
                    f'border:1px solid {C["border"]};">&#8212;</td>'
                )
            else:
                flag = " &#9888;" if abs(c) >= HIGH_CORRELATION_THRESHOLD else ""
                cells += f'<td style="{_corr_cell_style(c)}">{c:+.2f}{flag}</td>'
        rows += f"<tr>{cells}</tr>"
    legend = (
        f'<p style="margin:10px 0 0; font-size:11px; color:{C["muted"]};">'
        f'&#9888; = above {HIGH_CORRELATION_THRESHOLD:.0%} correlation threshold'
        f'</p>'
    )
    return f"""
    <div style="margin:0 0 28px;">
      {_section_header("Correlation Matrix",
          "Pairwise Pearson correlations of daily returns · 90-day window")}
      <div style="overflow-x:auto;">
        <table style="border-collapse:collapse; min-width:100%;">
          <thead><tr>{hdrs}</tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
      {legend}
    </div>"""


def _build_portfolio_summary(risk: RiskReport) -> str:
    p     = risk.portfolio
    vol_c = C["red"] if p.high_volatility_flag else C["green"]
    bc    = C["red"] if p.high_beta_flag else (C["yellow"] if p.portfolio_beta > 1.0 else C["green"])
    rc    = _pct_color(p.portfolio_return_90d)
    ks    = (
        f"display:inline-block; background:{C['card_alt']}; border:1px solid {C['border']}; "
        f"border-radius:8px; padding:14px 20px; margin:6px; min-width:150px; text-align:center;"
    )
    kpis = f"""
    <div style="margin:0 0 16px; text-align:center;">
      <div style="{ks}">
        <div style="font-size:11px;color:{C['muted']};text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Volatility</div>
        <div style="font-size:24px;font-weight:700;color:{vol_c};">{p.portfolio_volatility:.1%}</div>
        <div style="font-size:11px;color:{C['muted']};margin-top:3px;">annualised</div>
      </div>
      <div style="{ks}">
        <div style="font-size:11px;color:{C['muted']};text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Beta</div>
        <div style="font-size:24px;font-weight:700;color:{bc};">{p.portfolio_beta:.2f}</div>
        <div style="font-size:11px;color:{C['muted']};margin-top:3px;">vs. S&amp;P 500</div>
      </div>
      <div style="{ks}">
        <div style="font-size:11px;color:{C['muted']};text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">90-Day Return</div>
        <div style="font-size:24px;font-weight:700;color:{rc};">{p.portfolio_return_90d:+.1%}</div>
        <div style="font-size:11px;color:{C['muted']};margin-top:3px;">equal-weighted</div>
      </div>
      <div style="{ks}">
        <div style="font-size:11px;color:{C['muted']};text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Holdings</div>
        <div style="font-size:24px;font-weight:700;color:{C['text']};">{len(risk.ticker_risks)}</div>
        <div style="font-size:11px;color:{C['muted']};margin-top:3px;">equal-weighted</div>
      </div>
    </div>"""

    if p.flags:
        fi = "".join(f'<li style="margin:4px 0;font-size:13px;">{f}</li>' for f in p.flags)
        fh = f"""
        <div style="background:{C['yellow_bg']};border:1px solid {C['yellow_border']};
                    border-radius:8px;padding:14px 18px;margin-top:12px;">
          <div style="font-size:12px;font-weight:700;color:{C['yellow']};margin-bottom:8px;">
            &#9888; Risk Flags ({len(p.flags)})
          </div>
          <ul style="margin:0;padding-left:18px;color:{C['text']};">{fi}</ul>
        </div>"""
    else:
        fh = f"""
        <div style="background:{C['green_bg']};border:1px solid {C['green_border']};
                    border-radius:8px;padding:14px 18px;margin-top:12px;">
          <div style="font-size:13px;color:{C['green']};font-weight:600;">
            &#10003; No risk flags — portfolio appears well-diversified within thresholds.
          </div>
        </div>"""

    pairs_section = ""
    if p.high_corr_pairs:
        pr = ""
        for cp in p.high_corr_pairs:
            cc = C["red"] if cp.correlation > 0 else C["blue"]
            pr += f"""
            <tr>
              <td style="{BASE_CELL} font-weight:600;">{cp.ticker_a}</td>
              <td style="{BASE_CELL} font-weight:600;">{cp.ticker_b}</td>
              <td style="{BASE_CELL} color:{cc}; font-weight:700;">{cp.correlation:+.3f}</td>
            </tr>"""
        pairs_section = f"""
        <div style="margin-top:16px;">
          <div style="font-size:13px;font-weight:600;color:{C['text']};margin-bottom:8px;">
            High-Correlation Pairs (|r| &#8805; {HIGH_CORRELATION_THRESHOLD:.0%})
          </div>
          <table style="border-collapse:collapse;width:auto;">
            <thead><tr>
              <th style="{HDR_CELL}">Ticker A</th>
              <th style="{HDR_CELL}">Ticker B</th>
              <th style="{HDR_CELL}">Correlation</th>
            </tr></thead>
            <tbody>{pr}</tbody>
          </table>
        </div>"""

    return f"""
    <div style="margin:0 0 28px;">
      {_section_header("Portfolio Risk Summary", "Equal-weighted · 90-day window")}
      {kpis}{fh}{pairs_section}
    </div>"""


def _build_portfolio_risk_section(risk: RiskReport) -> str:
    divider = (
        f'<div style="border-top:2px solid {C["accent"]}; margin:32px 0 28px; opacity:0.4;"></div>'
    )
    title = (
        f'<div style="margin:0 0 24px;">'
        f'<h1 style="margin:0 0 4px;font-size:20px;font-weight:800;color:{C["text"]};">'
        f'<span style="color:{C["accent"]};">&#11042;</span> Portfolio Risk Analysis</h1>'
        f'<p style="margin:0;font-size:12px;color:{C["muted"]};">'
        f"Beta · Jensen's Alpha · Correlation Matrix · Risk Flags</p></div>"
    )
    return (
        divider + title
        + _build_beta_alpha_table(risk)
        + _build_correlation_matrix(risk)
        + _build_portfolio_summary(risk)
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_html_report(
    quotes: dict[str, dict],
    risk: RiskReport,
    techs: dict[str, TechnicalSignal] | None = None,
    news_map: dict[str, list[dict]] | None = None,
    report_date: datetime | None = None,
) -> str:
    """Return the complete HTML email string."""
    if report_date is None:
        report_date = datetime.now(timezone.utc)
    if techs is None:
        techs = {}
    if news_map is None:
        news_map = {}

    date_str = report_date.strftime("%A, %B %-d, %Y")
    time_str = report_date.strftime("%H:%M UTC")

    dashboard = _build_decision_dashboard(techs, quotes) if techs else ""
    snapshot  = _build_price_snapshot(quotes)
    cards     = _build_all_stock_cards(techs, news_map) if techs else ""
    risk_sec  = _build_portfolio_risk_section(risk)

    footer = (
        f'<p style="margin:0;font-size:11px;color:{C["muted"]};text-align:center;">'
        f'Data: Yahoo Finance via yfinance · Generated {time_str} · '
        f'Watchlist: {", ".join(WATCHLIST)} · Benchmark: ^GSPC</p>'
        f'<p style="margin:8px 0 0;font-size:10px;color:{C["muted"]};text-align:center;">'
        f'For informational purposes only. Not financial advice.</p>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Daily Stock Report &#8212; {date_str}</title>
</head>
<body style="margin:0;padding:0;background:{C['bg']};font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div style="max-width:900px;margin:0 auto;padding:24px 16px;">

    <div style="background:linear-gradient(135deg,{C['header1']},{C['header2']});
                border:1px solid {C['border']};border-radius:12px;
                padding:28px 32px;margin-bottom:28px;text-align:center;">
      <div style="font-size:11px;color:{C['accent']};text-transform:uppercase;
                  letter-spacing:0.12em;font-weight:700;margin-bottom:8px;">
        Daily Market Report
      </div>
      <h1 style="margin:0;font-size:26px;font-weight:800;color:{C['text']};">{date_str}</h1>
      <p style="margin:8px 0 0;font-size:13px;color:{C['muted']};">
        After-market analysis · {time_str}
      </p>
    </div>

    <div style="background:{C['card']};border:1px solid {C['border']};
                border-radius:12px;padding:28px 32px;margin-bottom:20px;">
      {dashboard}
      {snapshot}
      {cards}
      {risk_sec}
    </div>

    <div style="padding:0 8px;">{footer}</div>
  </div>
</body>
</html>"""
