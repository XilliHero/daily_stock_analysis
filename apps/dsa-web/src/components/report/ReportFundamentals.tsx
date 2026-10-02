import type React from 'react';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { analysisApi } from '../../api/analysis';
import type { IntrinsicValue, StockFundamentals } from '../../types/analysis';
import { Card } from '../common';
import { DashboardPanelHeader } from '../dashboard';

interface ReportFundamentalsProps {
  stockCode?: string;
}

type Status = 'good' | 'warn' | null;

const num = (v?: number | null): string =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(2);

const pct = (v?: number | null): string =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : `${Number(v).toFixed(2)}%`;

const money = (v?: number | null): string =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : `$${Number(v).toFixed(2)}`;

const signedPct = (v?: number | null): string =>
  v === null || v === undefined || Number.isNaN(v)
    ? '—'
    : `${v > 0 ? '+' : ''}${Number(v).toFixed(1)}%`;

const cap = (v?: number | null): string => {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const abs = Math.abs(v);
  if (abs >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `$${(v / 1e6).toFixed(2)}M`;
  return `$${v.toFixed(0)}`;
};

// Assess a value against the general rule-of-thumb target (see the Ratios tab).
// Returns 'good' (meets), 'warn' (outside), or null (no clear good/bad, e.g. size).
const assess = (key: keyof StockFundamentals, v?: number | null): Status => {
  if (v === null || v === undefined || Number.isNaN(v)) return null;
  switch (key) {
    case 'peRatio':
    case 'forwardPe':
      if (v <= 0) return 'warn';
      if (v <= 20) return 'good';
      return v <= 30 ? null : 'warn';
    case 'pbRatio':
      if (v <= 0) return 'warn';
      if (v <= 3) return 'good';
      return v <= 6 ? null : 'warn';
    case 'psRatio':
      if (v <= 0) return 'warn';
      if (v <= 2) return 'good';
      return v <= 6 ? null : 'warn';
    case 'dividendYield':
      if (v >= 2 && v <= 6) return 'good';
      return v > 6 ? 'warn' : null; // 0–2% is fine (e.g. growth stocks)
    case 'roe':
      if (v >= 15) return 'good';
      return v >= 10 ? null : 'warn';
    case 'profitMargin':
      if (v >= 10) return 'good';
      return v >= 0 ? null : 'warn';
    case 'operatingMargin':
      if (v >= 12) return 'good';
      return v >= 0 ? null : 'warn';
    case 'revenueGrowth':
    case 'earningsGrowth':
      if (v >= 10) return 'good';
      return v >= 0 ? null : 'warn';
    case 'eps':
      return v > 0 ? 'good' : 'warn';
    case 'debtToEquity':
      if (v < 0) return null;
      if (v <= 100) return 'good';
      return v <= 200 ? null : 'warn';
    default:
      return null; // marketCap and anything else: size/no good-bad
  }
};

const StatusMark = ({ status }: { status: Status }) => {
  if (!status) return null;
  const good = status === 'good';
  const title = good ? 'Meets the general target' : 'Outside the general target';
  return (
    <span
      role="img"
      aria-label={title}
      className="ml-1 text-sm"
      style={{ color: good ? 'hsl(var(--success))' : 'hsl(var(--warning))' }}
    >
      {good ? '✓' : '⚠'}
    </span>
  );
};

const Item = ({ label, value, status }: { label: string; value: string; status: Status }) => (
  <div className="home-subpanel p-3">
    <div className="flex flex-col">
      <span className="home-strategy-label mb-0.5 text-xs">{label}</span>
      <span
        className="home-strategy-value flex items-center text-lg font-bold font-mono"
        style={value === '—' ? { color: 'var(--text-muted-text)' } : undefined}
      >
        {value}
        <StatusMark status={status} />
      </span>
    </div>
  </div>
);

const VERDICT_META: Record<string, { label: string; color: string }> = {
  undervalued: { label: 'Undervalued', color: 'hsl(var(--success))' },
  fair: { label: 'Fairly valued', color: 'hsl(var(--primary))' },
  overvalued: { label: 'Overvalued', color: 'hsl(var(--warning))' },
};

/**
 * Intrinsic-value block — a conservative DCF fair value with a Graham
 * cross-check and the margin of safety vs the current price. Rendered only
 * when the backend could compute an estimate (US/Canada stocks).
 */
const IntrinsicValueBlock = ({ iv }: { iv: IntrinsicValue }) => {
  const verdict = iv.verdict ? VERDICT_META[iv.verdict] : null;
  const upsidePositive = (iv.upsidePct ?? 0) >= 0;
  const upsideColor = upsidePositive ? 'hsl(var(--success))' : 'hsl(var(--warning))';
  const a = iv.assumptions;

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <div className="label-uppercase text-xs text-secondary-text">Intrinsic Value</div>
        {verdict && (
          <span
            className="rounded-full px-2 py-0.5 text-xs font-semibold"
            style={{ color: verdict.color, backgroundColor: `color-mix(in srgb, ${verdict.color} 15%, transparent)` }}
          >
            {verdict.label}
          </span>
        )}
      </div>
      <div className="home-subpanel p-3">
        <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1">
          <div className="flex flex-col">
            <span className="home-strategy-label text-xs">Fair value / share</span>
            <span className="home-strategy-value text-2xl font-bold font-mono">{money(iv.fairValue)}</span>
          </div>
          {iv.currentPrice != null && (
            <div className="flex flex-col">
              <span className="home-strategy-label text-xs">Current price</span>
              <span className="home-strategy-value text-lg font-bold font-mono">{money(iv.currentPrice)}</span>
            </div>
          )}
          {iv.upsidePct != null && (
            <div className="flex flex-col">
              <span className="home-strategy-label text-xs">Upside vs price</span>
              <span className="text-lg font-bold font-mono" style={{ color: upsideColor }}>
                {signedPct(iv.upsidePct)}
              </span>
            </div>
          )}
          {iv.marginOfSafetyPct != null && (
            <div className="flex flex-col">
              <span className="home-strategy-label text-xs">Margin of safety</span>
              <span className="home-strategy-value text-lg font-bold font-mono">{signedPct(iv.marginOfSafetyPct)}</span>
            </div>
          )}
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-secondary-text">
          <span>
            DCF <span className="font-mono text-primary-text">{money(iv.dcf)}</span>
          </span>
          <span>
            Graham <span className="font-mono text-primary-text">{money(iv.graham)}</span>
          </span>
          {iv.agreement === 'diverge' && (
            <span style={{ color: 'hsl(var(--warning))' }}>⚠ models disagree — treat as a wide range</span>
          )}
          {iv.agreement === 'agree' && <span style={{ color: 'hsl(var(--success))' }}>✓ models broadly agree</span>}
        </div>
        <p className="mt-2 text-xs text-secondary-text">
          Conservative estimate — growth capped at {a.growthRatePct}%, {a.discountRatePct}% discount rate,{' '}
          {a.terminalGrowthPct}% terminal growth over {a.projectionYears} yrs. Not a price target. See{' '}
          <Link to="/ratios" className="text-[hsl(var(--primary))] hover:underline">
            Ratios
          </Link>
          .
        </p>
      </div>
    </div>
  );
};

/**
 * Fundamentals card — an intrinsic-value estimate plus valuation, profitability,
 * growth and size ratios for the analysed stock, each flagged against a general
 * rule-of-thumb target. Data is fetched live by ticker, so it works for fresh
 * and historical reports alike.
 */
export const ReportFundamentals: React.FC<ReportFundamentalsProps> = ({ stockCode }) => {
  const [data, setData] = useState<StockFundamentals | null>(null);
  const [intrinsic, setIntrinsic] = useState<IntrinsicValue | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!stockCode) {
      setLoading(false);
      return;
    }
    let active = true;
    setLoading(true);
    analysisApi
      .getFundamentals(stockCode)
      .then((res) => {
        if (active) {
          setData(res.fundamentals || null);
          setIntrinsic(res.intrinsicValue || null);
        }
      })
      .catch(() => {
        if (active) {
          setData(null);
          setIntrinsic(null);
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [stockCode]);

  if (!stockCode) return null;

  const hasData = data && Object.values(data).some((v) => v !== null && v !== undefined);

  const item = (label: string, key: keyof StockFundamentals, fmt: (v?: number | null) => string) => ({
    label,
    value: fmt(data ? data[key] : null),
    status: assess(key, data ? data[key] : null),
  });

  const groups = data
    ? [
        {
          heading: 'Valuation',
          items: [
            item('P/E (TTM)', 'peRatio', num),
            item('P/E (Fwd)', 'forwardPe', num),
            item('P/B', 'pbRatio', num),
            item('P/S', 'psRatio', num),
            item('Dividend Yield', 'dividendYield', pct),
          ],
        },
        {
          heading: 'Profitability',
          items: [
            item('ROE', 'roe', pct),
            item('Profit Margin', 'profitMargin', pct),
            item('Operating Margin', 'operatingMargin', pct),
          ],
        },
        {
          heading: 'Growth',
          items: [
            item('Revenue Growth', 'revenueGrowth', pct),
            item('Earnings Growth', 'earningsGrowth', pct),
          ],
        },
        {
          heading: 'Size & Health',
          items: [
            item('Market Cap', 'marketCap', cap),
            item('EPS', 'eps', num),
            item('Debt / Equity', 'debtToEquity', num),
          ],
        },
      ]
    : [];

  return (
    <Card className="p-5">
      <DashboardPanelHeader
        title="Fundamentals"
        actions={
          <Link to="/ratios" className="text-xs text-[hsl(var(--primary))] hover:underline">
            What do these mean?
          </Link>
        }
      />
      {loading ? (
        <p className="text-sm text-secondary-text">Loading fundamentals…</p>
      ) : !hasData && !intrinsic ? (
        <p className="text-sm text-secondary-text">Fundamental ratios aren't available for this stock.</p>
      ) : (
        <div className="space-y-4">
          {intrinsic && <IntrinsicValueBlock iv={intrinsic} />}
          {groups.map((g) => (
            <div key={g.heading}>
              <div className="label-uppercase mb-2 text-xs text-secondary-text">{g.heading}</div>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
                {g.items.map((it) => (
                  <Item key={it.label} label={it.label} value={it.value} status={it.status} />
                ))}
              </div>
            </div>
          ))}
          {hasData && (
            <p className="pt-1 text-xs text-secondary-text">
              <span style={{ color: 'hsl(var(--success))' }}>✓</span> meets ·{' '}
              <span style={{ color: 'hsl(var(--warning))' }}>⚠</span> outside a general target (rule of thumb —
              varies by industry). See <Link to="/ratios" className="text-[hsl(var(--primary))] hover:underline">Ratios</Link>.
            </p>
          )}
        </div>
      )}
    </Card>
  );
};

export default ReportFundamentals;
