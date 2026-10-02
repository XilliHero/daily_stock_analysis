import type React from 'react';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { analysisApi } from '../../api/analysis';
import type { StockFundamentals } from '../../types/analysis';
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

/**
 * Fundamentals card — valuation, profitability, growth and size ratios for the
 * analysed stock, each flagged against a general rule-of-thumb target. Data is
 * fetched live by ticker, so it works for fresh and historical reports alike.
 */
export const ReportFundamentals: React.FC<ReportFundamentalsProps> = ({ stockCode }) => {
  const [data, setData] = useState<StockFundamentals | null>(null);
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
        if (active) setData(res.fundamentals || null);
      })
      .catch(() => {
        if (active) setData(null);
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
      ) : !hasData ? (
        <p className="text-sm text-secondary-text">Fundamental ratios aren't available for this stock.</p>
      ) : (
        <div className="space-y-4">
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
          <p className="pt-1 text-xs text-secondary-text">
            <span style={{ color: 'hsl(var(--success))' }}>✓</span> meets ·{' '}
            <span style={{ color: 'hsl(var(--warning))' }}>⚠</span> outside a general target (rule of thumb —
            varies by industry). See <Link to="/ratios" className="text-[hsl(var(--primary))] hover:underline">Ratios</Link>.
          </p>
        </div>
      )}
    </Card>
  );
};

export default ReportFundamentals;
