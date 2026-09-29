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

const Item = ({ label, value }: { label: string; value: string }) => (
  <div className="home-subpanel p-3">
    <div className="flex flex-col">
      <span className="home-strategy-label mb-0.5 text-xs">{label}</span>
      <span
        className="home-strategy-value text-lg font-bold font-mono"
        style={value === '—' ? { color: 'var(--text-muted-text)' } : undefined}
      >
        {value}
      </span>
    </div>
  </div>
);

/**
 * Fundamentals card — valuation, profitability, growth and size ratios for the
 * analysed stock. Data is fetched live (current ratios) by ticker, so it works
 * for both fresh and historical reports.
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

  const groups: { heading: string; items: { label: string; value: string }[] }[] = data
    ? [
        {
          heading: 'Valuation',
          items: [
            { label: 'P/E (TTM)', value: num(data.peRatio) },
            { label: 'P/E (Fwd)', value: num(data.forwardPe) },
            { label: 'P/B', value: num(data.pbRatio) },
            { label: 'P/S', value: num(data.psRatio) },
            { label: 'Dividend Yield', value: pct(data.dividendYield) },
          ],
        },
        {
          heading: 'Profitability',
          items: [
            { label: 'ROE', value: pct(data.roe) },
            { label: 'Profit Margin', value: pct(data.profitMargin) },
            { label: 'Operating Margin', value: pct(data.operatingMargin) },
          ],
        },
        {
          heading: 'Growth',
          items: [
            { label: 'Revenue Growth', value: pct(data.revenueGrowth) },
            { label: 'Earnings Growth', value: pct(data.earningsGrowth) },
          ],
        },
        {
          heading: 'Size & Health',
          items: [
            { label: 'Market Cap', value: cap(data.marketCap) },
            { label: 'EPS', value: num(data.eps) },
            { label: 'Debt / Equity', value: num(data.debtToEquity) },
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
                  <Item key={it.label} label={it.label} value={it.value} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
};

export default ReportFundamentals;
