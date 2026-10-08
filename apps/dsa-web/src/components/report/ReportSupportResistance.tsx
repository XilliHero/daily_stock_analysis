import type React from 'react';
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { analysisApi } from '../../api/analysis';
import type { PivotLevels } from '../../types/analysis';
import { Card } from '../common';
import { DashboardPanelHeader } from '../dashboard';

interface ReportSupportResistanceProps {
  stockCode?: string;
}

type LevelType = 'resistance' | 'pivot' | 'support';

const price = (v?: number | null): string =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(2);

const signedPct = (level: number, ref: number): string => {
  if (!ref) return '';
  const d = ((level - ref) / ref) * 100;
  return `${d > 0 ? '+' : ''}${d.toFixed(1)}%`;
};

const TYPE_COLOR: Record<LevelType, string> = {
  resistance: 'hsl(var(--warning))',
  pivot: 'hsl(var(--primary))',
  support: 'hsl(var(--success))',
};

interface Row {
  key: string;
  label: string;
  value: number;
  type: LevelType;
}

const LevelRow = ({ row, refPrice, nearest }: { row: Row; refPrice: number; nearest: boolean }) => {
  const color = TYPE_COLOR[row.type];
  return (
    <div
      className="flex items-center justify-between rounded-md px-3 py-1.5"
      style={
        nearest
          ? { backgroundColor: `color-mix(in srgb, ${color} 14%, transparent)`, border: `1px solid ${color}` }
          : { border: '1px solid transparent' }
      }
    >
      <div className="flex items-center gap-2">
        <span
          className="inline-block w-9 rounded px-1 text-center text-xs font-semibold"
          style={{ color, backgroundColor: `color-mix(in srgb, ${color} 15%, transparent)` }}
        >
          {row.label}
        </span>
        {nearest && <span className="text-xs text-secondary-text">nearest</span>}
      </div>
      <div className="flex items-baseline gap-3">
        <span className="font-mono text-sm font-bold">{price(row.value)}</span>
        <span className="w-14 text-right font-mono text-xs text-secondary-text">
          {signedPct(row.value, refPrice)}
        </span>
      </div>
    </div>
  );
};

/**
 * Support & Resistance card — classic pivot-point levels (pivot, R1–R3, S1–S3)
 * for the analysed stock, laid out as a ladder around the current price with
 * the nearest support and resistance highlighted. Fetched live by ticker, so
 * it works for any market and on historical reports.
 */
export const ReportSupportResistance: React.FC<ReportSupportResistanceProps> = ({ stockCode }) => {
  const [levels, setLevels] = useState<PivotLevels | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!stockCode) {
      setLoading(false);
      return;
    }
    let active = true;
    setLoading(true);
    analysisApi
      .getSupportResistance(stockCode)
      .then((res) => {
        if (active) setLevels(res.levels || null);
      })
      .catch(() => {
        if (active) setLevels(null);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [stockCode]);

  if (!stockCode) return null;

  const refPrice = levels?.currentPrice ?? 0;
  const rows: Row[] = levels
    ? [
        { key: 'r3', label: 'R3', value: levels.r3, type: 'resistance' },
        { key: 'r2', label: 'R2', value: levels.r2, type: 'resistance' },
        { key: 'r1', label: 'R1', value: levels.r1, type: 'resistance' },
        { key: 'pivot', label: 'P', value: levels.pivot, type: 'pivot' },
        { key: 's1', label: 'S1', value: levels.s1, type: 'support' },
        { key: 's2', label: 'S2', value: levels.s2, type: 'support' },
        { key: 's3', label: 'S3', value: levels.s3, type: 'support' },
      ]
    : [];

  return (
    <Card className="p-5">
      <DashboardPanelHeader
        title="Support & Resistance"
        actions={
          <Link to="/ratios" className="text-xs text-[hsl(var(--primary))] hover:underline">
            What do these mean?
          </Link>
        }
      />
      {loading ? (
        <p className="text-sm text-secondary-text">Loading levels…</p>
      ) : !levels ? (
        <p className="text-sm text-secondary-text">Price levels aren't available for this stock.</p>
      ) : (
        <div className="space-y-3">
          <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1 text-sm">
            <span className="text-secondary-text">
              Last close{' '}
              <span className="font-mono text-base font-bold text-primary-text">{price(levels.currentPrice)}</span>
            </span>
            {levels.nearestSupport != null && (
              <span className="text-secondary-text">
                Support{' '}
                <span className="font-mono font-bold" style={{ color: TYPE_COLOR.support }}>
                  {price(levels.nearestSupport)}
                </span>
              </span>
            )}
            {levels.nearestResistance != null && (
              <span className="text-secondary-text">
                Resistance{' '}
                <span className="font-mono font-bold" style={{ color: TYPE_COLOR.resistance }}>
                  {price(levels.nearestResistance)}
                </span>
              </span>
            )}
          </div>

          <div className="space-y-1">
            {rows.map((row, i) => {
              const nextBelow = rows[i + 1];
              const crossesPrice =
                row.value > refPrice && (!nextBelow || nextBelow.value <= refPrice);
              const nearest =
                (levels.nearestSupport != null && row.value === levels.nearestSupport) ||
                (levels.nearestResistance != null && row.value === levels.nearestResistance);
              return (
                <div key={row.key}>
                  <LevelRow row={row} refPrice={refPrice} nearest={nearest} />
                  {crossesPrice && (
                    <div className="my-1 flex items-center gap-2 px-3">
                      <div className="h-px flex-1" style={{ backgroundColor: 'hsl(var(--primary))' }} />
                      <span className="font-mono text-xs font-semibold text-[hsl(var(--primary))]">
                        Price {price(levels.currentPrice)}
                      </span>
                      <div className="h-px flex-1" style={{ backgroundColor: 'hsl(var(--primary))' }} />
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <p className="pt-1 text-xs text-secondary-text">
            Classic pivot points{levels.basisDate ? ` from the ${levels.basisDate} session` : ''}.{' '}
            <span style={{ color: TYPE_COLOR.resistance }}>R1–R3</span> are resistance above the price,{' '}
            <span style={{ color: TYPE_COLOR.support }}>S1–S3</span> support below it. Educational levels, not a
            trade signal. See{' '}
            <Link to="/ratios" className="text-[hsl(var(--primary))] hover:underline">
              Ratios
            </Link>
            .
          </p>
        </div>
      )}
    </Card>
  );
};

export default ReportSupportResistance;
