import { useCallback, useMemo, useState } from 'react';
import { getParsedApiError } from '../../api/error';
import { systemConfigApi, SystemConfigConflictError } from '../../api/systemConfig';
import { watchlistApi } from '../../api/watchlist';
import { Button, InlineAlert, Input } from '../common';

interface WatchlistEditorProps {
  stockListValue: string;
  configVersion: string;
  maskToken: string;
  onSaved: (newValue: string) => void | Promise<void>;
  disabled?: boolean;
}

/** Parse a comma-separated STOCK_LIST into a de-duplicated, upper-cased list. */
export function parseWatchlist(value: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of (value ?? '').split(',')) {
    const t = raw.trim().toUpperCase();
    if (t && !seen.has(t)) {
      seen.add(t);
      out.push(t);
    }
  }
  return out;
}

export function WatchlistEditor({
  stockListValue,
  configVersion,
  maskToken,
  onSaved,
  disabled = false,
}: WatchlistEditorProps) {
  const tickers = useMemo(() => parseWatchlist(stockListValue), [stockListValue]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  const persist = useCallback(
    async (next: string[]) => {
      const value = next.join(', ');
      await systemConfigApi.update({
        configVersion,
        maskToken,
        items: [{ key: 'STOCK_LIST', value }],
      });
      await onSaved(value);
    },
    [configVersion, maskToken, onSaved],
  );

  const reportError = useCallback((e: unknown) => {
    if (e instanceof SystemConfigConflictError) {
      setError('Settings changed elsewhere — refresh the page and try again.');
    } else {
      setError(getParsedApiError(e).message);
    }
  }, []);

  const handleAdd = useCallback(async () => {
    const sym = input.trim().toUpperCase();
    setError(null);
    setInfo(null);
    if (!sym) return;
    if (tickers.includes(sym)) {
      setError(`${sym} is already in your watchlist.`);
      return;
    }
    setBusy(true);
    try {
      const res = await watchlistApi.validateSymbol(sym);
      if (!res.valid) {
        setError(`"${sym}" didn't return market data — double-check the symbol.`);
        return;
      }
      await persist([...tickers, sym]);
      setInput('');
      setInfo(`${sym} added.`);
    } catch (e) {
      reportError(e);
    } finally {
      setBusy(false);
    }
  }, [input, tickers, persist, reportError]);

  const handleRemove = useCallback(
    async (sym: string) => {
      setError(null);
      setInfo(null);
      setBusy(true);
      try {
        await persist(tickers.filter((t) => t !== sym));
        setInfo(`${sym} removed.`);
      } catch (e) {
        reportError(e);
      } finally {
        setBusy(false);
      }
    },
    [tickers, persist, reportError],
  );

  const isDisabled = disabled || busy || !configVersion;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2" data-testid="watchlist-chips">
        {tickers.length === 0 ? (
          <span className="text-sm text-muted-foreground">No tickers yet — add one below.</span>
        ) : (
          tickers.map((t) => (
            <span
              key={t}
              className="inline-flex items-center gap-1 rounded-full border border-border px-3 py-1 text-sm"
            >
              {t}
              <button
                type="button"
                aria-label={`Remove ${t}`}
                disabled={isDisabled}
                onClick={() => handleRemove(t)}
                className="ml-1 leading-none text-muted-foreground hover:text-foreground disabled:opacity-50"
              >
                ×
              </button>
            </span>
          ))
        )}
      </div>

      <div className="flex gap-2">
        <Input
          value={input}
          disabled={isDisabled}
          placeholder="Add ticker (e.g. NVDA, BTC-USD, MDA.TO)"
          aria-label="Add ticker"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              void handleAdd();
            }
          }}
        />
        <Button
          variant="secondary"
          onClick={() => void handleAdd()}
          disabled={isDisabled || !input.trim()}
        >
          {busy ? 'Checking…' : 'Add'}
        </Button>
      </div>

      {error ? <InlineAlert variant="danger" message={error} /> : null}
      {info ? <InlineAlert variant="success" message={info} /> : null}
    </div>
  );
}

export default WatchlistEditor;
