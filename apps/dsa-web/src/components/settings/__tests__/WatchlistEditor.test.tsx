import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { WatchlistEditor } from '../WatchlistEditor';

const { validateSymbol, update, onSaved } = vi.hoisted(() => ({
  validateSymbol: vi.fn(),
  update: vi.fn(),
  onSaved: vi.fn(),
}));

vi.mock('../../../api/watchlist', () => ({
  watchlistApi: { validateSymbol },
}));

vi.mock('../../../api/systemConfig', async () => {
  const actual =
    await vi.importActual<typeof import('../../../api/systemConfig')>('../../../api/systemConfig');
  return {
    ...actual,
    systemConfigApi: { ...actual.systemConfigApi, update },
  };
});

function renderEditor(stockListValue = 'XLE, OKLO') {
  return render(
    <WatchlistEditor
      stockListValue={stockListValue}
      configVersion="v1"
      maskToken="******"
      onSaved={onSaved}
    />,
  );
}

describe('WatchlistEditor', () => {
  beforeEach(() => {
    validateSymbol.mockReset();
    update.mockReset();
    onSaved.mockReset();
    update.mockResolvedValue({ configVersion: 'v2' });
  });

  it('renders existing tickers as chips', () => {
    renderEditor();
    expect(screen.getByText('XLE')).toBeInTheDocument();
    expect(screen.getByText('OKLO')).toBeInTheDocument();
  });

  it('adds a valid ticker and persists STOCK_LIST', async () => {
    validateSymbol.mockResolvedValue({ symbol: 'NVDA', valid: true, price: 132 });
    renderEditor();

    fireEvent.change(screen.getByLabelText('Add ticker'), { target: { value: 'nvda' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add' }));

    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    expect(validateSymbol).toHaveBeenCalledWith('NVDA');
    expect(update).toHaveBeenCalledWith(
      expect.objectContaining({
        items: [{ key: 'STOCK_LIST', value: 'XLE, OKLO, NVDA' }],
      }),
    );
    expect(onSaved).toHaveBeenCalledWith('XLE, OKLO, NVDA');
  });

  it('rejects an invalid ticker without saving', async () => {
    validateSymbol.mockResolvedValue({ symbol: 'ZZZZ', valid: false, price: null });
    renderEditor();

    fireEvent.change(screen.getByLabelText('Add ticker'), { target: { value: 'ZZZZ' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add' }));

    await waitFor(() => expect(screen.getByText(/didn't return market data/i)).toBeInTheDocument());
    expect(update).not.toHaveBeenCalled();
  });

  it('rejects a duplicate without calling the API', async () => {
    renderEditor();
    fireEvent.change(screen.getByLabelText('Add ticker'), { target: { value: 'xle' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add' }));

    await waitFor(() => expect(screen.getByText(/already in your watchlist/i)).toBeInTheDocument());
    expect(validateSymbol).not.toHaveBeenCalled();
    expect(update).not.toHaveBeenCalled();
  });

  it('removes a ticker and persists the shorter list', async () => {
    renderEditor();
    fireEvent.click(screen.getByRole('button', { name: 'Remove OKLO' }));

    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    expect(update).toHaveBeenCalledWith(
      expect.objectContaining({ items: [{ key: 'STOCK_LIST', value: 'XLE' }] }),
    );
    expect(onSaved).toHaveBeenCalledWith('XLE');
  });
});
