import apiClient from './index';

export type ValidateSymbolResponse = {
  symbol: string;
  valid: boolean;
  price: number | null;
};

export const watchlistApi = {
  /** Check whether a ticker returns market data (via the backend yfinance path). */
  async validateSymbol(symbol: string): Promise<ValidateSymbolResponse> {
    const response = await apiClient.get('/api/v1/watchlist/validate', {
      params: { symbol },
      timeout: 15000, // yfinance can be slow
    });
    const data = response.data as { symbol?: string; valid?: boolean; price?: number | null };
    return {
      symbol: data.symbol ?? symbol.toUpperCase(),
      valid: Boolean(data.valid),
      price: data.price ?? null,
    };
  },
};
