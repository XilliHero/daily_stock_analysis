interface ValidationResult {
  valid: boolean;
  message?: string;
  normalized: string;
}

// Latin letters, digits, dot, hyphen, colon and spaces — enough for US/Canada
// tickers and company-name queries. (Chinese A-share / HK support was removed,
// so CJK characters are no longer accepted.)
const SUPPORTED_QUERY_CHARACTERS = /^[A-Z0-9.\-:\s]+$/;

// Chinese A-share / Hong Kong ticker shapes this app no longer supports.
const UNSUPPORTED_MARKET_PATTERNS = [
  /^\d{6}$/, // A-share 6-digit code
  /^(SH|SZ|BJ)\d{6}$/, // A-share code with exchange prefix
  /^\d{6}\.(SH|SZ|SS|BJ)$/, // A-share code with exchange suffix
  /^\d{5}$/, // HK code without prefix
  /^HK\d{1,5}$/, // HK-prefixed code, e.g. HK00700
  /^\d{1,5}\.HK$/, // HK suffix format, e.g. 00700.HK
];

// Supported markets: US and Canada (plus Yahoo crypto pairs like BTC-USD).
const STOCK_CODE_PATTERNS = [
  /^[A-Z]{1,5}(?:\.(?:US|[A-Z]))?$/, // US ticker, optional .US or class suffix (e.g. BRK.B)
  /^[A-Z]{1,5}\.(?:TO|V)$/, // Canada: Toronto (.TO) / TSX Venture (.V)
  /^TSE:[A-Z]{1,5}$/, // Canada: TSE: prefix
  /^[A-Z]{1,5}-[A-Z]{2,4}$/, // Yahoo pairs, e.g. BTC-USD
];

/**
 * US/Canada stocks only — this message is shown when an A-share or Hong Kong
 * ticker is entered.
 */
export const UNSUPPORTED_MARKET_MESSAGE =
  'This app covers US and Canada stocks only. Chinese A-share and Hong Kong tickers aren’t supported.';

/**
 * Check whether the input looks like a supported (US/Canada) stock code.
 */
export const looksLikeStockCode = (value: string): boolean => {
  const normalized = value.trim().toUpperCase();
  return STOCK_CODE_PATTERNS.some((regex) => regex.test(normalized));
};

/**
 * Detect Chinese A-share / Hong Kong ticker shapes, which are no longer
 * supported (US/Canada only).
 */
export const isUnsupportedMarketCode = (value: string): boolean => {
  const normalized = value.trim().toUpperCase();
  return UNSUPPORTED_MARKET_PATTERNS.some((regex) => regex.test(normalized));
};

/**
 * Validate US and Canada stock code formats; reject A-share/HK codes explicitly.
 */
export const validateStockCode = (value: string): ValidationResult => {
  const normalized = value.trim().toUpperCase();

  if (!normalized) {
    return { valid: false, message: 'Enter a stock code', normalized };
  }

  if (isUnsupportedMarketCode(normalized)) {
    return { valid: false, message: UNSUPPORTED_MARKET_MESSAGE, normalized };
  }

  const valid = looksLikeStockCode(normalized);

  return {
    valid,
    message: valid ? undefined : 'Invalid stock code format',
    normalized,
  };
};

/**
 * Reject obviously invalid free-text queries before they reach the backend.
 */
export const isObviouslyInvalidStockQuery = (value: string): boolean => {
  const normalized = value.trim().toUpperCase();

  if (!normalized || looksLikeStockCode(normalized)) {
    return false;
  }

  if (!SUPPORTED_QUERY_CHARACTERS.test(normalized)) {
    return true;
  }

  const hasLetters = /[A-Z]/.test(normalized);
  const hasDigits = /\d/.test(normalized);

  return hasLetters && hasDigits;
};
