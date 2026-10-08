/**
 * Analysis-related type definitions.
 * Aligned with the API schema.
 */

// ============ Request Types ============

export interface AnalysisRequest {
  stockCode?: string;
  stockCodes?: string[];
  reportType?: 'simple' | 'detailed' | 'full' | 'brief';
  forceRefresh?: boolean;
  asyncMode?: boolean;
  stockName?: string;
  originalQuery?: string;
  selectionSource?: 'manual' | 'autocomplete' | 'import' | 'image';
  notify?: boolean;
}

// ============ Report Types ============

export type ReportLanguage = 'zh' | 'en';

/** Report metadata */
export interface ReportMeta {
  id?: number;  // Analysis history record ID, present for persisted reports
  queryId: string;
  stockCode: string;
  stockName: string;
  reportType: 'simple' | 'detailed' | 'full' | 'brief';
  reportLanguage?: ReportLanguage;
  createdAt: string;
  currentPrice?: number;
  changePct?: number;
  modelUsed?: string;  // LLM model used for analysis
}

/** Sentiment label */
export type SentimentLabel =
  | 'Extreme fear'
  | 'Bearish'
  | 'Neutral'
  | 'Bullish'
  | 'Extreme greed'
  | 'Very Bearish'
  | 'Bearish'
  | 'Neutral'
  | 'Bullish'
  | 'Very Bullish';

/** Report summary section */
export interface ReportSummary {
  analysisSummary: string;
  operationAdvice: string;
  trendPrediction: string;
  sentimentScore: number;
  sentimentLabel?: SentimentLabel;
}

/** Strategy section */
export interface ReportStrategy {
  idealBuy?: string;
  secondaryBuy?: string;
  stopLoss?: string;
  takeProfit?: string;
}

export interface RelatedBoard {
  name: string;
  code?: string;
  type?: string;
}

export interface SectorRankingItem {
  name: string;
  changePct?: number;
}

export interface SectorRankings {
  top?: SectorRankingItem[];
  bottom?: SectorRankingItem[];
}

/** Details section */
export interface ReportDetails {
  newsContent?: string;
  rawResult?: Record<string, unknown>;
  contextSnapshot?: Record<string, unknown>;
  financialReport?: Record<string, unknown>;
  dividendMetrics?: Record<string, unknown>;
  belongBoards?: RelatedBoard[];
  sectorRankings?: SectorRankings;
}

/** Full analysis report */
export interface AnalysisReport {
  meta: ReportMeta;
  summary: ReportSummary;
  strategy?: ReportStrategy;
  details?: ReportDetails;
}

/** Fundamental ratios for a stock (camelCased from the API) */
export interface StockFundamentals {
  peRatio?: number | null;
  forwardPe?: number | null;
  pbRatio?: number | null;
  psRatio?: number | null;
  dividendYield?: number | null;
  roe?: number | null;
  profitMargin?: number | null;
  operatingMargin?: number | null;
  revenueGrowth?: number | null;
  earningsGrowth?: number | null;
  marketCap?: number | null;
  eps?: number | null;
  debtToEquity?: number | null;
}

export interface IntrinsicValueAssumptions {
  growthRatePct: number;
  discountRatePct: number;
  terminalGrowthPct: number;
  projectionYears: number;
}

export interface IntrinsicValue {
  /** Headline per-share fair value (DCF when available, else Graham). */
  fairValue: number;
  /** Which model produced the headline. */
  method: 'dcf' | 'graham';
  currentPrice?: number | null;
  /** (fairValue - price) / price, as a percentage. */
  upsidePct?: number | null;
  /** (fairValue - price) / fairValue, as a percentage (margin of safety). */
  marginOfSafetyPct?: number | null;
  verdict?: 'undervalued' | 'fair' | 'overvalued' | null;
  /** Discounted-cash-flow fair value (null if not computable). */
  dcf?: number | null;
  /** Graham Number cross-check. */
  graham?: number | null;
  /** Graham's revised growth-formula value. */
  grahamRevised?: number | null;
  /** Whether DCF and Graham broadly agree. */
  agreement?: 'agree' | 'diverge' | null;
  assumptions: IntrinsicValueAssumptions;
}

export interface FundamentalsResponse {
  code: string;
  name?: string;
  fundamentals: StockFundamentals;
  intrinsicValue?: IntrinsicValue | null;
}

export interface PivotLevels {
  pivot: number;
  r1: number;
  r2: number;
  r3: number;
  s1: number;
  s2: number;
  s3: number;
  currentPrice: number;
  /** Highest level below the price. */
  nearestSupport?: number | null;
  /** Lowest level above the price. */
  nearestResistance?: number | null;
  /** The session the pivots are based on (YYYY-MM-DD), when known. */
  basisDate?: string | null;
  period: string;
}

export interface SupportResistanceResponse {
  code: string;
  name?: string;
  levels?: PivotLevels | null;
}

// ============ Analysis Result Types ============

/** Sync analysis response */
export interface AnalysisResult {
  queryId: string;
  stockCode: string;
  stockName: string;
  report: AnalysisReport;
  createdAt: string;
}

/** Async task accepted response */
export interface TaskAccepted {
  taskId: string;
  status: 'pending' | 'processing';
  message?: string;
}

export interface BatchTaskAcceptedItem {
  taskId: string;
  stockCode: string;
  status: 'pending' | 'processing';
  message?: string;
}

export interface BatchDuplicateTaskItem {
  stockCode: string;
  existingTaskId: string;
  message: string;
}

export interface BatchTaskAcceptedResponse {
  accepted: BatchTaskAcceptedItem[];
  duplicates: BatchDuplicateTaskItem[];
  message: string;
}

export type AnalyzeAsyncResponse = TaskAccepted | BatchTaskAcceptedResponse;

export type AnalyzeResponse = AnalysisResult | AnalyzeAsyncResponse;

/** Task status */
export interface TaskStatus {
  taskId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress?: number;
  result?: AnalysisResult;
  error?: string;
  stockName?: string;
  originalQuery?: string;
  selectionSource?: string;
}

/** Task details used by task list and SSE events */
export interface TaskInfo {
  taskId: string;
  stockCode: string;
  stockName?: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  message?: string;
  reportType: string;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  error?: string;
  originalQuery?: string;
  selectionSource?: string;
}

/** Task list response */
export interface TaskListResponse {
  total: number;
  pending: number;
  processing: number;
  tasks: TaskInfo[];
}

/** Duplicate task error response */
export interface DuplicateTaskError {
  error: 'duplicate_task';
  message: string;
  stockCode: string;
  existingTaskId: string;
}

// ============ History Types ============

/** History item summary */
export interface HistoryItem {
  id: number;  // Record primary key ID, always present for persisted history items
  queryId: string;  // Linked analysis query ID
  stockCode: string;
  stockName?: string;
  reportType?: string;
  sentimentScore?: number;
  operationAdvice?: string;
  createdAt: string;
}

/** History list response */
export interface HistoryListResponse {
  total: number;
  page: number;
  limit: number;
  items: HistoryItem[];
}

/** News item */
export interface NewsIntelItem {
  title: string;
  snippet: string;
  url: string;
}

/** News response */
export interface NewsIntelResponse {
  total: number;
  items: NewsIntelItem[];
}

/** History filter parameters */
export interface HistoryFilters {
  stockCode?: string;
  startDate?: string;
  endDate?: string;
}

/** History pagination parameters */
export interface HistoryPagination {
  page: number;
  limit: number;
}

// ============ Error Types ============

export interface ApiError {
  error: string;
  message: string;
  detail?: Record<string, unknown>;
}

// ============ Helper Functions ============

/** Get sentiment label by score */
export const getSentimentLabel = (score: number, language: ReportLanguage = 'zh'): SentimentLabel => {
  if (language === 'en') {
    if (score <= 20) return 'Very Bearish';
    if (score <= 40) return 'Bearish';
    if (score <= 60) return 'Neutral';
    if (score <= 80) return 'Bullish';
    return 'Very Bullish';
  }
  if (score <= 20) return 'Extreme fear';
  if (score <= 40) return 'Bearish';
  if (score <= 60) return 'Neutral';
  if (score <= 80) return 'Bullish';
  return 'Extreme greed';
};

/** Get sentiment color by score */
export const getSentimentColor = (score: number): string => {
  if (score <= 20) return '#ef4444'; // red-500
  if (score <= 40) return '#f97316'; // orange-500
  if (score <= 60) return '#eab308'; // yellow-500
  if (score <= 80) return '#22c55e'; // green-500
  return '#10b981'; // emerald-500
};
