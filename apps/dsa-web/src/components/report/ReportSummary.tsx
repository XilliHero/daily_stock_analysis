import React from 'react';
import type { AnalysisResult, AnalysisReport } from '../../types/analysis';
import { ReportOverview } from './ReportOverview';
import { ReportStrategy } from './ReportStrategy';
import { ReportNews } from './ReportNews';
import { ReportDetails } from './ReportDetails';
import { getReportText, normalizeReportLanguage } from '../../utils/reportLanguage';

interface ReportSummaryProps {
  data: AnalysisResult | AnalysisReport;
  isHistory?: boolean;
}

/**
 * Full report viewer component
 * Combines the overview, strategy, news, and detail sections
 */
export const ReportSummary: React.FC<ReportSummaryProps> = ({
  data,
  isHistory = false,
}) => {
  // Handle both AnalysisResult and AnalysisReport data shapes
  const report: AnalysisReport = 'report' in data ? data.report : data;
  // Use the report id, since queryId can repeat in batch analysis and the detail endpoint needs recordId to fetch linked news and detail data
  const recordId = report.meta.id;

  const { meta, summary, strategy, details } = report;
  const reportLanguage = normalizeReportLanguage(meta.reportLanguage);
  const text = getReportText(reportLanguage);
  const modelUsed = (meta.modelUsed || '').trim();
  const shouldShowModel = Boolean(
    modelUsed && !['unknown', 'error', 'none', 'null', 'n/a'].includes(modelUsed.toLowerCase()),
  );

  return (
    <div className="space-y-5 pb-8 animate-fade-in">
      {/* Overview (above the fold) */}
      <ReportOverview
        meta={meta}
        summary={summary}
        details={details}
        isHistory={isHistory}
      />

      {/* Strategy levels */}
      <ReportStrategy strategy={strategy} language={reportLanguage} />

      {/* News */}
      <ReportNews recordId={recordId} limit={8} language={reportLanguage} />

      {/* Transparency & lineage */}
      <ReportDetails details={details} recordId={recordId} language={reportLanguage} />

      {/* Analysis model tag (Issue #528) — end of report */}
      {shouldShowModel && (
        <p className="px-1 text-xs text-muted-text">
          {text.analysisModel}: {modelUsed}
        </p>
      )}
    </div>
  );
};
