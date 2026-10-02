import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { analysisApi } from '../../../api/analysis';
import type { FundamentalsResponse } from '../../../types/analysis';
import { ReportFundamentals } from '../ReportFundamentals';

vi.mock('../../../api/analysis', () => ({
  analysisApi: {
    getFundamentals: vi.fn(),
  },
}));

const renderCard = () =>
  render(
    <MemoryRouter>
      <ReportFundamentals stockCode="NKE" />
    </MemoryRouter>,
  );

describe('ReportFundamentals — intrinsic value', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the intrinsic-value block with verdict, models and margin of safety', async () => {
    const res: FundamentalsResponse = {
      code: 'NKE',
      name: 'Nike, Inc.',
      fundamentals: { peRatio: 16, eps: 2.1 },
      intrinsicValue: {
        fairValue: 90.5,
        method: 'dcf',
        currentPrice: 72.0,
        upsidePct: 25.7,
        marginOfSafetyPct: 20.4,
        verdict: 'undervalued',
        dcf: 90.5,
        graham: 85.0,
        grahamRevised: 120.0,
        agreement: 'agree',
        assumptions: {
          growthRatePct: 10,
          discountRatePct: 10,
          terminalGrowthPct: 2.5,
          projectionYears: 10,
        },
      },
    };
    vi.mocked(analysisApi.getFundamentals).mockResolvedValue(res);

    renderCard();

    expect(await screen.findByText('Undervalued')).toBeInTheDocument();
    // Fair value headline equals the DCF, so $90.50 appears twice (headline + cross-check).
    expect(screen.getAllByText('$90.50').length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText('$72.00')).toBeInTheDocument(); // current price
    expect(screen.getByText('+25.7%')).toBeInTheDocument(); // upside
    expect(screen.getByText('$85.00')).toBeInTheDocument(); // Graham cross-check
    expect(screen.getByText(/models broadly agree/)).toBeInTheDocument();
    expect(screen.getByText(/growth capped at 10%/)).toBeInTheDocument();
  });

  it('omits the intrinsic-value block when none is provided', async () => {
    const res: FundamentalsResponse = {
      code: 'NKE',
      name: 'Nike, Inc.',
      fundamentals: { peRatio: 16 },
      intrinsicValue: null,
    };
    vi.mocked(analysisApi.getFundamentals).mockResolvedValue(res);

    renderCard();

    // Ratios still render, but no intrinsic-value heading.
    expect(await screen.findByText('Fundamentals')).toBeInTheDocument();
    expect(screen.queryByText('Intrinsic Value')).not.toBeInTheDocument();
  });

  it('flags divergence between the DCF and Graham models', async () => {
    const res: FundamentalsResponse = {
      code: 'XYZ',
      name: 'XYZ Corp',
      fundamentals: {},
      intrinsicValue: {
        fairValue: 50,
        method: 'dcf',
        currentPrice: 48,
        upsidePct: 4.2,
        marginOfSafetyPct: 4.0,
        verdict: 'fair',
        dcf: 50,
        graham: 120,
        grahamRevised: 90,
        agreement: 'diverge',
        assumptions: {
          growthRatePct: 5,
          discountRatePct: 10,
          terminalGrowthPct: 2.5,
          projectionYears: 10,
        },
      },
    };
    vi.mocked(analysisApi.getFundamentals).mockResolvedValue(res);

    renderCard();

    expect(await screen.findByText('Fairly valued')).toBeInTheDocument();
    expect(screen.getByText(/models disagree/)).toBeInTheDocument();
  });
});
