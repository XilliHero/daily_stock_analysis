import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { analysisApi } from '../../../api/analysis';
import type { SupportResistanceResponse } from '../../../types/analysis';
import { ReportSupportResistance } from '../ReportSupportResistance';

vi.mock('../../../api/analysis', () => ({
  analysisApi: {
    getSupportResistance: vi.fn(),
  },
}));

const renderCard = () =>
  render(
    <MemoryRouter>
      <ReportSupportResistance stockCode="AAPL" />
    </MemoryRouter>,
  );

describe('ReportSupportResistance', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the pivot ladder with nearest support and resistance', async () => {
    const res: SupportResistanceResponse = {
      code: 'AAPL',
      name: 'Apple Inc.',
      levels: {
        pivot: 100,
        r1: 110,
        r2: 120,
        r3: 130,
        s1: 90,
        s2: 80,
        s3: 70,
        currentPrice: 105,
        nearestSupport: 100,
        nearestResistance: 110,
        basisDate: '2026-10-06',
        period: 'daily',
      },
    };
    vi.mocked(analysisApi.getSupportResistance).mockResolvedValue(res);

    renderCard();

    // All seven levels render.
    expect(await screen.findByText('R3')).toBeInTheDocument();
    expect(screen.getByText('P')).toBeInTheDocument();
    expect(screen.getByText('S3')).toBeInTheDocument();
    // Basis session is surfaced.
    expect(screen.getByText(/from the 2026-10-06 session/)).toBeInTheDocument();
    // Nearest support/resistance summary present.
    expect(screen.getAllByText('nearest').length).toBe(2);
  });

  it('shows an empty state when no levels are available', async () => {
    vi.mocked(analysisApi.getSupportResistance).mockResolvedValue({
      code: 'AAPL',
      name: 'Apple Inc.',
      levels: null,
    });

    renderCard();

    expect(await screen.findByText(/Price levels aren't available/)).toBeInTheDocument();
  });
});
