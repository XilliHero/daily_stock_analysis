import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import PlanPage from '../PlanPage';
import { advisorApi } from '../../api/advisor';

vi.mock('../../api/advisor');

const baseProfile = {
  risk_tolerance: 'moderate', horizon_years: 10, goals: [], constraints: {},
  investable_cash: 1000, base_currency: 'USD',
  target: { weights: { equity: 1 }, max_position_pct: 0.15, max_sector_pct: 0.3, locked: true },
};

describe('PlanPage', () => {
  beforeEach(() => vi.resetAllMocks());

  it('renders the profile form and generates a plan', async () => {
    (advisorApi.getProfile as any).mockResolvedValue(baseProfile);
    (advisorApi.generatePlan as any).mockResolvedValue({
      mode: 'deploy', base: 1000, target: { equity: 1 }, rationale: 'Deploying your cash.',
      generated_at: '2026-09-18T10:00:00', markdown: '## Plan', engine: 'deterministic', actions: [
        { side: 'buy', asset_class: 'equity', symbol: 'INGR', amount: 150, reason: 'fills equity target' },
      ],
    });
    render(<PlanPage />);
    await waitFor(() => expect(screen.getByText(/Investment Plan/i)).toBeInTheDocument());
    fireEvent.click(screen.getByRole('button', { name: /generate plan/i }));
    await waitFor(() => expect(screen.getByText(/INGR/)).toBeInTheDocument());
  });

  it('passes the AI toggle to generatePlan and shows the engine', async () => {
    (advisorApi.getProfile as any).mockResolvedValue(baseProfile);
    (advisorApi.generatePlan as any).mockResolvedValue({
      mode: 'deploy', base: 1000, target: { equity: 1 }, rationale: 'r', engine: 'ai',
      generated_at: 't', markdown: '#', actions: [],
    });
    render(<PlanPage />);
    await waitFor(() => expect(screen.getByText(/Investment Plan/i)).toBeInTheDocument());
    fireEvent.click(screen.getByRole('button', { name: /generate plan/i }));
    await waitFor(() => expect(advisorApi.generatePlan).toHaveBeenCalledWith(true));
    await waitFor(() => expect(screen.getByText(/AI-refined/i)).toBeInTheDocument());
  });

  it('prompts to create a profile when none exists', async () => {
    (advisorApi.getProfile as any).mockResolvedValue(null);
    render(<PlanPage />);
    await waitFor(() => expect(screen.getByText(/create your profile/i)).toBeInTheDocument());
  });
});
