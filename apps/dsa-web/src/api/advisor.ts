import apiClient from './index';

export interface AdvisorTarget {
  weights: Record<string, number>;
  max_position_pct: number;
  max_sector_pct: number;
  locked: boolean;
}

export interface AdvisorProfile {
  owner_id?: string;
  risk_tolerance: string;
  horizon_years: number;
  goals: string[];
  constraints: Record<string, boolean>;
  investable_cash: number;
  base_currency: string;
  target: AdvisorTarget;
}

export interface PlanAction {
  side: string;
  asset_class: string;
  symbol: string;
  amount: number;
  reason: string;
}

export interface AdvisorPlan {
  mode: string;
  base: number;
  target: Record<string, number>;
  actions: PlanAction[];
  rationale: string;
  markdown: string;
  generated_at: string;
}

export const advisorApi = {
  async getProfile(): Promise<AdvisorProfile | null> {
    try {
      const res = await apiClient.get<AdvisorProfile>('/api/v1/advisor/profile');
      return res.data;
    } catch {
      return null; // 404 => no profile yet
    }
  },
  async saveProfile(profile: AdvisorProfile): Promise<AdvisorProfile> {
    const res = await apiClient.put<AdvisorProfile>('/api/v1/advisor/profile', profile);
    return res.data;
  },
  async suggest(): Promise<AdvisorTarget> {
    const res = await apiClient.post<{ target: AdvisorTarget }>('/api/v1/advisor/suggest');
    return res.data.target;
  },
  async generatePlan(): Promise<AdvisorPlan> {
    const res = await apiClient.post<AdvisorPlan>('/api/v1/advisor/plan');
    return res.data;
  },
};
