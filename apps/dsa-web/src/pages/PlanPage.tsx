import { useEffect, useState } from 'react';
import { advisorApi } from '../api/advisor';
import type { AdvisorProfile, AdvisorPlan } from '../api/advisor';

const EMPTY: AdvisorProfile = {
  risk_tolerance: 'moderate',
  horizon_years: 10,
  goals: [],
  constraints: {},
  investable_cash: 0,
  base_currency: 'USD',
  target: {
    weights: { equity: 0.7, fixed_income: 0.2, cash: 0.1 },
    max_position_pct: 0.15,
    max_sector_pct: 0.3,
    locked: false,
  },
};

const actionTag: Record<string, string> = { buy: '🟢 Buy', trim: '🔻 Trim', unallocated: '⚪ Unallocated' };

export default function PlanPage() {
  const [profile, setProfile] = useState<AdvisorProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [plan, setPlan] = useState<AdvisorPlan | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useAI, setUseAI] = useState(true);

  useEffect(() => {
    advisorApi.getProfile().then((p) => {
      setProfile(p);
      setLoading(false);
    });
  }, []);

  const save = async () => {
    if (!profile) return;
    setBusy(true);
    setError(null);
    try {
      setProfile(await advisorApi.saveProfile(profile));
    } catch {
      setError('Could not save your profile. Check the values and try again.');
    } finally {
      setBusy(false);
    }
  };

  const suggest = async () => {
    if (!profile) return;
    const target = await advisorApi.suggest();
    setProfile({ ...profile, target });
  };

  const generate = async () => {
    setBusy(true);
    setError(null);
    try {
      setPlan(await advisorApi.generatePlan(useAI));
    } catch {
      setError('Lock your target allocation and save first, then generate.');
    } finally {
      setBusy(false);
    }
  };

  if (loading) return <div className="mx-auto max-w-3xl p-6">Loading…</div>;

  if (!profile) {
    return (
      <div className="mx-auto max-w-3xl space-y-4 p-6">
        <h1 className="text-2xl font-semibold">Investment Plan</h1>
        <p className="text-[hsl(var(--muted-foreground))]">
          Set up your profile to generate a whole-portfolio plan.
        </p>
        <button className="rounded-lg bg-primary-gradient px-4 py-2 text-white"
                onClick={() => setProfile({ ...EMPTY })}>
          Create your profile
        </button>
      </div>
    );
  }

  const locked = profile.target.locked;

  return (
    <div className="mx-auto max-w-3xl space-y-8 p-6">
      <h1 className="text-2xl font-semibold">Your Investment Plan</h1>

      <section className="space-y-3">
        <h2 className="text-lg font-medium">My profile</h2>
        <label className="flex items-center justify-between gap-4">
          <span>Risk tolerance</span>
          <select value={profile.risk_tolerance}
                  onChange={(e) => setProfile({ ...profile, risk_tolerance: e.target.value })}>
            <option value="conservative">Conservative</option>
            <option value="moderate">Moderate</option>
            <option value="aggressive">Aggressive</option>
          </select>
        </label>
        <label className="flex items-center justify-between gap-4">
          <span>Horizon (years)</span>
          <input type="number" value={profile.horizon_years}
                 onChange={(e) => setProfile({ ...profile, horizon_years: Number(e.target.value) })} />
        </label>
        <label className="flex items-center justify-between gap-4">
          <span>Investable cash</span>
          <input type="number" value={profile.investable_cash}
                 onChange={(e) => setProfile({ ...profile, investable_cash: Number(e.target.value) })} />
        </label>
        <div className="flex gap-2">
          <button onClick={suggest}>Suggest target</button>
          <button onClick={save} disabled={busy}>Save profile</button>
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-medium">Target allocation {locked ? '🔒' : '(unlocked)'}</h2>
        <ul className="list-disc pl-6">
          {Object.entries(profile.target.weights).map(([k, v]) => (
            <li key={k}>{k.replace('_', ' ')}: {(v * 100).toFixed(0)}%</li>
          ))}
        </ul>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={locked}
                 onChange={(e) => setProfile({ ...profile, target: { ...profile.target, locked: e.target.checked } })} />
          Lock target
        </label>
        <button onClick={save} disabled={busy}>Save</button>
      </section>

      <div className="flex items-center gap-3">
        <button className="rounded-lg bg-primary-gradient px-4 py-2 text-white disabled:opacity-50"
                onClick={generate} disabled={busy || !locked}>
          Generate plan
        </button>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={useAI} onChange={(e) => setUseAI(e.target.checked)} />
          Refine with AI
        </label>
      </div>
      {error && <p role="alert" className="text-[hsl(var(--destructive))]">{error}</p>}

      {plan && (
        <section className="space-y-3">
          <h2 className="text-lg font-medium">
            Plan · {plan.mode} mode · base ${plan.base.toLocaleString()} ·{' '}
            {plan.engine === 'ai' ? 'AI-refined' : 'Deterministic'}
          </h2>
          <p>{plan.rationale}</p>
          <ul className="space-y-1">
            {plan.actions.map((a, i) => (
              <li key={i}>
                <strong>{actionTag[a.side] ?? a.side} {a.symbol || a.asset_class.replace('_', ' ')}</strong>
                {' '}— ${a.amount.toLocaleString()} · {a.reason}
              </li>
            ))}
          </ul>
          <p className="text-xs text-[hsl(var(--muted-foreground))]">
            Not financial advice. Educational only; does not account for taxes. You decide and act.
          </p>
        </section>
      )}
    </div>
  );
}
