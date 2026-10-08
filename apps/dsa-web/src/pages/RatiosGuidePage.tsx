interface Ratio {
  name: string;
  what: string;
  read: string;
  target?: string;
}

interface RatioGroup {
  heading: string;
  blurb: string;
  ratios: Ratio[];
}

const GROUPS: RatioGroup[] = [
  {
    heading: 'Valuation',
    blurb: 'How expensive the stock is relative to what the company produces.',
    ratios: [
      {
        name: 'P/E — Price / Earnings (TTM)',
        what: 'How many dollars you pay for $1 of the company’s profit over the last 12 months.',
        read: 'Lower can mean cheaper; a high P/E usually means the market expects strong growth. Only meaningful vs. peers in the same industry.',
        target: 'Roughly ≤ 20 looks fair and ≤ 15 is cheap (S&P 500 average ~15–20). Mature firms sit near 10–15; 30+ is only justified by high growth.',
      },
      {
        name: 'P/E — Forward',
        what: 'The same idea, but using analysts’ expected earnings for the year ahead instead of the past year.',
        read: 'If forward P/E is lower than trailing P/E, earnings are expected to grow.',
        target: 'Ideally lower than the trailing P/E (earnings expected to rise); same fair range (~≤ 20).',
      },
      {
        name: 'P/B — Price / Book',
        what: 'Share price compared to the company’s net asset (book) value.',
        read: 'Below ~1 can flag undervaluation (or trouble). Asset-light businesses (software) naturally run very high.',
        target: 'Under ~3 is reasonable; under 1 is deep value. Naturally high for asset-light / brand-heavy businesses.',
      },
      {
        name: 'P/S — Price / Sales',
        what: 'Share price compared to annual revenue.',
        read: 'Useful when profits are thin or negative (e.g. fast-growing companies) where P/E doesn’t work.',
        target: '1–2 is good and under 1 is exceptional — but much higher is normal for fast growers (e.g. SaaS).',
      },
      {
        name: 'Dividend Yield',
        what: 'The annual dividend as a percentage of the share price — the cash income from simply holding the stock.',
        read: 'Higher = more income now. Unusually high yields can signal a falling price or an at-risk dividend.',
        target: '~2–6% is a healthy range (S&P 500 average ~1.8%). Above ~6%, double-check the payout is sustainable.',
      },
    ],
  },
  {
    heading: 'Profitability',
    blurb: 'How efficiently the company turns revenue and capital into profit.',
    ratios: [
      {
        name: 'ROE — Return on Equity',
        what: 'Profit generated for each dollar of shareholders’ equity.',
        read: 'Higher = capital used more efficiently. Very high ROE can also come from heavy debt, so check Debt/Equity too.',
        target: 'Above 15% is strong, 12%+ is good, under 10% is weak (S&P 500 average ~14%).',
      },
      {
        name: 'Profit Margin',
        what: 'Of every $1 of revenue, how much is left as net profit after all costs, interest and tax.',
        read: 'Higher is better. Compare within an industry — grocers run thin, software runs fat.',
        target: 'Above 10% is good and 20%+ is excellent (cross-industry average ~9%).',
      },
      {
        name: 'Operating Margin',
        what: 'Profit from the core business (before interest and tax) as a percentage of revenue.',
        read: 'Shows how profitable the actual operations are, stripping out financing and tax effects.',
        target: 'Above ~12% is healthy, 15–20% is strong, and 20%+ is excellent.',
      },
    ],
  },
  {
    heading: 'Growth',
    blurb: 'How fast the business is expanding year over year.',
    ratios: [
      {
        name: 'Revenue Growth',
        what: 'How much sales grew compared with the same period a year ago.',
        read: 'Consistent revenue growth is a sign of demand and a healthy top line.',
        target: 'Positive and steady; ~10%+ is strong. For growth names, the “Rule of 40”: revenue growth + operating margin ≥ 40%.',
      },
      {
        name: 'Earnings Growth',
        what: 'How much profit (earnings per share) grew year over year.',
        read: 'Often the biggest driver of a stock’s price over time. Growing faster than revenue implies improving margins.',
        target: 'Positive and ideally faster than revenue growth (margins expanding); double-digit growth is strong.',
      },
    ],
  },
  {
    heading: 'Intrinsic Value',
    blurb:
      'An estimate of what a share is actually worth based on the business, so you can compare it to the market price. The app combines two models and shows the margin of safety.',
    ratios: [
      {
        name: 'DCF — Discounted Cash Flow',
        what: 'Projects the company’s free cash flow years into the future, then discounts it back to today’s dollars to get a per-share fair value.',
        read: 'The headline estimate. If the current price is well below the DCF value you may be getting a discount; well above it suggests the market is pricing in a lot of optimism. Very sensitive to the growth and discount-rate assumptions.',
        target: 'We use conservative assumptions: growth capped at ~10%, a ~10% discount rate, ~2.5% terminal growth over a 10-year projection. Needs positive free cash flow, so it’s skipped for many banks and unprofitable firms.',
      },
      {
        name: 'Graham Number & revised formula',
        what: 'Benjamin Graham’s quick sanity checks. The Graham Number is √(22.5 × EPS × book value per share); the revised formula is EPS × (8.5 + 2 × growth%).',
        read: 'A transparent cross-check on the DCF. When the two broadly agree, you can trust the range more; when they diverge sharply, treat the fair value as a wide band rather than a precise figure.',
        target: 'Used as a conservative floor and cross-check — not the headline. Requires positive earnings and book value.',
      },
      {
        name: 'Margin of Safety',
        what: 'How far the current price sits below the estimated fair value — the cushion that protects you if the estimate is too optimistic.',
        read: 'A positive margin means the stock trades below its estimated worth (undervalued); a negative one means it trades above (overvalued). Value investors like a meaningful cushion before buying.',
        target: 'Undervalued when price is ≥10% below fair value, overvalued when ≥10% above, otherwise fairly valued. A bigger positive margin is safer — but it’s only as good as the assumptions behind the estimate.',
      },
    ],
  },
  {
    heading: 'Size & Health',
    blurb: 'How large the company is and how much leverage it carries.',
    ratios: [
      {
        name: 'Market Cap',
        what: 'The total value of all shares (share price × number of shares).',
        read: 'Signals company size — small caps are more volatile, large caps more stable.',
        target: 'Not “good/bad” — a size/risk tier: large-cap >$10B is steadier, mid $2–10B, small <$2B more volatile.',
      },
      {
        name: 'EPS — Earnings Per Share',
        what: 'Net profit divided by the number of shares — the profit attributable to each share.',
        read: 'This is the “E” in P/E. Rising EPS over time is what you want to see.',
        target: 'No single number — it should be positive and growing year over year.',
      },
      {
        name: 'Debt / Equity',
        what: 'How much debt the company uses relative to shareholders’ equity.',
        read: 'Higher = more leverage and risk (but also potentially higher returns). Varies a lot by industry.',
        target: 'Under ~1 is conservative (shown here as roughly under 100), 1–2 is moderate, over 2 is high — varies by industry.',
      },
    ],
  },
  {
    heading: 'Support & Resistance',
    blurb:
      'Price levels where a stock has tended to stop falling (support) or stop rising (resistance). The app uses classic pivot points — a standard, formula-based way to project them from recent price action.',
    ratios: [
      {
        name: 'Pivot (P)',
        what: 'The central reference level, the average of the last session’s high, low and close: P = (High + Low + Close) / 3.',
        read: 'Trading above the pivot is a mildly bullish bias for the session; below it, mildly bearish. The other levels are measured out from here.',
      },
      {
        name: 'Resistance — R1, R2, R3',
        what: 'Projected ceilings above the current price, derived from the pivot and the recent range (R1 is nearest, R3 furthest).',
        read: 'Prices often pause or pull back at these levels. A decisive break above one tends to put the next level in play as the new target.',
      },
      {
        name: 'Support — S1, S2, S3',
        what: 'Projected floors below the current price (S1 nearest, S3 furthest).',
        read: 'Prices often find a footing at these levels. They’re common spots to consider stop-losses just beneath, or to watch for a bounce. A clean break below one opens the way to the next.',
      },
      {
        name: 'Nearest support / resistance',
        what: 'The closest level below the price (support) and the closest above it (resistance) — the two levels that matter most right now.',
        read: 'The gap up to resistance versus down to support gives a quick read on the near-term risk/reward. Pivots are short-term and recompute each session, so treat them as a guide, not a guarantee — and confirm with trend and volume.',
      },
    ],
  },
];

export default function RatiosGuidePage() {
  return (
    <div className="mx-auto max-w-3xl space-y-6 p-6">
      <header>
        <h1 className="text-2xl font-semibold">Metrics — Explained</h1>
        <p className="mt-1 text-secondary-text">
          A plain-English guide to the metrics shown in a stock analysis (Home &rarr; Analyze, and Ask):
          the fundamental ratios, the intrinsic-value estimate, and the support &amp; resistance levels.
        </p>
        <p className="mt-2 rounded-md border border-[hsl(var(--border))] bg-[hsl(var(--muted))] p-3 text-sm text-secondary-text">
          <span className="label-uppercase mr-1 text-xs">Important:</span>
          the &ldquo;desirable&rdquo; values below are general rules of thumb. What counts as healthy
          depends heavily on the <strong>industry</strong> and the company&rsquo;s <strong>growth stage</strong>,
          so always compare a stock to its own sector peers rather than to these numbers in isolation.
        </p>
      </header>

      {GROUPS.map((group) => (
        <section key={group.heading} className="space-y-3">
          <div>
            <h2 className="text-lg font-medium">{group.heading}</h2>
            <p className="text-sm text-secondary-text">{group.blurb}</p>
          </div>
          <div className="space-y-3">
            {group.ratios.map((r) => (
              <div key={r.name} className="home-subpanel p-4">
                <div className="font-semibold">{r.name}</div>
                <p className="mt-1 text-sm">{r.what}</p>
                <p className="mt-1 text-sm text-secondary-text">
                  <span className="label-uppercase mr-1 text-xs">How to read it:</span>
                  {r.read}
                </p>
                {r.target && (
                  <p className="mt-2 text-sm font-medium text-[hsl(var(--primary))]">
                    <span className="label-uppercase mr-1 text-xs">Desirable:</span>
                    {r.target}
                  </p>
                )}
              </div>
            ))}
          </div>
        </section>
      ))}

      <p className="pt-2 text-xs text-muted-text">
        Educational reference only &mdash; not investment advice. Figures come from live market data and
        can vary by source and reporting period.
      </p>
    </div>
  );
}
