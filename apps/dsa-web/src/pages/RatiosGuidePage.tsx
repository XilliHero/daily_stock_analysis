interface Ratio {
  name: string;
  what: string;
  read: string;
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
      },
      {
        name: 'P/E — Forward',
        what: 'The same idea, but using analysts’ expected earnings for the year ahead instead of the past year.',
        read: 'If forward P/E is lower than trailing P/E, earnings are expected to grow.',
      },
      {
        name: 'P/B — Price / Book',
        what: 'Share price compared to the company’s net asset (book) value.',
        read: 'Below ~1 can flag undervaluation (or trouble). Asset-light businesses (software) naturally run very high.',
      },
      {
        name: 'P/S — Price / Sales',
        what: 'Share price compared to annual revenue.',
        read: 'Useful when profits are thin or negative (e.g. fast-growing companies) where P/E doesn’t work.',
      },
      {
        name: 'Dividend Yield',
        what: 'The annual dividend as a percentage of the share price — the cash income from simply holding the stock.',
        read: 'Higher = more income now. Unusually high yields can signal a falling price or an at-risk dividend.',
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
      },
      {
        name: 'Profit Margin',
        what: 'Of every $1 of revenue, how much is left as net profit after all costs, interest and tax.',
        read: 'Higher is better. Compare within an industry — grocers run thin, software runs fat.',
      },
      {
        name: 'Operating Margin',
        what: 'Profit from the core business (before interest and tax) as a percentage of revenue.',
        read: 'Shows how profitable the actual operations are, stripping out financing and tax effects.',
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
      },
      {
        name: 'Earnings Growth',
        what: 'How much profit (earnings per share) grew year over year.',
        read: 'Often the biggest driver of a stock’s price over time. Growing faster than revenue implies improving margins.',
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
      },
      {
        name: 'EPS — Earnings Per Share',
        what: 'Net profit divided by the number of shares — the profit attributable to each share.',
        read: 'This is the “E” in P/E. Rising EPS over time is what you want to see.',
      },
      {
        name: 'Debt / Equity',
        what: 'How much debt the company uses relative to shareholders’ equity.',
        read: 'Higher = more leverage and risk (but also potentially higher returns). Varies a lot by industry.',
      },
    ],
  },
];

export default function RatiosGuidePage() {
  return (
    <div className="mx-auto max-w-3xl space-y-6 p-6">
      <header>
        <h1 className="text-2xl font-semibold">Fundamental Ratios — Explained</h1>
        <p className="mt-1 text-secondary-text">
          A plain-English guide to the ratios shown in the Fundamentals section of a stock analysis
          (Home &rarr; Analyze, and Ask). Ratios are only comparable within the same industry.
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
