import { AlertTriangle, ShieldAlert } from "lucide-react";
import { stackServerApp } from "@/stack";
import { api, ApiError } from "@/lib/api";
import { EmptyState, Panel } from "@/components/ui";
import { formatCurrency } from "@/lib/risk";

export const metadata = { title: "Decision policy" };
export const dynamic = "force-dynamic";

const FIELD_HELP: Record<string, string> = {
  chargeback_fee: "Fixed fee incurred per confirmed fraud, on top of the transaction value.",
  false_positive_cost: "Estimated goodwill and support cost of declining a legitimate customer.",
  review_cost: "Analyst time per manual review, in currency units.",
  review_leakage: "Fraction of fraud that still gets through after a manual review.",
  hard_block_score: "Score above which the engine blocks regardless of amount.",
  min_block_score: "Floor below which the engine never auto-blocks, whatever the arithmetic says.",
  anomaly_review_score: "Unsupervised score that forces a review even on a relaxed supervised score.",
  max_review_rate: "Ceiling on the share of traffic that may enter the review queue.",
};

export default async function PolicyPage() {
  const user = await stackServerApp.getUser();
  const { accessToken } = (await user?.getAuthJson()) ?? { accessToken: null };

  let data: { policy: Record<string, number>; breakeven_curve: any[] } | null = null;
  let error: string | null = null;
  try {
    data = await api.policy(accessToken);
  } catch (e) {
    error = e instanceof ApiError ? `${e.message} (${e.status})` : String(e);
  }

  if (error || !data) {
    return (
      <div className="p-6">
        <Panel>
          <EmptyState icon={AlertTriangle} title="Could not load policy" detail={error ?? ""} />
        </Panel>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-5 animate-slide-up max-w-5xl">
      <header>
        <h1 className="text-xl font-semibold tracking-tight">Decision policy</h1>
        <p className="text-sm text-ink-lo mt-0.5">
          Expected-cost parameters governing allow / review / block
        </p>
      </header>

      <div className="panel p-3 flex items-start gap-2.5 border-risk-moderate/25 bg-risk-moderate/5">
        <ShieldAlert size={15} className="text-risk-moderate mt-0.5 shrink-0" />
        <p className="text-xs text-ink-mid leading-relaxed">
          These parameters move money. Every change is written to the audit log
          with your user ID, and is ADMIN-gated. The engine chooses the minimum
          expected-cost action per transaction, so a change here shifts the
          allow/review/block boundary for <em>all</em> traffic immediately.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Panel title="Current parameters">
          <dl className="p-4 space-y-3">
            {Object.entries(data.policy).map(([key, value]) => (
              <div key={key}>
                <div className="flex items-baseline justify-between gap-3">
                  <dt className="text-xs font-mono text-ink-mid">{key}</dt>
                  <dd className="tnum text-sm text-ink-hi font-medium">
                    {typeof value === "number" && value < 1 && value > 0
                      ? value.toFixed(3)
                      : value}
                  </dd>
                </div>
                {FIELD_HELP[key] && (
                  <p className="text-2xs text-ink-lo mt-0.5 leading-relaxed">
                    {FIELD_HELP[key]}
                  </p>
                )}
              </div>
            ))}
          </dl>
        </Panel>

        <Panel title="Break-even curve">
          <div className="p-4">
            <p className="text-2xs text-ink-lo mb-3 leading-relaxed">
              At each risk score, the transaction amount above which sending the
              case to review costs less than allowing it. This is the shape of
              the policy — a risk manager should be able to read it rather than
              trust a black box.
            </p>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-2xs uppercase tracking-wider text-ink-lo border-b border-surface-3">
                  <th className="text-left font-medium py-2">Risk score</th>
                  <th className="text-right font-medium py-2">Review above</th>
                </tr>
              </thead>
              <tbody>
                {data.breakeven_curve.map((row: any) => (
                  <tr key={row.risk_score} className="border-b border-surface-2 last:border-0">
                    <td className="py-2 tnum text-ink-mid">
                      {(row.risk_score * 100).toFixed(0)}%
                    </td>
                    <td className="py-2 text-right tnum text-ink-hi">
                      {formatCurrency(row.breakeven_amount)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>

      <Panel title="How a decision is made">
        <div className="p-4 space-y-3 text-xs text-ink-mid leading-relaxed">
          <p>For each transaction the engine computes three expected costs:</p>
          <pre className="bg-surface-2 rounded-md p-3 font-mono text-2xs text-ink-hi overflow-x-auto">
{`E[cost | allow]  = P(fraud) × (amount + chargeback_fee)
E[cost | block]  = (1 − P(fraud)) × false_positive_cost
E[cost | review] = review_cost + P(fraud) × review_leakage × (amount + chargeback_fee)`}
          </pre>
          <p>
            It picks the minimum, then applies two guard rails: never auto-block
            below <span className="font-mono">min_block_score</span> however
            favourable the arithmetic, and when the review queue exceeds{" "}
            <span className="font-mono">max_review_rate</span>, only reviews
            whose expected saving justifies an analyst slot survive.
          </p>
          <p className="text-ink-lo">
            This is also the seam where a reinforcement-learning policy replaces
            the analytic rule — the decision function already takes the state it
            would need.
          </p>
        </div>
      </Panel>
    </div>
  );
}
