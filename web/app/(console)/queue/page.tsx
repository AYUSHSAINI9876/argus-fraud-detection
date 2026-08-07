import Link from "next/link";
import { formatDistanceToNow, parseISO } from "date-fns";
import { AlertTriangle, Inbox, ArrowUpDown } from "lucide-react";
import { stackServerApp } from "@/stack";
import { api, ApiError, type CaseSummary } from "@/lib/api";
import { DecisionBadge, EmptyState, Panel, RiskBadge } from "@/components/ui";
import { formatCurrency } from "@/lib/risk";

export const metadata = { title: "Case queue" };
export const dynamic = "force-dynamic";

/**
 * The queue is sorted by EXPECTED LOSS (risk x amount), not by risk score.
 *
 * This is the single most consequential design decision in the console. An
 * analyst hour spent on a 0.92-risk $14 transaction is an hour not spent on a
 * 0.61-risk $8,400 one — and the second is worth roughly four hundred times
 * more to the business. Sorting a fraud queue by score is intuitive and
 * wrong.
 */
export default async function QueuePage({
  searchParams,
}: {
  searchParams: Promise<{ sort?: string; status?: string }>;
}) {
  const params = await searchParams;
  const sort = params.sort ?? "expected_loss";
  const status = params.status ?? "open";

  const user = await stackServerApp.getUser();
  const { accessToken } = (await user?.getAuthJson()) ?? { accessToken: null };

  let cases: CaseSummary[] = [];
  let error: string | null = null;

  try {
    cases = await api.cases(accessToken, { sort, status, limit: 100 });
  } catch (e) {
    error = e instanceof ApiError ? `${e.message} (${e.status})` : String(e);
  }

  const totalExposure = cases.reduce((s, c) => s + c.expected_loss, 0);

  return (
    <div className="p-6 space-y-5 animate-slide-up">
      <header className="flex items-end justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Case queue</h1>
          <p className="text-sm text-ink-lo mt-0.5">
            {cases.length} open ·{" "}
            <span className="tnum">{formatCurrency(totalExposure)}</span> expected
            exposure
          </p>
        </div>

        <div className="flex items-center gap-1.5 text-xs">
          <ArrowUpDown size={13} className="text-ink-lo" />
          {[
            { key: "expected_loss", label: "Expected loss" },
            { key: "risk_score", label: "Risk score" },
            { key: "created_at", label: "Newest" },
          ].map((opt) => (
            <Link
              key={opt.key}
              href={`/queue?sort=${opt.key}&status=${status}`}
              className={
                sort === opt.key
                  ? "rounded px-2 py-1 bg-surface-3 text-ink-hi font-medium"
                  : "rounded px-2 py-1 text-ink-lo hover:text-ink-mid hover:bg-surface-2"
              }
            >
              {opt.label}
            </Link>
          ))}
        </div>
      </header>

      {error ? (
        <Panel>
          <EmptyState icon={AlertTriangle} title="Could not load the queue" detail={error} />
        </Panel>
      ) : cases.length === 0 ? (
        <Panel>
          <EmptyState
            icon={Inbox}
            title="Queue is clear"
            detail="No cases are waiting. New alerts appear here as transactions are routed to review."
          />
        </Panel>
      ) : (
        <Panel className="overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-surface-3 text-2xs uppercase tracking-wider text-ink-lo">
                <th className="text-left font-medium px-4 py-2.5">Risk</th>
                <th className="text-left font-medium px-4 py-2.5">Decision</th>
                <th className="text-right font-medium px-4 py-2.5">Amount</th>
                <th className="text-right font-medium px-4 py-2.5">Expected loss</th>
                <th className="text-left font-medium px-4 py-2.5">Top driver</th>
                <th className="text-left font-medium px-4 py-2.5">Customer</th>
                <th className="text-left font-medium px-4 py-2.5">Age</th>
              </tr>
            </thead>
            <tbody>
              {cases.map((c) => {
                const topDriver = c.attributions
                  .filter((a) => a.direction === "increases_risk")
                  .sort((a, b) => b.contribution - a.contribution)[0];

                return (
                  <tr
                    key={c.id}
                    className="border-b border-surface-2 last:border-0 hover:bg-surface-2/60 transition-colors"
                  >
                    <td className="px-4 py-2.5">
                      <Link href={`/cases/${c.id}`} className="block">
                        <RiskBadge score={c.risk_score} />
                      </Link>
                    </td>
                    <td className="px-4 py-2.5">
                      <DecisionBadge decision={c.decision} />
                    </td>
                    <td className="px-4 py-2.5 text-right tnum text-ink-hi">
                      {formatCurrency(c.amount)}
                    </td>
                    <td className="px-4 py-2.5 text-right tnum font-medium text-risk-elevated">
                      {formatCurrency(c.expected_loss)}
                    </td>
                    <td className="px-4 py-2.5 text-ink-mid text-xs max-w-[220px] truncate">
                      {topDriver ? featureShort(topDriver.feature) : "—"}
                    </td>
                    <td className="px-4 py-2.5 font-mono text-2xs text-ink-lo">
                      {c.customer_id}
                    </td>
                    <td className="px-4 py-2.5 text-xs text-ink-lo whitespace-nowrap">
                      {formatDistanceToNow(parseISO(c.created_at), { addSuffix: false })}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Panel>
      )}
    </div>
  );
}

function featureShort(name: string): string {
  return name.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}
