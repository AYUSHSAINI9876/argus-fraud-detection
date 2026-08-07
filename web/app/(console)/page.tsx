import {
  AlertTriangle,
  Ban,
  DollarSign,
  Eye,
  Gauge,
  Timer,
  TrendingUp,
} from "lucide-react";
import { stackServerApp } from "@/stack";
import { api, ApiError, type Overview, type TimeseriesPoint } from "@/lib/api";
import { EmptyState, Panel, StatTile } from "@/components/ui";
import { DecisionVolumeChart } from "@/components/charts";
import { formatCompact, formatCurrency } from "@/lib/risk";

export const metadata = { title: "Overview" };
export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  const user = await stackServerApp.getUser();
  const { accessToken } = (await user?.getAuthJson()) ?? { accessToken: null };

  let overview: Overview | null = null;
  let series: TimeseriesPoint[] = [];
  let error: string | null = null;

  try {
    const [o, t] = await Promise.all([
      api.overview(accessToken, 24),
      api.timeseries(accessToken, 48),
    ]);
    overview = o;
    series = t.points;
  } catch (e) {
    error = e instanceof ApiError ? `${e.message} (${e.status})` : String(e);
  }

  if (error || !overview) {
    return (
      <div className="p-6">
        <Panel>
          <EmptyState
            icon={AlertTriangle}
            title="Could not reach the risk engine"
            detail={error ?? "No data returned."}
          />
        </Panel>
      </div>
    );
  }

  const hasTraffic = overview.transactions_scored > 0;

  return (
    <div className="p-6 space-y-5 animate-slide-up">
      <header className="flex items-end justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Overview</h1>
          <p className="text-sm text-ink-lo mt-0.5">
            Live decisioning across the last {overview.window_hours} hours
          </p>
        </div>
      </header>

      {!hasTraffic ? (
        <Panel>
          <EmptyState
            icon={Gauge}
            title="No transactions scored yet"
            detail="Start the replay stream to feed live traffic into the engine."
          />
        </Panel>
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <StatTile
              label="Scored"
              value={formatCompact(overview.transactions_scored)}
              sub={`${formatCurrency(overview.total_volume)} volume`}
              icon={TrendingUp}
            />
            <StatTile
              label="Blocked"
              value={overview.block_count.toLocaleString()}
              sub={`${(overview.block_rate * 100).toFixed(2)}% of volume`}
              icon={Ban}
              tone="danger"
            />
            <StatTile
              label="In review"
              value={overview.review_count.toLocaleString()}
              sub={`${(overview.review_rate * 100).toFixed(2)}% of volume`}
              icon={Eye}
              tone="warn"
            />
            <StatTile
              label="Value protected"
              value={formatCurrency(overview.blocked_value)}
              sub="Gross, before false-positive cost"
              icon={DollarSign}
              tone="good"
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
            <Panel title="Decision volume" className="lg:col-span-2">
              <div className="p-4">
                <DecisionVolumeChart points={series} />
              </div>
            </Panel>

            <div className="space-y-3">
              <StatTile
                label="Scoring latency"
                value={`${overview.latency_ms.p99.toFixed(0)} ms`}
                sub={`p95 ${overview.latency_ms.p95.toFixed(0)}ms · avg ${overview.latency_ms.avg.toFixed(1)}ms`}
                icon={Timer}
                tone={overview.latency_ms.p99 > 150 ? "warn" : "good"}
              />
              <StatTile
                label="Realised precision"
                value={
                  overview.realised_precision === null
                    ? "—"
                    : `${(overview.realised_precision * 100).toFixed(1)}%`
                }
                sub={
                  overview.realised_precision === null
                    ? "Awaiting analyst dispositions"
                    : `From ${overview.cases_resolved} resolved cases`
                }
                icon={Gauge}
              />
              <StatTile
                label="Mean risk score"
                value={(overview.avg_risk_score * 100).toFixed(2) + "%"}
                sub="Population average"
                icon={Gauge}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
