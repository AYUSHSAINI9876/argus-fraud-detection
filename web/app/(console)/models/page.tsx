import { AlertTriangle, GitCompare } from "lucide-react";
import { stackServerApp } from "@/stack";
import { api, ApiError } from "@/lib/api";
import { EmptyState, Panel } from "@/components/ui";
import { PRCurveChart } from "@/components/charts";

export const metadata = { title: "Model health" };
export const dynamic = "force-dynamic";

interface Report {
  model_name: string;
  pr_auc: number;
  roc_auc: number;
  brier: number;
  ece: number;
  precision_at_capacity: number;
  recall_at_capacity: number;
  savings_per_1k_txn: number;
  recall_by_typology: Record<string, number>;
  reliability: { mean_predicted: number; observed_rate: number; count: number }[];
}

export default async function ModelsPage() {
  const user = await stackServerApp.getUser();
  const { accessToken } = (await user?.getAuthJson()) ?? { accessToken: null };

  let data: any = null;
  let error: string | null = null;
  try {
    data = await api.modelMetrics(accessToken);
  } catch (e) {
    error = e instanceof ApiError ? `${e.message} (${e.status})` : String(e);
  }

  if (error || !data?.evaluation) {
    return (
      <div className="p-6">
        <Panel>
          <EmptyState
            icon={AlertTriangle}
            title="No evaluation artefacts found"
            detail={error ?? "Run `python -m argus_ml.train` to generate them."}
          />
        </Panel>
      </div>
    );
  }

  const reports: Report[] = data.evaluation;
  // The baseline is rendered from `reports` directly in the comparison table —
  // it needs no separate binding here. Only the champion is pulled out, for
  // the per-typology and calibration panels below.
  const champion = reports.find((r) => r.model_name === "xgboost_risk");
  const importance: Record<string, number> = data.global_importance ?? {};
  const topFeatures = Object.entries(importance).slice(0, 12);

  return (
    <div className="p-6 space-y-5 animate-slide-up max-w-6xl">
      <header>
        <h1 className="text-xl font-semibold tracking-tight">Model health</h1>
        <p className="text-sm text-ink-lo mt-0.5">
          Offline evaluation on the held-out test window ·{" "}
          <span className="font-mono">{data.champion?.name}</span> v
          {data.champion?.version}
        </p>
      </header>

      {/* ── Comparison table ─────────────────────────────────────────── */}
      <Panel
        title="Model comparison"
        action={
          <span className="chip bg-surface-3 text-ink-lo">
            <GitCompare size={10} /> baseline first
          </span>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-surface-3 text-2xs uppercase tracking-wider text-ink-lo">
                <th className="text-left font-medium px-4 py-2.5">Model</th>
                <th className="text-right font-medium px-4 py-2.5">PR-AUC</th>
                <th className="text-right font-medium px-4 py-2.5">ROC-AUC</th>
                <th className="text-right font-medium px-4 py-2.5">Brier</th>
                <th className="text-right font-medium px-4 py-2.5">ECE</th>
                <th className="text-right font-medium px-4 py-2.5">P@cap</th>
                <th className="text-right font-medium px-4 py-2.5">R@cap</th>
                <th className="text-right font-medium px-4 py-2.5">$/1k txn</th>
              </tr>
            </thead>
            <tbody>
              {reports.map((r) => (
                <tr
                  key={r.model_name}
                  className={
                    r.model_name === "xgboost_risk"
                      ? "border-b border-surface-2 bg-accent/5"
                      : "border-b border-surface-2 last:border-0"
                  }
                >
                  <td className="px-4 py-2.5 font-mono text-xs">
                    {r.model_name}
                    {r.model_name === "xgboost_risk" && (
                      <span className="ml-2 chip bg-accent/15 text-accent">champion</span>
                    )}
                  </td>
                  <td className="px-4 py-2.5 text-right tnum font-medium">
                    {r.pr_auc.toFixed(4)}
                  </td>
                  <td className="px-4 py-2.5 text-right tnum text-ink-lo">
                    {r.roc_auc.toFixed(4)}
                  </td>
                  <td className="px-4 py-2.5 text-right tnum text-ink-mid">
                    {r.brier.toFixed(5)}
                  </td>
                  <td className="px-4 py-2.5 text-right tnum text-ink-mid">
                    {r.ece.toFixed(4)}
                  </td>
                  <td className="px-4 py-2.5 text-right tnum">
                    {r.precision_at_capacity.toFixed(3)}
                  </td>
                  <td className="px-4 py-2.5 text-right tnum">
                    {r.recall_at_capacity.toFixed(3)}
                  </td>
                  <td className="px-4 py-2.5 text-right tnum">
                    {r.savings_per_1k_txn.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="px-4 py-3 text-2xs text-ink-lo border-t border-surface-3 leading-relaxed">
          PR-AUC is the headline metric. ROC-AUC is shown only for comparability
          with published baselines — at this class balance it flatters every
          model and should not drive decisions.
        </p>
      </Panel>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* ── PR curve ───────────────────────────────────────────────── */}
        <Panel title="Precision–recall">
          <div className="p-4">
            {data.pr_curve ? (
              <PRCurveChart curves={data.pr_curve} />
            ) : (
              <p className="text-xs text-ink-lo">No curve data.</p>
            )}
          </div>
        </Panel>

        {/* ── Per-typology recall ────────────────────────────────────── */}
        <Panel title="Recall by attack type">
          <div className="p-4 space-y-2.5">
            {champion &&
              Object.entries(champion.recall_by_typology)
                .sort((a, b) => b[1] - a[1])
                .map(([typ, recall]) => (
                  <div key={typ}>
                    <div className="flex items-baseline justify-between mb-1">
                      <span className="text-xs text-ink-mid capitalize">
                        {typ.replace(/_/g, " ")}
                      </span>
                      <span className="text-xs tnum text-ink-hi font-medium">
                        {(recall * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-1.5 rounded-full bg-surface-3 overflow-hidden">
                      <div
                        className={
                          recall >= 0.6
                            ? "h-full rounded-full bg-risk-low"
                            : recall >= 0.3
                              ? "h-full rounded-full bg-risk-moderate"
                              : "h-full rounded-full bg-risk-severe"
                        }
                        style={{ width: `${Math.max(recall * 100, 1)}%` }}
                      />
                    </div>
                  </div>
                ))}
            <p className="text-2xs text-ink-lo pt-2 border-t border-surface-3 leading-relaxed">
              An aggregate recall figure can hide an attack type the model
              catches almost none of. Bust-out is expected to sit lowest — every
              individual transaction in that pattern looks legitimate.
            </p>
          </div>
        </Panel>

        {/* ── Calibration ────────────────────────────────────────────── */}
        <Panel title="Calibration">
          <div className="p-4 space-y-2">
            {champion?.reliability?.slice(0, 8).map((bin, i) => (
              <div key={i} className="flex items-center gap-3 text-xs">
                <span className="tnum text-ink-lo w-16">
                  {(bin.mean_predicted * 100).toFixed(1)}%
                </span>
                <div className="flex-1 h-1.5 rounded-full bg-surface-3 relative">
                  <div
                    className="absolute inset-y-0 rounded-full bg-accent/50"
                    style={{ width: `${Math.min(bin.mean_predicted * 100, 100)}%` }}
                  />
                  <div
                    className="absolute inset-y-0 w-0.5 bg-risk-low"
                    style={{ left: `${Math.min(bin.observed_rate * 100, 100)}%` }}
                  />
                </div>
                <span className="tnum text-ink-hi w-16 text-right">
                  {(bin.observed_rate * 100).toFixed(1)}%
                </span>
              </div>
            ))}
            <p className="text-2xs text-ink-lo pt-2 border-t border-surface-3 leading-relaxed">
              Predicted (bar) versus observed (marker). The decision policy
              computes expected cost from these scores, so a systematic gap here
              corrupts every allow/review/block decision downstream — not just
              the reported metrics.
            </p>
          </div>
        </Panel>

        {/* ── Feature importance ─────────────────────────────────────── */}
        <Panel title="Global feature importance">
          <div className="p-4 space-y-2">
            {topFeatures.map(([name, gain]) => (
              <div key={name}>
                <div className="flex items-baseline justify-between mb-1">
                  <span className="text-xs text-ink-mid font-mono truncate">{name}</span>
                  <span className="text-xs tnum text-ink-lo">
                    {(gain * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-1 rounded-full bg-surface-3 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-accent"
                    style={{ width: `${Math.max(gain * 100 * 3, 1)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}
