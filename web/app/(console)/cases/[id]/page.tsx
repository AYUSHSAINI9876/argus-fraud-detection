import Link from "next/link";
import { notFound } from "next/navigation";
import { format, parseISO } from "date-fns";
import { ArrowLeft, Bot, MapPin, Sparkles } from "lucide-react";
import { stackServerApp } from "@/stack";
import { api, ApiError, type CaseDetail } from "@/lib/api";
import { DecisionBadge, Panel, RiskBadge } from "@/components/ui";
import { AttributionBar } from "@/components/attribution";
import { CaseActions } from "@/components/case-actions";
import { featureLabel, featureValue, formatCurrency } from "@/lib/risk";

export const dynamic = "force-dynamic";

export default async function CasePage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const user = await stackServerApp.getUser();
  const { accessToken } = (await user?.getAuthJson()) ?? { accessToken: null };

  let detail: CaseDetail;
  try {
    detail = await api.caseDetail(accessToken, id);
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) notFound();
    throw e;
  }

  const role =
    ((user?.serverMetadata as Record<string, unknown> | null)?.role as string) ??
    "VIEWER";

  const increases = detail.attributions
    .filter((a) => a.direction === "increases_risk")
    .sort((a, b) => b.contribution - a.contribution);
  const decreases = detail.attributions
    .filter((a) => a.direction === "decreases_risk")
    .sort((a, b) => a.contribution - b.contribution);

  return (
    <div className="p-6 space-y-4 animate-slide-up max-w-6xl">
      <Link
        href="/queue"
        className="inline-flex items-center gap-1.5 text-xs text-ink-lo hover:text-ink-mid"
      >
        <ArrowLeft size={13} /> Back to queue
      </Link>

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between gap-6">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-xl font-semibold tracking-tight tnum">
              {formatCurrency(detail.amount)}
            </h1>
            <RiskBadge score={detail.risk_score} />
            <DecisionBadge decision={detail.decision} />
            {detail.triggered_rule && (
              <span className="chip bg-surface-3 text-ink-mid border border-surface-4">
                {detail.triggered_rule.replace(/_/g, " ")}
              </span>
            )}
          </div>
          <p className="text-xs text-ink-lo mt-1 font-mono">
            {detail.transaction_id} · {detail.customer_id} · {detail.merchant_id}
          </p>
        </div>

        <CaseActions
          caseId={detail.id}
          status={detail.status}
          decision={detail.decision}
          assignedTo={detail.assigned_to}
          currentUserId={user?.id ?? null}
          role={role}
          hasCopilotSummary={Boolean(detail.copilot_summary)}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 space-y-4">
          {/* ── Why this was flagged ─────────────────────────────────── */}
          <Panel title="Why this scored the way it did">
            <div className="p-4 space-y-4">
              <p className="text-xs text-ink-mid leading-relaxed bg-surface-2 rounded-md p-3 border border-surface-3">
                {detail.rationale}
              </p>

              <div>
                <div className="field-label mb-2">Drivers of risk</div>
                <div className="space-y-1.5">
                  {increases.length === 0 && (
                    <p className="text-xs text-ink-lo">No positive contributors recorded.</p>
                  )}
                  {increases.map((a) => (
                    <AttributionBar
                      key={a.feature}
                      label={featureLabel(a.feature)}
                      value={featureValue(a.feature, a.value)}
                      contribution={a.contribution}
                      max={Math.max(...detail.attributions.map((x) => Math.abs(x.contribution)))}
                    />
                  ))}
                </div>
              </div>

              {decreases.length > 0 && (
                <div>
                  <div className="field-label mb-2">Mitigating factors</div>
                  <div className="space-y-1.5">
                    {decreases.map((a) => (
                      <AttributionBar
                        key={a.feature}
                        label={featureLabel(a.feature)}
                        value={featureValue(a.feature, a.value)}
                        contribution={a.contribution}
                        max={Math.max(...detail.attributions.map((x) => Math.abs(x.contribution)))}
                      />
                    ))}
                  </div>
                </div>
              )}

              <p className="text-2xs text-ink-lo leading-relaxed border-t border-surface-3 pt-3">
                Attributions are exact SHAP values from{" "}
                <span className="font-mono">{detail.model_version}</span>. They
                describe this model&apos;s reasoning — they are not a claim about
                real-world causation.
              </p>
            </div>
          </Panel>

          {/* ── Copilot ───────────────────────────────────────────────── */}
          {detail.copilot_summary && (
            <Panel
              title="Analyst copilot"
              action={
                <span className="chip bg-accent/10 text-accent border border-accent/25">
                  <Sparkles size={10} /> AI draft
                </span>
              }
            >
              <div className="p-4">
                <p className="text-sm text-ink-mid leading-relaxed whitespace-pre-wrap">
                  {detail.copilot_summary}
                </p>
                <p className="text-2xs text-ink-lo mt-3 flex items-center gap-1.5 border-t border-surface-3 pt-3">
                  <Bot size={11} />
                  Generated from the SHAP attributions above and similar historical
                  cases. Verify before relying on it — accept/reject feeds the
                  copilot&apos;s evaluation set.
                </p>
              </div>
            </Panel>
          )}

          {/* ── Notes ─────────────────────────────────────────────────── */}
          <Panel title={`Notes (${detail.notes.length})`}>
            <div className="p-4 space-y-3">
              {detail.notes.length === 0 && (
                <p className="text-xs text-ink-lo">No notes on this case yet.</p>
              )}
              {detail.notes.map((n) => (
                <div key={n.id} className="text-sm">
                  <div className="flex items-baseline gap-2">
                    <span className="font-medium text-ink-hi text-xs">
                      {n.author_name ?? n.author_id}
                    </span>
                    <span className="text-2xs text-ink-lo tnum">
                      {format(parseISO(n.created_at), "d MMM HH:mm")}
                    </span>
                  </div>
                  <p className="text-ink-mid mt-0.5 leading-relaxed">{n.body}</p>
                </div>
              ))}
            </div>
          </Panel>
        </div>

        {/* ── Side rail ─────────────────────────────────────────────────── */}
        <div className="space-y-4">
          <Panel title="Signals">
            <dl className="p-4 space-y-2.5 text-sm">
              <Row label="Model score" value={`${(detail.risk_score * 100).toFixed(2)}%`} />
              {detail.anomaly_score !== null && (
                <Row
                  label="Anomaly score"
                  value={`${(detail.anomaly_score * 100).toFixed(1)}%`}
                  hint={
                    detail.anomaly_score > 0.95
                      ? "Pattern is unlike normal traffic — may be a novel attack type"
                      : undefined
                  }
                />
              )}
              <Row label="Expected loss" value={formatCurrency(detail.expected_loss)} />
              <Row label="Model" value={detail.model_version} mono />
              <Row
                label="Scored"
                value={format(parseISO(detail.created_at), "d MMM yyyy HH:mm")}
              />
            </dl>
          </Panel>

          <Panel title="Behavioural context">
            <dl className="p-4 space-y-2.5 text-sm">
              {[
                "txn_count_1h",
                "txn_count_24h",
                "amount_zscore",
                "distance_from_home_km",
                "implied_velocity_kmh",
                "is_new_device",
                "is_new_country",
                "known_device_count",
              ]
                .filter((k) => k in detail.features)
                .map((k) => (
                  <Row
                    key={k}
                    label={featureLabel(k)}
                    value={featureValue(k, detail.features[k])}
                  />
                ))}
            </dl>
          </Panel>

          <Panel title="Location">
            <div className="p-4 flex items-start gap-2 text-sm">
              <MapPin size={14} className="text-ink-lo mt-0.5 shrink-0" />
              <div className="text-ink-mid">
                <div className="tnum">
                  {featureValue(
                    "distance_from_home_km",
                    detail.features.distance_from_home_km ?? -1,
                  )}{" "}
                  from home
                </div>
                {detail.features.is_foreign_country >= 0.5 && (
                  <div className="text-risk-elevated text-xs mt-0.5">
                    Outside home country
                  </div>
                )}
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

function Row({
  label,
  value,
  mono,
  hint,
}: {
  label: string;
  value: string;
  mono?: boolean;
  hint?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-xs text-ink-lo shrink-0">{label}</dt>
      <dd className="text-right">
        <span className={mono ? "font-mono text-2xs text-ink-hi" : "tnum text-ink-hi"}>
          {value}
        </span>
        {hint && <div className="text-2xs text-risk-elevated mt-0.5 max-w-[180px]">{hint}</div>}
      </dd>
    </div>
  );
}
