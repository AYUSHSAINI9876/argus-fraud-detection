import { format, parseISO } from "date-fns";
import { AlertTriangle, ScrollText } from "lucide-react";
import { stackServerApp } from "@/stack";
import { api, ApiError } from "@/lib/api";
import { EmptyState, Panel } from "@/components/ui";

export const metadata = { title: "Audit log" };
export const dynamic = "force-dynamic";

const ACTION_STYLE: Record<string, string> = {
  "case.release_block": "bg-risk-severe/12 text-risk-severe border-risk-severe/25",
  "policy.update": "bg-risk-elevated/12 text-risk-elevated border-risk-elevated/25",
  "model.promote": "bg-risk-elevated/12 text-risk-elevated border-risk-elevated/25",
  "case.disposition": "bg-risk-low/12 text-risk-low border-risk-low/25",
  "case.escalate": "bg-risk-moderate/12 text-risk-moderate border-risk-moderate/25",
};

export default async function AuditPage() {
  const user = await stackServerApp.getUser();
  const { accessToken } = (await user?.getAuthJson()) ?? { accessToken: null };

  let rows: any[] = [];
  let error: string | null = null;
  try {
    rows = await api.auditLog(accessToken, 24 * 30);
  } catch (e) {
    error = e instanceof ApiError ? `${e.message} (${e.status})` : String(e);
  }

  return (
    <div className="p-6 space-y-5 animate-slide-up max-w-6xl">
      <header>
        <h1 className="text-xl font-semibold tracking-tight">Audit log</h1>
        <p className="text-sm text-ink-lo mt-0.5">
          Append-only record of privileged actions · last 30 days
        </p>
      </header>

      {error ? (
        <Panel>
          <EmptyState icon={AlertTriangle} title="Could not load the audit log" detail={error} />
        </Panel>
      ) : rows.length === 0 ? (
        <Panel>
          <EmptyState
            icon={ScrollText}
            title="No audited actions yet"
            detail="Case dispositions, block releases, policy changes and model promotions appear here."
          />
        </Panel>
      ) : (
        <Panel className="overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-surface-3 text-2xs uppercase tracking-wider text-ink-lo">
                <th className="text-left font-medium px-4 py-2.5">When</th>
                <th className="text-left font-medium px-4 py-2.5">Action</th>
                <th className="text-left font-medium px-4 py-2.5">Actor</th>
                <th className="text-left font-medium px-4 py-2.5">Role</th>
                <th className="text-left font-medium px-4 py-2.5">Target</th>
                <th className="text-left font-medium px-4 py-2.5">Change</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id} className="border-b border-surface-2 last:border-0">
                  <td className="px-4 py-2.5 text-xs text-ink-lo tnum whitespace-nowrap">
                    {format(parseISO(r.created_at), "d MMM HH:mm:ss")}
                  </td>
                  <td className="px-4 py-2.5">
                    <span
                      className={`chip border ${
                        ACTION_STYLE[r.action] ?? "bg-surface-3 text-ink-mid border-surface-4"
                      }`}
                    >
                      {r.action}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 font-mono text-2xs text-ink-mid truncate max-w-[140px]">
                    {r.actor_id}
                  </td>
                  <td className="px-4 py-2.5 text-2xs text-ink-lo uppercase">{r.actor_role}</td>
                  <td className="px-4 py-2.5 font-mono text-2xs text-ink-lo truncate max-w-[120px]">
                    {r.target_id}
                  </td>
                  <td className="px-4 py-2.5 text-2xs text-ink-lo font-mono truncate max-w-[220px]">
                    {r.before && r.after ? summariseChange(r.before, r.after) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="px-4 py-3 text-2xs text-ink-lo border-t border-surface-3 leading-relaxed">
            This table has no update or delete path in the API — the repository
            exposes only an insert. Reconstructing who released a blocked
            transaction should never require reading application logs.
          </p>
        </Panel>
      )}
    </div>
  );
}

/** Show only the fields that actually changed. */
function summariseChange(before: Record<string, any>, after: Record<string, any>): string {
  const diffs: string[] = [];
  for (const key of Object.keys(after)) {
    if (JSON.stringify(before[key]) !== JSON.stringify(after[key])) {
      diffs.push(`${key}: ${JSON.stringify(before[key])} → ${JSON.stringify(after[key])}`);
    }
  }
  return diffs.join(", ") || "—";
}
