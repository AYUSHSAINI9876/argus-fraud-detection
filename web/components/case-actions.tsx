"use client";

import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";
import { AlertTriangle, ArrowUpCircle, Check, Hand, Unlock, X } from "lucide-react";

/**
 * Case action bar.
 *
 * Two deliberate friction points, both because these actions are hard to
 * undo:
 *
 *  - a disposition requires a note, so "why" is captured while it is still in
 *    the analyst's head rather than reconstructed later from a score;
 *  - releasing a block asks for confirmation and is REVIEWER-gated, because
 *    that one moves money.
 */
export function CaseActions({
  caseId,
  status,
  decision,
  assignedTo,
  currentUserId,
  role,
  hasCopilotSummary,
}: {
  caseId: string;
  status: string;
  decision: string;
  assignedTo: string | null;
  currentUserId: string | null;
  role: string;
  hasCopilotSummary: boolean;
}) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();
  const [error, setError] = useState<string | null>(null);
  const [showDisposition, setShowDisposition] = useState<"fraud" | "legitimate" | null>(null);
  const [note, setNote] = useState("");
  const [copilotAccepted, setCopilotAccepted] = useState<boolean | null>(null);

  const isResolved = status.startsWith("resolved");
  const isMine = assignedTo === currentUserId;
  const canAct = ["ANALYST", "REVIEWER", "ADMIN"].includes(role);
  const canRelease = ["REVIEWER", "ADMIN"].includes(role);

  async function call(action: string, body?: unknown) {
    setError(null);
    const res = await fetch(`/api/cases/${caseId}/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      setError(data.detail ?? `Request failed (${res.status})`);
      return false;
    }
    startTransition(() => router.refresh());
    return true;
  }

  if (isResolved) {
    return (
      <span className="chip bg-surface-3 text-ink-mid border border-surface-4">
        <Check size={11} /> {status.replace("resolved_", "")}
      </span>
    );
  }

  if (!canAct) {
    return (
      <span className="text-xs text-ink-lo">
        Read-only — analyst role required to act
      </span>
    );
  }

  return (
    <div className="flex flex-col items-end gap-2">
      <div className="flex items-center gap-1.5">
        {!isMine && (
          <button
            className="btn-ghost"
            disabled={pending}
            onClick={() => call("claim")}
          >
            <Hand size={13} /> Claim
          </button>
        )}

        <button
          className="btn-ghost"
          disabled={pending}
          onClick={() => call("escalate")}
        >
          <ArrowUpCircle size={13} /> Escalate
        </button>

        {decision === "block" && canRelease && (
          <button
            className="btn-danger"
            disabled={pending}
            onClick={() => {
              if (
                confirm(
                  "Release this block?\n\nThe transaction will be permitted and the customer's funds will move. This action is recorded in the audit log.",
                )
              ) {
                call("release");
              }
            }}
          >
            <Unlock size={13} /> Release block
          </button>
        )}

        <button
          className="btn bg-risk-low/15 text-risk-low border border-risk-low/30 hover:bg-risk-low/25"
          disabled={pending}
          onClick={() => setShowDisposition("legitimate")}
        >
          <Check size={13} /> Legitimate
        </button>
        <button
          className="btn-danger"
          disabled={pending}
          onClick={() => setShowDisposition("fraud")}
        >
          <X size={13} /> Confirm fraud
        </button>
      </div>

      {showDisposition && (
        <div className="panel p-3 w-80 space-y-2 animate-slide-up">
          <label className="field-label block">
            Why? (required — becomes part of the case record)
          </label>
          <textarea
            className="w-full rounded-md bg-surface-2 border border-surface-4 px-2.5 py-1.5
                       text-sm text-ink-hi placeholder:text-ink-lo resize-none
                       focus:outline-none focus:ring-1 focus:ring-accent"
            rows={3}
            value={note}
            placeholder={
              showDisposition === "fraud"
                ? "e.g. Customer confirmed they did not authorise this."
                : "e.g. Customer verified travel to Dubai."
            }
            onChange={(e) => setNote(e.target.value)}
            autoFocus
          />

          {hasCopilotSummary && (
            <label className="flex items-center gap-2 text-xs text-ink-mid">
              <input
                type="checkbox"
                checked={copilotAccepted === true}
                onChange={(e) => setCopilotAccepted(e.target.checked)}
                className="accent-accent"
              />
              Copilot summary was accurate
            </label>
          )}

          <div className="flex justify-end gap-1.5">
            <button
              className="btn-ghost"
              onClick={() => {
                setShowDisposition(null);
                setNote("");
              }}
            >
              Cancel
            </button>
            <button
              className="btn-primary"
              disabled={note.trim().length === 0 || pending}
              onClick={async () => {
                const ok = await call("disposition", {
                  verdict: showDisposition,
                  note: note.trim(),
                  copilot_accepted: copilotAccepted,
                });
                if (ok) {
                  setShowDisposition(null);
                  setNote("");
                }
              }}
            >
              Submit
            </button>
          </div>
        </div>
      )}

      {error && (
        <p className="text-xs text-risk-severe flex items-center gap-1">
          <AlertTriangle size={12} /> {error}
        </p>
      )}
    </div>
  );
}
