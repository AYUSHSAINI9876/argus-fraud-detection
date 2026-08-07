import { cn } from "@/components/ui";

/**
 * One SHAP contribution, drawn as a diverging bar from a centre line.
 *
 * Bars are scaled against the largest absolute contribution in the *same*
 * case, not against a global constant. An analyst reads this to answer "what
 * drove THIS decision", so relative magnitude within the case is the
 * question; a global scale would flatten most cases into invisible stubs.
 */
export function AttributionBar({
  label,
  value,
  contribution,
  max,
}: {
  label: string;
  value: string;
  contribution: number;
  max: number;
}) {
  const pct = max > 0 ? (Math.abs(contribution) / max) * 100 : 0;
  const increases = contribution > 0;

  return (
    <div className="group">
      <div className="flex items-baseline justify-between gap-3 mb-1">
        <span className="text-xs text-ink-mid truncate">{label}</span>
        <span className="text-xs tnum text-ink-hi shrink-0 font-medium">{value}</span>
      </div>
      <div className="relative h-1.5 rounded-full bg-surface-3 overflow-hidden">
        <div
          className={cn(
            "absolute inset-y-0 rounded-full transition-all",
            increases ? "bg-risk-severe" : "bg-risk-low",
          )}
          style={{ width: `${Math.max(pct, 2)}%` }}
        />
      </div>
    </div>
  );
}
