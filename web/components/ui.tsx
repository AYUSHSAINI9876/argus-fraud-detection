import clsx from "clsx";
import { twMerge } from "tailwind-merge";
import {
  AlertTriangle,
  Ban,
  CheckCircle2,
  Eye,
  type LucideIcon,
} from "lucide-react";
import { DECISION_STYLE, RISK_BG, riskBand } from "@/lib/risk";

export function cn(...inputs: Parameters<typeof clsx>) {
  return twMerge(clsx(inputs));
}

/* -------------------------------------------------------------------------
   Risk and decision encoding
   ------------------------------------------------------------------------- */

/**
 * Risk is encoded three ways at once — colour, numeric score, and a word.
 * Colour alone fails for colour-blind users and fails again on a projector,
 * and this UI is used to justify blocking someone's card.
 */
export function RiskBadge({ score, showValue = true }: { score: number; showValue?: boolean }) {
  const { band, label } = riskBand(score);
  return (
    <span className={cn("chip border", RISK_BG[band])}>
      {showValue && <span className="tnum font-semibold">{(score * 100).toFixed(1)}%</span>}
      <span className="opacity-75">{label}</span>
    </span>
  );
}

const DECISION_ICON: Record<string, LucideIcon> = {
  allow: CheckCircle2,
  review: Eye,
  block: Ban,
};

export function DecisionBadge({ decision }: { decision: string }) {
  const Icon = DECISION_ICON[decision] ?? AlertTriangle;
  return (
    <span className={cn("chip border", DECISION_STYLE[decision] ?? "bg-surface-3 text-ink-mid")}>
      <Icon size={11} strokeWidth={2.5} />
      {decision}
    </span>
  );
}

/* -------------------------------------------------------------------------
   Layout primitives
   ------------------------------------------------------------------------- */

export function Panel({
  title,
  action,
  children,
  className,
}: {
  title?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("panel", className)}>
      {(title || action) && (
        <header className="panel-header">
          {title && (
            <h2 className="text-sm font-semibold text-ink-hi tracking-tight">{title}</h2>
          )}
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function StatTile({
  label,
  value,
  sub,
  icon: Icon,
  tone = "neutral",
}: {
  label: string;
  value: string;
  sub?: string;
  icon?: LucideIcon;
  tone?: "neutral" | "warn" | "danger" | "good";
}) {
  const toneClass = {
    neutral: "text-ink-hi",
    good: "text-risk-low",
    warn: "text-risk-moderate",
    danger: "text-risk-severe",
  }[tone];

  return (
    <div className="panel p-4 flex flex-col gap-1">
      <div className="flex items-center gap-1.5">
        {Icon && <Icon size={13} className="text-ink-lo" strokeWidth={2} />}
        <span className="field-label">{label}</span>
      </div>
      <div className={cn("text-2xl font-semibold tnum tracking-tight", toneClass)}>
        {value}
      </div>
      {sub && <div className="text-xs text-ink-lo tnum">{sub}</div>}
    </div>
  );
}

export function EmptyState({
  icon: Icon,
  title,
  detail,
}: {
  icon: LucideIcon;
  title: string;
  detail?: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-16 text-center">
      <Icon size={28} className="text-ink-lo" strokeWidth={1.5} />
      <p className="text-sm font-medium text-ink-mid">{title}</p>
      {detail && <p className="text-xs text-ink-lo max-w-xs">{detail}</p>}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded bg-surface-3", className)} />;
}
