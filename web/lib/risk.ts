/**
 * Risk presentation rules — one place, so a score is encoded identically
 * everywhere it appears.
 *
 * The bands are not evenly spaced. Fraud scores are heavily skewed toward
 * zero, so linear bands would put essentially every transaction in the lowest
 * bucket and the ramp would carry no information. These cut points follow the
 * decision policy's own break-even structure instead.
 */

export type RiskBand = "minimal" | "low" | "moderate" | "elevated" | "severe";

const BANDS: { max: number; band: RiskBand; label: string }[] = [
  { max: 0.02, band: "minimal", label: "Minimal" },
  { max: 0.1, band: "low", label: "Low" },
  { max: 0.35, band: "moderate", label: "Moderate" },
  { max: 0.7, band: "elevated", label: "Elevated" },
  { max: 1.01, band: "severe", label: "Severe" },
];

export function riskBand(score: number): { band: RiskBand; label: string } {
  const hit = BANDS.find((b) => score < b.max) ?? BANDS[BANDS.length - 1];
  return { band: hit.band, label: hit.label };
}

export const RISK_TEXT: Record<RiskBand, string> = {
  minimal: "text-risk-minimal",
  low: "text-risk-low",
  moderate: "text-risk-moderate",
  elevated: "text-risk-elevated",
  severe: "text-risk-severe",
};

export const RISK_BG: Record<RiskBand, string> = {
  minimal: "bg-risk-minimal/12 text-risk-minimal border-risk-minimal/25",
  low: "bg-risk-low/12 text-risk-low border-risk-low/25",
  moderate: "bg-risk-moderate/12 text-risk-moderate border-risk-moderate/25",
  elevated: "bg-risk-elevated/12 text-risk-elevated border-risk-elevated/25",
  severe: "bg-risk-severe/12 text-risk-severe border-risk-severe/25",
};

export const DECISION_STYLE: Record<string, string> = {
  allow: "bg-risk-low/12 text-risk-low border-risk-low/25",
  review: "bg-risk-moderate/12 text-risk-moderate border-risk-moderate/25",
  block: "bg-risk-severe/12 text-risk-severe border-risk-severe/25",
};

export function formatCurrency(value: number, currency = "USD"): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: value >= 1000 ? 0 : 2,
  }).format(value);
}

export function formatCompact(value: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

/**
 * Turn a raw feature name into something an analyst can read without a
 * data dictionary. Unmapped names fall through de-snaked rather than being
 * hidden — a missing label should look untidy, not invisible.
 */
const FEATURE_LABELS: Record<string, string> = {
  amount: "Transaction amount",
  log_amount: "Amount (log scale)",
  amount_zscore: "Amount vs customer's normal",
  amount_to_mean_ratio: "Amount ÷ customer average",
  amount_to_limit_ratio: "Amount ÷ credit limit",
  threshold_proximity: "Proximity to $10k reporting threshold",
  implied_velocity_kmh: "Implied travel speed",
  distance_from_prev_km: "Distance from previous transaction",
  distance_from_home_km: "Distance from home",
  seconds_since_prev: "Time since previous transaction",
  txn_count_1h: "Transactions in last hour",
  txn_count_24h: "Transactions in last 24h",
  txn_count_7d: "Transactions in last 7 days",
  amount_sum_1h: "Spend in last hour",
  amount_sum_24h: "Spend in last 24h",
  unique_merchants_24h: "Distinct merchants (24h)",
  unique_countries_24h: "Distinct countries (24h)",
  merchant_concentration_24h: "Merchant spread (24h)",
  is_new_device: "First time on this device",
  is_new_country: "First time from this country",
  is_new_category: "First purchase in this category",
  is_foreign_country: "Outside home country",
  is_first_transaction: "Customer's first transaction",
  merchant_risk_index: "Merchant risk rating",
  account_age_days: "Account age",
  known_device_count: "Known devices on file",
  is_night: "Overnight transaction",
  entry_magstripe: "Magstripe entry (fallback)",
  entry_manual: "Manually keyed",
  channel_ecommerce: "Card-not-present",
  cat_crypto: "Crypto merchant",
  cat_money_transfer: "Money transfer",
  cat_gambling: "Gambling merchant",
  cat_electronics: "Electronics merchant",
};

export function featureLabel(name: string): string {
  if (FEATURE_LABELS[name]) return FEATURE_LABELS[name];
  return name.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

/** Format a feature value with a unit appropriate to the feature. */
export function featureValue(name: string, value: number): string {
  if (name.startsWith("is_") || name.startsWith("cat_") || name.startsWith("channel_") || name.startsWith("entry_")) {
    return value >= 0.5 ? "Yes" : "No";
  }
  if (name === "seconds_since_prev") {
    if (value < 0) return "—";
    if (value < 60) return `${Math.round(value)}s`;
    if (value < 3600) return `${Math.round(value / 60)}m`;
    return `${(value / 3600).toFixed(1)}h`;
  }
  if (name.includes("_km")) return value < 0 ? "—" : `${Math.round(value).toLocaleString()} km`;
  if (name === "implied_velocity_kmh") return value < 0 ? "—" : `${Math.round(value).toLocaleString()} km/h`;
  if (name.startsWith("amount") && !name.includes("ratio") && !name.includes("zscore")) {
    return formatCurrency(value);
  }
  if (name === "account_age_days") return `${Math.round(value).toLocaleString()} days`;
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(3);
}
