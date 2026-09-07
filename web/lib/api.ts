/**
 * Typed client for the Argus API.
 *
 * The access token is attached server-side. Client components never hold a
 * raw token — they call Next.js route handlers or server actions, which
 * forward the request with credentials. That keeps the token out of the
 * browser's reach entirely.
 */

/**
 * Where the risk API lives, resolved server-side at request time.
 *
 * `API_URL` takes precedence over `NEXT_PUBLIC_API_URL` because this value is
 * only ever read on the server — by server components and the route handler,
 * never by the browser, which talks to same-origin Next routes instead. A
 * NEXT_PUBLIC_ variable is inlined at build time, so it cannot express "the
 * address differs between building and running", which is exactly the case
 * under docker-compose: the image is built with a localhost URL, but inside
 * the container `localhost:8000` is the console itself, not the API. That
 * mismatch made the composed console render with "Could not reach the risk
 * engine" on every page.
 *
 * NEXT_PUBLIC_API_URL stays as the fallback so existing setups keep working.
 */
export const API_BASE =
  process.env.API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000/api/v1";

const BASE = API_BASE;

export interface Attribution {
  feature: string;
  value: number;
  contribution: number;
  direction: "increases_risk" | "decreases_risk";
}

export interface CaseSummary {
  id: string;
  status: string;
  priority: number;
  assigned_to: string | null;
  transaction_id: string;
  customer_id: string;
  merchant_id: string;
  amount: number;
  risk_score: number;
  anomaly_score: number | null;
  decision: string;
  rationale: string;
  triggered_rule: string | null;
  model_version: string;
  attributions: Attribution[];
  expected_loss: number;
  copilot_summary: string | null;
  created_at: string;
  note_count: number;
}

export interface CaseDetail extends CaseSummary {
  features: Record<string, number>;
  notes: {
    id: string;
    author_id: string;
    author_name: string | null;
    body: string;
    created_at: string;
  }[];
}

export interface Overview {
  window_hours: number;
  transactions_scored: number;
  total_volume: number;
  avg_risk_score: number;
  block_count: number;
  review_count: number;
  block_rate: number;
  review_rate: number;
  blocked_value: number;
  latency_ms: { avg: number; p95: number; p99: number };
  cases_resolved: number;
  realised_precision: number | null;
}

export interface TimeseriesPoint {
  t: string;
  count: number;
  avg_risk: number;
  blocks: number;
  reviews: number;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly requestId?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  token: string | null,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...init?.headers,
    },
    cache: "no-store",
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* non-JSON error body — keep the status text */
    }
    throw new ApiError(detail, res.status, res.headers.get("x-request-id") ?? undefined);
  }

  return res.json() as Promise<T>;
}

export const api = {
  overview: (token: string | null, hours = 24) =>
    request<Overview>(`/metrics/overview?hours=${hours}`, token),

  timeseries: (token: string | null, hours = 48) =>
    request<{ points: TimeseriesPoint[] }>(`/metrics/timeseries?hours=${hours}`, token),

  modelMetrics: (token: string | null) =>
    request<Record<string, unknown>>(`/metrics/model`, token),

  cases: (
    token: string | null,
    opts: { status?: string; sort?: string; limit?: number; assignedToMe?: boolean } = {},
  ) => {
    const q = new URLSearchParams();
    if (opts.status) q.set("status", opts.status);
    if (opts.sort) q.set("sort", opts.sort);
    if (opts.limit) q.set("limit", String(opts.limit));
    if (opts.assignedToMe) q.set("assigned_to_me", "true");
    return request<CaseSummary[]>(`/cases?${q}`, token);
  },

  caseDetail: (token: string | null, id: string) =>
    request<CaseDetail>(`/cases/${id}`, token),

  claimCase: (token: string | null, id: string) =>
    request<CaseSummary>(`/cases/${id}/claim`, token, { method: "POST" }),

  addNote: (token: string | null, id: string, body: string) =>
    request<{ id: string }>(`/cases/${id}/notes`, token, {
      method: "POST",
      body: JSON.stringify({ body }),
    }),

  setDisposition: (
    token: string | null,
    id: string,
    verdict: "fraud" | "legitimate",
    note: string,
    copilotAccepted?: boolean,
  ) =>
    request<CaseSummary>(`/cases/${id}/disposition`, token, {
      method: "POST",
      body: JSON.stringify({ verdict, note, copilot_accepted: copilotAccepted }),
    }),

  escalate: (token: string | null, id: string) =>
    request<CaseSummary>(`/cases/${id}/escalate`, token, { method: "POST" }),

  releaseBlock: (token: string | null, id: string) =>
    request<CaseSummary>(`/cases/${id}/release`, token, { method: "POST" }),

  policy: (token: string | null) =>
    request<{ policy: Record<string, number>; breakeven_curve: unknown[] }>(
      `/admin/policy`,
      token,
    ),

  auditLog: (token: string | null, hours = 168) =>
    request<Record<string, unknown>[]>(`/admin/audit?hours=${hours}`, token),
};
