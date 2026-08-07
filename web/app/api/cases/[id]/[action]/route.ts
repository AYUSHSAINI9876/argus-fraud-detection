import { NextRequest, NextResponse } from "next/server";
import { stackServerApp } from "@/stack";

/**
 * Server-side proxy for case mutations.
 *
 * Client components never hold an access token. They POST here; this handler
 * resolves the session server-side, attaches the bearer token, and forwards
 * to FastAPI. The token stays in an httpOnly cookie the whole time, so an XSS
 * in the console cannot exfiltrate a credential that authorises releasing
 * blocked funds.
 *
 * The allow-list matters: without it this becomes an open proxy that lets a
 * caller reach any API path with the signed-in user's privileges.
 */
const ALLOWED = new Set(["claim", "notes", "disposition", "escalate", "release"]);

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ id: string; action: string }> },
) {
  const { id, action } = await params;

  if (!ALLOWED.has(action)) {
    return NextResponse.json({ detail: "Unknown action" }, { status: 400 });
  }

  const user = await stackServerApp.getUser();
  if (!user) {
    return NextResponse.json({ detail: "Not authenticated" }, { status: 401 });
  }
  const { accessToken } = await user.getAuthJson();

  const body = await req.text();

  const res = await fetch(`${BASE}/cases/${id}/${action}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
    },
    body: body || undefined,
    cache: "no-store",
  });

  const payload = await res.text();
  return new NextResponse(payload, {
    status: res.status,
    headers: { "Content-Type": "application/json" },
  });
}
