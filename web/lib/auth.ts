import "server-only";
import { stackServerApp } from "@/stack";

/**
 * One place that answers "who is calling, and what may they do".
 *
 * The API already supports a keyless local mode (`AUTH_ENABLED=false`, which
 * injects a synthetic admin). Before this module the console had no
 * equivalent: every page called `stackServerApp.getUser()`, and Stack Auth
 * throws on a missing project ID, so a fresh clone without credentials got a
 * 500 on every route rather than a working console. That asymmetry made
 * `git clone && npm run dev` a dead end.
 *
 * The bypass is deliberately narrow:
 *
 *  - it engages only when NEXT_PUBLIC_STACK_PROJECT_ID is absent, so
 *    configuring Stack Auth silently switches real auth back on;
 *  - it refuses to engage in a production runtime, throwing instead. That
 *    mirrors the API's `refusing to start: auth disabled in production` and
 *    means a forgotten env var on Vercel fails loudly rather than publishing
 *    an unauthenticated console over customer financial data.
 *
 * The throw is request-time, not build-time. Every route is `force-dynamic`,
 * so `next build` never evaluates this — which is what keeps a credential-free
 * `npm run build` working, as CI relies on.
 */

export interface ConsoleUser {
  id: string;
  displayName: string | null;
  primaryEmail: string | null;
  /** VIEWER | ANALYST | REVIEWER | ADMIN — mirrors the API's Role ladder. */
  role: string;
  /** Bearer token for the API. Null in local mode, where the API accepts none. */
  accessToken: string | null;
}

const LOCAL_USER: ConsoleUser = {
  id: "dev-local",
  displayName: "Local Developer",
  primaryEmail: "dev@argus.local",
  // ADMIN so the policy and audit pages are reachable while developing. It is
  // the same synthetic identity the API grants in its keyless mode.
  role: "ADMIN",
  accessToken: null,
};

export function authDisabled(): boolean {
  if (process.env.NEXT_PUBLIC_STACK_PROJECT_ID) return false;

  if (process.env.NODE_ENV === "production") {
    throw new Error(
      "NEXT_PUBLIC_STACK_PROJECT_ID is not set. Refusing to serve the console " +
        "unauthenticated in production — set the Stack Auth environment " +
        "variables (see .env.example).",
    );
  }
  return true;
}

/** The signed-in user, or null when auth is on and nobody is signed in. */
export async function getConsoleUser(): Promise<ConsoleUser | null> {
  if (authDisabled()) return LOCAL_USER;

  const user = await stackServerApp.getUser();
  if (!user) return null;

  const { accessToken } = await user.getAuthJson();
  // Role lives in Stack Auth's server metadata; VIEWER is the safe default for
  // someone who has signed up but not yet been granted anything.
  const role =
    ((user.serverMetadata as Record<string, unknown> | null)?.role as string) ??
    "VIEWER";

  return {
    id: user.id,
    displayName: user.displayName ?? null,
    primaryEmail: user.primaryEmail ?? null,
    role,
    accessToken: accessToken ?? null,
  };
}

/** Roles permitted to act on a case, rather than only read it. */
export function canAct(role: string): boolean {
  return ["ANALYST", "REVIEWER", "ADMIN"].includes(role);
}

export function isAdmin(role: string): boolean {
  return role === "ADMIN";
}
