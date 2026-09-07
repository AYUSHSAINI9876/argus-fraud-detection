import { notFound } from "next/navigation";
import { StackHandler } from "@stackframe/stack";
import { stackServerApp } from "@/stack";
import { authDisabled } from "@/lib/auth";

export const dynamic = "force-dynamic";

/**
 * Catch-all for Stack Auth's hosted flows: sign-in, sign-up, password reset,
 * email verification, OAuth callback, account settings.
 *
 * Using the hosted handler rather than hand-rolling these pages is a
 * deliberate choice — auth UI is where custom implementations quietly get
 * token handling and verification flows wrong.
 */
export default function Handler(props: unknown) {
  // In keyless local mode there is no Stack Auth project to host these flows,
  // and mounting the handler would throw. The console signs everyone in as a
  // local admin there, so these routes genuinely do not exist.
  if (authDisabled()) notFound();

  return <StackHandler fullPage app={stackServerApp} routeProps={props} />;
}
