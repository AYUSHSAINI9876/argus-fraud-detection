import { StackHandler } from "@stackframe/stack";
import { stackServerApp } from "@/stack";

/**
 * Catch-all for Stack Auth's hosted flows: sign-in, sign-up, password reset,
 * email verification, OAuth callback, account settings.
 *
 * Using the hosted handler rather than hand-rolling these pages is a
 * deliberate choice — auth UI is where custom implementations quietly get
 * token handling and verification flows wrong.
 */
export default function Handler(props: unknown) {
  return <StackHandler fullPage app={stackServerApp} routeProps={props} />;
}
