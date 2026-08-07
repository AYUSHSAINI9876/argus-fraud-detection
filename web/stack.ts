import "server-only";
import { StackServerApp } from "@stackframe/stack";

/**
 * Stack Auth server app.
 *
 * `tokenStore: "nextjs-cookie"` keeps the session in an httpOnly cookie, so
 * the access token is never readable from client JavaScript — the difference
 * between an XSS being a bug and an XSS being a breach.
 *
 * The FastAPI service verifies the same token independently against Stack
 * Auth's JWKS. Neither side trusts the other's word about who the caller is.
 */
export const stackServerApp = new StackServerApp({
  tokenStore: "nextjs-cookie",
  urls: {
    signIn: "/handler/sign-in",
    signUp: "/handler/sign-up",
    afterSignIn: "/",
    afterSignUp: "/",
    afterSignOut: "/handler/sign-in",
  },
});
