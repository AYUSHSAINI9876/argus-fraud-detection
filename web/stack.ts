import "server-only";
import { StackServerApp } from "@stackframe/stack";

/**
 * Stack Auth server app, constructed lazily.
 *
 * `tokenStore: "nextjs-cookie"` keeps the session in an httpOnly cookie, so
 * the access token is never readable from client JavaScript — the difference
 * between an XSS being a bug and an XSS being a breach.
 *
 * The FastAPI service verifies the same token independently against Stack
 * Auth's JWKS. Neither side trusts the other's word about who the caller is.
 *
 * ---
 *
 * Why the proxy rather than a plain `new StackServerApp(...)` at module scope:
 *
 * `next build` imports every route module to collect page data. A top-level
 * constructor therefore runs at *build* time, where Stack Auth validates the
 * project ID and throws if it is absent or a placeholder. That turns a missing
 * build-time environment variable into a hard build failure — which breaks
 * `git clone && npm run build` for anyone without credentials, and breaks the
 * Vercel build unless the keys happen to be present in the build environment
 * as well as at runtime.
 *
 * Deferring construction to first property access moves that validation to
 * request time, where it belongs: the build succeeds without secrets, and a
 * genuinely missing key fails loudly on the first authenticated request
 * instead of silently. Every existing `stackServerApp.getUser()` call site
 * keeps working unchanged.
 */
let instance: StackServerApp | undefined;

function getApp(): StackServerApp {
  if (!instance) {
    instance = new StackServerApp({
      tokenStore: "nextjs-cookie",
      urls: {
        signIn: "/handler/sign-in",
        signUp: "/handler/sign-up",
        afterSignIn: "/",
        afterSignUp: "/",
        afterSignOut: "/handler/sign-in",
      },
    });
  }
  return instance;
}

export const stackServerApp = new Proxy({} as StackServerApp, {
  get(_target, prop, receiver) {
    const app = getApp();
    const value = Reflect.get(app as object, prop, receiver);
    // Methods must stay bound to the real instance, or `this` is the proxy.
    return typeof value === "function" ? value.bind(app) : value;
  },
  has(_target, prop) {
    return Reflect.has(getApp() as object, prop);
  },
});
