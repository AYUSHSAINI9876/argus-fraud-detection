import type { Metadata } from "next";
import { StackProvider, StackTheme } from "@stackframe/stack";
import { stackServerApp } from "@/stack";
import { authDisabled } from "@/lib/auth";
import { Providers } from "@/components/providers";
import "./globals.css";

/**
 * Nothing in this console is statically prerenderable.
 *
 * Every route reads the session cookie and calls the risk API, so a cached
 * HTML shell would be either useless or a data leak. Declaring it here rather
 * than per-page also keeps the build from evaluating StackProvider at compile
 * time — prerendering `/_not-found` otherwise constructs the Stack Auth app
 * without credentials and fails the build.
 */
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: {
    default: "Argus — Risk Intelligence",
    template: "%s · Argus",
  },
  description:
    "Real-time transaction fraud detection with explainable decisions and analyst case management.",
  icons: { icon: "/favicon.svg" },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // StackProvider resolves the Stack Auth app eagerly, which throws when no
  // project ID is configured. In keyless local mode there is no session to
  // provide, so the provider is skipped entirely rather than mounted against
  // a client that cannot exist. See lib/auth.ts for why that mode is safe.
  const noAuth = authDisabled();

  const body = <Providers>{children}</Providers>;

  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-surface-0">
        {noAuth ? (
          body
        ) : (
          <StackProvider app={stackServerApp}>
            <StackTheme>{body}</StackTheme>
          </StackProvider>
        )}
      </body>
    </html>
  );
}
