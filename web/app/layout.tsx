import type { Metadata } from "next";
import { StackProvider, StackTheme } from "@stackframe/stack";
import { stackServerApp } from "@/stack";
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
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-surface-0">
        <StackProvider app={stackServerApp}>
          <StackTheme>
            <Providers>{children}</Providers>
          </StackTheme>
        </StackProvider>
      </body>
    </html>
  );
}
