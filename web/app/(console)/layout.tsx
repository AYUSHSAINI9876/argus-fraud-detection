import Link from "next/link";
import { redirect } from "next/navigation";
import { UserButton } from "@stackframe/stack";
import {
  Activity,
  ClipboardList,
  LayoutDashboard,
  ScrollText,
  ShieldCheck,
  SlidersHorizontal,
  TriangleAlert,
} from "lucide-react";
import { authDisabled, getConsoleUser, isAdmin } from "@/lib/auth";

/**
 * Console shell.
 *
 * Auth is enforced here rather than in middleware so the check runs on the
 * server for every nested route — a client-side guard on a page that renders
 * customer financial data is not a guard at all.
 */

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/queue", label: "Case queue", icon: ClipboardList },
  { href: "/models", label: "Model health", icon: Activity },
  { href: "/policy", label: "Policy", icon: SlidersHorizontal, minRole: "ADMIN" },
  { href: "/audit", label: "Audit log", icon: ScrollText, minRole: "ADMIN" },
];

export default async function ConsoleLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const noAuth = authDisabled();
  const user = await getConsoleUser();
  if (!user) redirect("/handler/sign-in");

  const admin = isAdmin(user.role);

  return (
    <div className="flex min-h-screen">
      <aside className="w-56 shrink-0 border-r border-surface-3 bg-surface-1 flex flex-col">
        <div className="flex items-center gap-2 px-4 h-14 border-b border-surface-3">
          <ShieldCheck size={18} className="text-accent" strokeWidth={2.5} />
          <span className="font-semibold tracking-tight">Argus</span>
          <span className="ml-auto chip bg-surface-3 text-ink-lo">v0.1</span>
        </div>

        <nav className="flex-1 p-2 space-y-0.5">
          {NAV.filter((i) => !i.minRole || admin).map(({ href, label, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              className="flex items-center gap-2.5 rounded-md px-3 py-2 text-sm
                         text-ink-mid hover:bg-surface-2 hover:text-ink-hi
                         transition-colors"
            >
              <Icon size={15} strokeWidth={2} />
              {label}
            </Link>
          ))}
        </nav>

        {/* Local mode is stated permanently and in the risk palette. Someone
            demoing this should never be in doubt about whether the identity
            in the corner is real. */}
        {noAuth && (
          <div
            className="mx-2 mb-2 rounded-md border border-risk-moderate/30
                       bg-risk-moderate/10 px-2.5 py-2 flex gap-2"
          >
            <TriangleAlert
              size={13}
              className="text-risk-moderate shrink-0 mt-0.5"
              strokeWidth={2}
            />
            <div className="min-w-0">
              <div className="text-2xs font-semibold text-risk-moderate uppercase tracking-wide">
                Auth disabled
              </div>
              <div className="text-2xs text-ink-lo leading-snug mt-0.5">
                Local mode — set Stack Auth keys to enable sign-in.
              </div>
            </div>
          </div>
        )}

        <div className="border-t border-surface-3 p-3 flex items-center gap-2">
          {!noAuth && <UserButton />}
          <div className="min-w-0">
            <div className="text-xs font-medium truncate text-ink-hi">
              {user.displayName ?? user.primaryEmail}
            </div>
            <div className="text-2xs text-ink-lo uppercase tracking-wide">
              {user.role}
            </div>
          </div>
        </div>
      </aside>

      <main className="flex-1 min-w-0 overflow-x-hidden">{children}</main>
    </div>
  );
}
