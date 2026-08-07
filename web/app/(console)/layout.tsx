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
} from "lucide-react";
import { stackServerApp } from "@/stack";

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
  const user = await stackServerApp.getUser();
  if (!user) redirect("/handler/sign-in");

  // Role lives in Stack Auth's server metadata; VIEWER is the safe default
  // for a user who has signed up but not yet been granted anything.
  const role =
    ((user.serverMetadata as Record<string, unknown> | null)?.role as string) ??
    "VIEWER";
  const isAdmin = role === "ADMIN";

  return (
    <div className="flex min-h-screen">
      <aside className="w-56 shrink-0 border-r border-surface-3 bg-surface-1 flex flex-col">
        <div className="flex items-center gap-2 px-4 h-14 border-b border-surface-3">
          <ShieldCheck size={18} className="text-accent" strokeWidth={2.5} />
          <span className="font-semibold tracking-tight">Argus</span>
          <span className="ml-auto chip bg-surface-3 text-ink-lo">v0.1</span>
        </div>

        <nav className="flex-1 p-2 space-y-0.5">
          {NAV.filter((i) => !i.minRole || isAdmin).map(({ href, label, icon: Icon }) => (
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

        <div className="border-t border-surface-3 p-3 flex items-center gap-2">
          <UserButton />
          <div className="min-w-0">
            <div className="text-xs font-medium truncate text-ink-hi">
              {user.displayName ?? user.primaryEmail}
            </div>
            <div className="text-2xs text-ink-lo uppercase tracking-wide">{role}</div>
          </div>
        </div>
      </aside>

      <main className="flex-1 min-w-0 overflow-x-hidden">{children}</main>
    </div>
  );
}
