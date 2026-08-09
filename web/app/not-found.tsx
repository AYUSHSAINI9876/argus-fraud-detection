import Link from "next/link";
import { FileQuestion } from "lucide-react";

/**
 * Explicit 404.
 *
 * Next generates a built-in `/_not-found` and prerenders it statically. That
 * page inherits the root layout — including StackProvider — so it gets built
 * without credentials and fails. Defining the route ourselves lets us opt it
 * out of prerendering alongside everything else.
 */
export const dynamic = "force-dynamic";

export default function NotFound() {
  return (
    <div className="min-h-screen grid place-items-center bg-surface-0 p-6">
      <div className="panel max-w-md w-full p-8 text-center">
        <FileQuestion
          size={30}
          className="text-ink-lo mx-auto mb-4"
          strokeWidth={1.5}
        />
        <h1 className="text-lg font-semibold tracking-tight mb-2">
          Page not found
        </h1>
        <p className="text-sm text-ink-mid mb-6">
          That route does not exist. If you followed a link to a case, it may
          have been resolved and archived.
        </p>
        <Link href="/" className="btn-primary">
          Back to overview
        </Link>
      </div>
    </div>
  );
}
