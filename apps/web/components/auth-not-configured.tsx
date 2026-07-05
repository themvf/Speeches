import Link from "next/link";

export function AuthNotConfigured() {
  return (
    <main className="mx-auto grid min-h-[70vh] w-full max-w-xl place-items-center px-4 py-12">
      <section className="w-full rounded-lg border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.92)] p-6">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[color:var(--ink-faint)]">Optional login</p>
        <h1 className="mt-3 text-2xl font-semibold text-[color:var(--ink)]">Sign-in is not configured</h1>
        <p className="mt-3 text-sm leading-6 text-[color:var(--ink-muted)]">
          Add Clerk environment variables to enable account sync. Until then, saved items continue to work in this browser.
        </p>
        <Link
          href="/saved"
          className="mt-5 inline-flex rounded-xl border border-[color:var(--line)] px-4 py-2 text-sm font-semibold text-[color:var(--ink)] hover:border-[color:var(--line-strong)]"
        >
          Go to saved items
        </Link>
      </section>
    </main>
  );
}
