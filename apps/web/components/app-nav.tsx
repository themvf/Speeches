"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

type NavItem = {
  href: "/" | "/research" | "/operations";
  label: string;
  prefetch?: boolean;
};

const NAV_ITEMS: NavItem[] = [
  { href: "/", label: "Overview", prefetch: true },
  { href: "/research", label: "Research Feed", prefetch: true },
  { href: "/operations", label: "Operations", prefetch: true }
];

function isActive(pathname: string, href: NavItem["href"]): boolean {
  if (href === "/") {
    return pathname === "/";
  }
  return pathname === href || pathname.startsWith(`${href}/`);
}

function navLinkClass(active: boolean): string {
  return active
    ? "rounded-xl border border-[color:rgba(16,36,59,0.35)] bg-white px-3 py-2 text-sm font-semibold text-[color:#10243b]"
    : "rounded-xl border border-transparent px-3 py-2 text-sm font-medium text-[color:rgba(16,36,59,0.72)] hover:border-[color:var(--line)] hover:bg-white/80";
}

export function AppNav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, []);

  return (
    <header className="sticky top-0 z-40 border-b border-[color:var(--line)] bg-white/70 backdrop-blur-md">
      <div className="mx-auto flex w-full max-w-7xl items-center justify-between gap-3 px-4 py-3 md:px-8">
        <Link href="/" prefetch className="inline-flex items-center gap-2 rounded-xl px-2 py-1 text-sm font-semibold text-[color:#10243b]">
          <span className="h-2.5 w-2.5 rounded-full bg-[color:#c77d28]" aria-hidden="true" />
          Policy Research Hub
        </Link>

        <nav aria-label="Primary" className="hidden items-center gap-1 md:flex">
          {NAV_ITEMS.map((item) => {
            const active = isActive(pathname, item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                prefetch={item.prefetch}
                className={navLinkClass(active)}
                aria-current={active ? "page" : undefined}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        <button
          type="button"
          onClick={() => setOpen((prev) => !prev)}
          aria-expanded={open}
          aria-controls="mobile-nav-drawer"
          aria-label="Toggle navigation"
          className="min-h-11 min-w-11 rounded-xl border border-[color:var(--line)] bg-white px-3 text-sm font-semibold md:hidden"
        >
          {open ? "Close" : "Menu"}
        </button>
      </div>

      {open ? (
        <>
          <div className="fixed inset-0 z-40 bg-black/35 md:hidden" onClick={() => setOpen(false)} aria-hidden="true" />
          <nav
            id="mobile-nav-drawer"
            aria-label="Mobile"
            className="fixed inset-y-0 left-0 z-50 w-72 border-r border-[color:var(--line)] bg-[color:#fefcf7] p-5 md:hidden"
          >
            <p className="text-xs font-semibold uppercase tracking-[0.1em] text-[color:rgba(16,36,59,0.68)]">Navigate</p>
            <div className="mt-3 space-y-2">
              {NAV_ITEMS.map((item) => {
                const active = isActive(pathname, item.href);
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    prefetch={item.prefetch}
                    className={`block min-h-11 rounded-xl px-3 py-2 ${navLinkClass(active)}`}
                    aria-current={active ? "page" : undefined}
                  >
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </nav>
        </>
      ) : null}
    </header>
  );
}
