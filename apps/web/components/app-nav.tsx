"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

type NavItem = {
  href:
    | "/"
    | "/analytics"
    | "/chats"
    | "/notices"
    | "/trends"
    | "/intelligence"
    | "/intelbeta"
    | "/research"
    | "/enforcement"
    | "/market"
    | "/saved";
  label: string;
  prefetch?: boolean;
};

const NAV_ITEMS: NavItem[] = [
  { href: "/", label: "News Feed", prefetch: true },
  { href: "/research", label: "Research", prefetch: true },
  { href: "/trends", label: "Trends", prefetch: true },
  { href: "/enforcement", label: "Enforcement", prefetch: true },
  { href: "/market", label: "Market", prefetch: true },
  { href: "/saved", label: "Saved", prefetch: true },
  { href: "/notices", label: "Rulemakings & Comments", prefetch: true },
  { href: "/chats", label: "Agentic Chats", prefetch: true }
];

function isActive(pathname: string, href: NavItem["href"]): boolean {
  if (href === "/") {
    return pathname === "/";
  }
  return pathname === href || pathname.startsWith(`${href}/`);
}

function navLinkClass(active: boolean): string {
  return active
    ? "rounded-xl border border-[color:var(--line-strong)] bg-[color:rgba(15,32,50,0.92)] px-3 py-2 text-sm font-semibold text-[color:var(--ink)] shadow-[inset_0_1px_0_rgba(79,213,255,0.15)]"
    : "rounded-xl border border-transparent px-3 py-2 text-sm font-medium text-[color:var(--ink-faint)] hover:border-[color:var(--line)] hover:bg-[color:rgba(79,213,255,0.1)] hover:text-[color:var(--ink)]";
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
    <header className="sticky top-0 z-40 border-b border-[color:var(--line)] bg-[color:rgba(5,12,19,0.72)] backdrop-blur-md">
      <div className="mx-auto flex w-full max-w-7xl items-center justify-between gap-3 px-4 py-3 md:px-8">
        <Link href="/" prefetch className="inline-flex items-center gap-2 rounded-xl px-2 py-1 text-sm font-semibold text-[color:var(--ink)]">
          <span className="h-2.5 w-2.5 rounded-full bg-[color:var(--accent)]" aria-hidden="true" />
          Policy Research Hub
        </Link>

        <nav aria-label="Primary" className="hidden items-center gap-1 lg:flex">
          {NAV_ITEMS.map((item) => {
            const active = isActive(pathname, item.href);
            return (
              <Link
                key={item.href}
                href={item.href as any}
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
          className="min-h-11 min-w-11 rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.95)] px-3 text-sm font-semibold text-[color:var(--ink)] lg:hidden"
        >
          {open ? "Close" : "Menu"}
        </button>
      </div>

      {open ? (
        <>
          <div className="fixed inset-0 z-40 bg-black/35 lg:hidden" onClick={() => setOpen(false)} aria-hidden="true" />
          <nav
            id="mobile-nav-drawer"
            aria-label="Mobile"
            className="fixed inset-y-0 left-0 z-50 w-72 border-r border-[color:var(--line)] bg-[color:rgba(6,15,24,0.98)] p-5 lg:hidden"
          >
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-[color:var(--ink-faint)]">Navigate</p>
            <div className="mt-3 space-y-2">
              {NAV_ITEMS.map((item) => {
                const active = isActive(pathname, item.href);
                return (
                  <Link
                    key={item.href}
                    href={item.href as any}
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
