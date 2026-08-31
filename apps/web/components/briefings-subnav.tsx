"use client";

import { usePathname } from "next/navigation";

const ITEMS = [
  { href: "/briefings", label: "Custom Builder" },
  { href: "/briefings/scheduled", label: "Scheduled Editorial" },
] as const;

export function BriefingsSubnav() {
  const pathname = usePathname();
  return (
    <nav aria-label="Briefings sections" className="mb-5 flex flex-wrap gap-2">
      {ITEMS.map((item) => {
        const active = pathname === item.href;
        return (
          <a
            key={item.href}
            href={item.href}
            className={active
              ? "rounded-full border border-[color:rgba(79,213,255,0.55)] bg-[color:rgba(79,213,255,0.14)] px-4 py-2 text-sm font-semibold text-[color:var(--ink)]"
              : "rounded-full border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.72)] px-4 py-2 text-sm font-semibold text-[color:var(--ink-faint)] hover:border-[color:var(--line-strong)] hover:text-[color:var(--ink)]"}
          >
            {item.label}
          </a>
        );
      })}
    </nav>
  );
}
