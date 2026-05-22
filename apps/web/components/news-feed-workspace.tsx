"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { IntelBetaDashboard } from "@/components/intelbeta-dashboard";
import { PolicyResearchHub } from "@/components/policy-research-hub";
import type { StoredRssArticle, StoredRssTopicRule } from "@/lib/server/neon";
import type { DocumentListItem } from "@/lib/server/types";

type WorkspaceMode = "feed" | "research";

function activeModeFromParams(value: string | null): WorkspaceMode {
  return value === "research" ? "research" : "feed";
}

function modeButtonClass(active: boolean): string {
  return active
    ? "rounded-lg border border-[color:rgba(79,213,255,0.58)] bg-[color:rgba(79,213,255,0.16)] px-4 py-2 text-sm font-semibold text-[color:var(--ink)]"
    : "rounded-lg border border-transparent px-4 py-2 text-sm font-semibold text-[color:var(--ink-faint)] hover:border-[color:var(--line)] hover:bg-[color:rgba(79,213,255,0.08)] hover:text-[color:var(--ink)]";
}

export function NewsFeedWorkspace({
  initialArticles,
  initialTopicRules,
  initialDocuments
}: {
  initialArticles: StoredRssArticle[];
  initialTopicRules: StoredRssTopicRule[];
  initialDocuments: DocumentListItem[];
}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const mode = activeModeFromParams(searchParams.get("mode"));

  const setMode = (nextMode: WorkspaceMode) => {
    const params = new URLSearchParams(searchParams.toString());
    if (nextMode === "feed") {
      params.delete("mode");
    } else {
      params.set("mode", nextMode);
    }
    const query = params.toString();
    router.replace((query ? `${pathname}?${query}` : pathname) as never, { scroll: false });
  };

  return (
    <div className="space-y-4">
      <section className="panel p-3 md:p-4">
        <div className="flex flex-col justify-between gap-3 md:flex-row md:items-center">
          <div>
            <h1 className="text-xl font-semibold text-[color:var(--ink)]">News Feed</h1>
            <p className="mt-1 text-sm text-[color:var(--ink-faint)]">
              Scan live regulatory news or switch into corpus research without leaving the discovery workspace.
            </p>
          </div>
          <div className="inline-flex rounded-xl border border-[color:var(--line)] bg-[color:rgba(8,18,30,0.76)] p-1">
            <button type="button" className={modeButtonClass(mode === "feed")} onClick={() => setMode("feed")}>
              Feed
            </button>
            <button type="button" className={modeButtonClass(mode === "research")} onClick={() => setMode("research")}>
              Research
            </button>
          </div>
        </div>
      </section>

      {mode === "feed" ? (
        <IntelBetaDashboard
          initialArticles={initialArticles}
          initialTopicRules={initialTopicRules}
          initialDocuments={initialDocuments}
        />
      ) : (
        <div className="-mx-4 md:-mx-8">
          <PolicyResearchHub mode="home" />
        </div>
      )}
    </div>
  );
}
