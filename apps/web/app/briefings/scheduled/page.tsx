import type { Metadata } from "next";
import { BriefingsSubnav } from "@/components/briefings-subnav";
import { ScheduledEditorialDashboard } from "@/components/scheduled-editorial-dashboard";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Scheduled Editorial | Policy Research Hub",
  description: "Configure and review the nightly AI editorial briefing.",
};

export default function ScheduledBriefingsPage() {
  return (
    <main className="mx-auto w-full max-w-7xl px-4 py-6 md:px-8">
      <BriefingsSubnav />
      <ScheduledEditorialDashboard />
    </main>
  );
}
