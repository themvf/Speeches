import type { Metadata } from "next";

import { TimelineView } from "@/components/timeline-view";

export const metadata: Metadata = {
  title: "Timeline | Policy Research Hub"
};

export default function TimelinePage() {
  return <TimelineView />;
}
