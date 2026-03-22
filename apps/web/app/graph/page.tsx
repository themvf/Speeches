import type { Metadata } from "next";

import { GraphView } from "@/components/graph-view";

export const metadata: Metadata = {
  title: "Graph | Policy Research Hub"
};

export default function GraphPage() {
  return <GraphView />;
}
