import type { Metadata } from "next";
import { PolicyResearchHub } from "@/components/policy-research-hub";

export const metadata: Metadata = {
  title: "Research | Policy Research Hub",
  description: "Search and explore primary regulatory documents with AI analysis.",
};

export default function ResearchPage() {
  return <PolicyResearchHub mode="home" />;
}
