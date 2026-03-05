import type { Metadata } from "next";
import { PolicyResearchHub } from "@/components/policy-research-hub";

export const metadata: Metadata = {
  title: "Research Feed | Policy Research Hub"
};

export default function ResearchPage() {
  return <PolicyResearchHub mode="research" />;
}