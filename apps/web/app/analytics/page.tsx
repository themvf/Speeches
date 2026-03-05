import type { Metadata } from "next";
import { PolicyResearchHub } from "@/components/policy-research-hub";

export const metadata: Metadata = {
  title: "Analytics | Policy Research Hub"
};

export default function AnalyticsPage() {
  return <PolicyResearchHub mode="analytics" />;
}
