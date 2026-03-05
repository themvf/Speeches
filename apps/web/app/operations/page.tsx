import type { Metadata } from "next";
import { PolicyResearchHub } from "@/components/policy-research-hub";

export const metadata: Metadata = {
  title: "Operations | Policy Research Hub"
};

export default function OperationsPage() {
  return <PolicyResearchHub mode="operations" />;
}