import type { Metadata } from "next";
import { PolicyResearchHub } from "@/components/policy-research-hub";

export const metadata: Metadata = {
  title: "Agentic Chats | Policy Research Hub"
};

export default function ChatsPage() {
  return <PolicyResearchHub mode="chats" />;
}
