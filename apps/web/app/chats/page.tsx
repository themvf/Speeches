import type { Metadata } from "next";
import { PolicyResearchHub } from "@/components/policy-research-hub";
import { PasswordGate } from "@/components/password-gate";

export const metadata: Metadata = {
  title: "Agentic Chats | Policy Research Hub"
};

export default function ChatsPage() {
  return (
    <PasswordGate>
      <PolicyResearchHub mode="chats" />
    </PasswordGate>
  );
}
