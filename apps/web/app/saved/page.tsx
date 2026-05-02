import type { Metadata } from "next";
import { SavedItemsPage } from "@/components/saved-items-page";

export const metadata: Metadata = {
  title: "Saved Research | Policy Research Hub",
  description: "Saved articles and documents for research follow-up.",
};

export default function SavedPage() {
  return <SavedItemsPage />;
}
