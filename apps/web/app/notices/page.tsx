import type { Metadata } from "next";

import { NoticeCommentSection } from "@/components/notice-comment-section";

export const metadata: Metadata = {
  title: "Rulemakings & Comments | Policy Research Hub"
};

export default function NoticesPage() {
  return <NoticeCommentSection />;
}
