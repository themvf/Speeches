import type { Metadata } from "next";

import { NoticeCommentSection } from "@/components/notice-comment-section";

export const metadata: Metadata = {
  title: "Notices & Comments | Policy Research Hub"
};

export default function NoticesPage() {
  return <NoticeCommentSection />;
}
