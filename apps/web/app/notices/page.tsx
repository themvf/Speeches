import type { Metadata } from "next";
import { Suspense } from "react";

import { NoticeCommentSection } from "@/components/notice-comment-section";

export const metadata: Metadata = {
  title: "Rulemakings & Comments | Policy Research Hub"
};

export default function NoticesPage() {
  return (
    <Suspense fallback={<div className="mx-auto max-w-7xl px-4 py-10"><p className="text-sm">Loading...</p></div>}>
      <NoticeCommentSection />
    </Suspense>
  );
}
