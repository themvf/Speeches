"use client";

import { useEffect } from "react";

export default function Error({
  error,
  reset
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="mx-auto flex min-h-[60vh] w-full max-w-4xl flex-col items-center justify-center gap-4 px-6 text-center">
      <p className="kicker">Something went wrong</p>
      <h2 className="text-3xl font-semibold">We could not load this view</h2>
      <p className="max-w-xl text-sm text-[color:rgba(16,36,59,0.72)]">{error.message || "An unexpected error occurred while loading the Policy Research Hub."}</p>
      <button
        onClick={() => reset()}
        className="rounded-xl bg-[color:#2d5673] px-4 py-2 text-sm font-semibold text-white"
      >
        Try again
      </button>
    </div>
  );
}