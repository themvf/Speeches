"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";

function ChatLoginForm() {
  const searchParams = useSearchParams();
  const [secret, setSecret] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/chats/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ secret }),
      });
      const data = await res.json();
      if (data.ok) {
        window.location.assign(searchParams.get("next") ?? "/chats");
      } else {
        setError(data.error ?? "Invalid password");
      }
    } catch {
      setError("Network error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3">
      <input
        type="password"
        value={secret}
        onChange={(e) => { setSecret(e.target.value); setError(null); }}
        placeholder="Password"
        autoFocus
        className="form-control px-3 py-2 text-sm"
      />
      {error && <p className="text-sm text-[color:var(--danger)]">{error}</p>}
      <button
        type="submit"
        disabled={!secret || loading}
        className="btn-solid rounded-xl px-4 py-2 text-sm font-semibold disabled:opacity-40"
      >
        {loading ? "Checking..." : "Continue"}
      </button>
    </form>
  );
}

export default function ChatLoginPage() {
  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <div className="w-full max-w-sm">
        <p className="mb-1 text-xs font-bold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">Agentic Chats</p>
        <h1 className="mb-6 text-2xl font-bold text-[color:var(--ink)]">Sign in</h1>
        <Suspense>
          <ChatLoginForm />
        </Suspense>
      </div>
    </div>
  );
}
