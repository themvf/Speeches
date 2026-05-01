"use client";

import { useEffect, useState } from "react";

const SESSION_KEY = "chats_unlocked";
const PASSWORD = "beta2025";

export function PasswordGate({ children }: { children: React.ReactNode }) {
  const [unlocked, setUnlocked] = useState(false);
  const [input, setInput] = useState("");
  const [error, setError] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (sessionStorage.getItem(SESSION_KEY) === "1") setUnlocked(true);
    setReady(true);
  }, []);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (input === PASSWORD) {
      sessionStorage.setItem(SESSION_KEY, "1");
      setUnlocked(true);
    } else {
      setError(true);
      setInput("");
    }
  }

  if (!ready) return null;
  if (unlocked) return <>{children}</>;

  return (
    <div
      style={{
        minHeight: "60vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <form
        onSubmit={submit}
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 12,
          width: 280,
          padding: "28px 24px",
          border: "1px solid rgba(99, 127, 170, 0.2)",
          background: "rgba(8,16,28,0.96)",
        }}
      >
        <div style={{ color: "#8fa7c8", fontSize: 11, letterSpacing: "0.16em", textTransform: "uppercase" }}>
          Beta Access
        </div>
        <input
          type="password"
          value={input}
          onChange={(e) => { setInput(e.target.value); setError(false); }}
          placeholder="Enter password"
          autoFocus
          style={{
            background: "rgba(255,255,255,0.04)",
            border: `1px solid ${error ? "rgba(255,107,127,0.6)" : "rgba(99,127,170,0.25)"}`,
            color: "#dbe7f5",
            padding: "8px 10px",
            fontSize: 13,
            outline: "none",
          }}
        />
        {error && (
          <div style={{ color: "#ff6b7f", fontSize: 12 }}>Incorrect password</div>
        )}
        <button
          type="submit"
          style={{
            background: "rgba(79,213,255,0.12)",
            border: "1px solid rgba(79,213,255,0.28)",
            color: "#4fd5ff",
            padding: "8px 0",
            fontSize: 12,
            letterSpacing: "0.1em",
            textTransform: "uppercase",
            cursor: "pointer",
          }}
        >
          Unlock
        </button>
      </form>
    </div>
  );
}
