"use client";

import { useEffect, useRef, useState } from "react";

type TickerEntry = { symbol: string; name: string };
type ValidationResult =
  | { valid: false; error: string }
  | { valid: true; symbol: string; name: string; price: number; change: number; pct: number; up: boolean };

const MAX = 10;

export default function AdminPage() {
  const [tickers, setTickers] = useState<TickerEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<"idle" | "ok" | "error">("idle");
  const [saveError, setSaveError] = useState<string | null>(null);

  const [input, setInput] = useState("");
  const [validating, setValidating] = useState(false);
  const [preview, setPreview] = useState<ValidationResult | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch("/api/admin/ticker")
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setTickers(data);
      })
      .finally(() => setLoading(false));
  }, []);

  async function handleConfirm() {
    const sym = input.trim().toUpperCase();
    if (!sym) return;
    if (tickers.length >= MAX) return;
    if (tickers.some((t) => t.symbol === sym)) {
      setPreview({ valid: false, error: `${sym} is already in the list` });
      return;
    }
    setValidating(true);
    setPreview(null);
    try {
      const res = await fetch("/api/admin/ticker/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: sym }),
      });
      const data: ValidationResult = await res.json();
      setPreview(data);
      if (data.valid) {
        setTickers((prev) => [...prev, { symbol: data.symbol, name: data.name }]);
        setInput("");
      }
    } finally {
      setValidating(false);
    }
  }

  function handleRename(symbol: string, name: string) {
    setTickers((prev) => prev.map((t) => (t.symbol === symbol ? { ...t, name } : t)));
    setSaveStatus("idle");
  }

  function handleRemove(symbol: string) {
    setTickers((prev) => prev.filter((t) => t.symbol !== symbol));
    setSaveStatus("idle");
  }

  async function handleSave() {
    setSaving(true);
    setSaveStatus("idle");
    setSaveError(null);
    try {
      const res = await fetch("/api/admin/ticker", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(tickers),
      });
      if (res.ok) {
        setSaveStatus("ok");
      } else {
        const body = await res.json().catch(() => ({}));
        setSaveError(body?.error ?? `HTTP ${res.status}`);
        setSaveStatus("error");
      }
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : "Network error");
      setSaveStatus("error");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="mx-auto max-w-xl px-4 py-12">
      <p className="mb-1 text-xs font-bold uppercase tracking-[0.1em] text-[color:var(--ink-faint)]">
        Admin
      </p>
      <h1 className="mb-8 text-2xl font-bold text-[color:var(--ink)]">Ticker Bar Configuration</h1>

      {/* Current tickers */}
      <section className="mb-8">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
            Active Tickers
          </h2>
          <span className="text-xs text-[color:var(--ink-faint)]">
            {tickers.length} / {MAX}
          </span>
        </div>

        {loading ? (
          <p className="text-sm text-[color:var(--ink-faint)]">Loading…</p>
        ) : tickers.length === 0 ? (
          <p className="text-sm text-[color:var(--ink-faint)]">No tickers configured.</p>
        ) : (
          <ul className="space-y-2">
            {tickers.map((t) => (
              <li
                key={t.symbol}
                className="flex items-center gap-3 rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,22,36,0.88)] px-4 py-3"
              >
                <span className="w-14 flex-shrink-0 font-mono text-sm font-bold text-[color:var(--accent)]">
                  {t.symbol}
                </span>
                <input
                  type="text"
                  value={t.name}
                  onChange={(e) => handleRename(t.symbol, e.target.value)}
                  placeholder="Display name"
                  className="form-control min-w-0 flex-1 px-2 py-1 text-sm"
                />
                <button
                  type="button"
                  onClick={() => handleRemove(t.symbol)}
                  className="flex-shrink-0 rounded-lg border border-[color:rgba(255,107,127,0.4)] bg-[color:rgba(255,107,127,0.1)] px-3 py-1 text-xs font-semibold text-[color:var(--danger)] transition hover:bg-[color:rgba(255,107,127,0.2)]"
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* Add ticker */}
      <section className="mb-8">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-[0.08em] text-[color:var(--ink-faint)]">
          Add Ticker
        </h2>
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => {
              setInput(e.target.value.toUpperCase());
              setPreview(null);
            }}
            onKeyDown={(e) => e.key === "Enter" && handleConfirm()}
            placeholder="e.g. AAPL, SPY, ^VIX"
            disabled={tickers.length >= MAX}
            className="form-control flex-1 px-3 py-2 text-sm"
          />
          <button
            type="button"
            onClick={handleConfirm}
            disabled={!input.trim() || validating || tickers.length >= MAX}
            className="btn-solid min-w-[90px] rounded-xl px-4 py-2 text-sm disabled:opacity-40"
          >
            {validating ? "Checking…" : "Confirm"}
          </button>
        </div>

        {tickers.length >= MAX && (
          <p className="mt-2 text-xs text-[color:var(--warn)]">Maximum of {MAX} tickers reached.</p>
        )}

        {/* Validation result */}
        {preview && (
          <div
            className={`mt-3 rounded-xl border px-4 py-3 text-sm ${
              preview.valid
                ? "border-[color:rgba(65,211,157,0.48)] bg-[color:rgba(65,211,157,0.08)] text-[color:var(--ok)]"
                : "border-[color:rgba(255,107,127,0.48)] bg-[color:rgba(255,107,127,0.08)] text-[color:var(--danger)]"
            }`}
          >
            {preview.valid ? (
              <span>
                <strong>{preview.symbol}</strong> — {preview.name} &nbsp;
                <span className="font-mono">
                  ${preview.price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </span>
                &nbsp;
                <span className={preview.up ? "text-[color:var(--ok)]" : "text-[color:var(--danger)]"}>
                  {preview.up ? "▲" : "▼"} {Math.abs(preview.pct).toFixed(2)}%
                </span>
                &nbsp;— Added
              </span>
            ) : (
              preview.error
            )}
          </div>
        )}
      </section>

      {/* Save */}
      <div className="flex items-center gap-4">
        <button
          type="button"
          onClick={handleSave}
          disabled={saving || loading}
          className="btn-solid rounded-xl px-6 py-2.5 text-sm font-semibold disabled:opacity-40"
        >
          {saving ? "Saving…" : "Save Changes"}
        </button>
        {saveStatus === "ok" && (
          <span className="text-sm text-[color:var(--ok)]">Saved — ticker bar will update within 60 s</span>
        )}
        {saveStatus === "error" && (
          <span className="text-sm text-[color:var(--danger)]">
            Save failed{saveError ? `: ${saveError}` : " — try again"}
          </span>
        )}
      </div>
    </div>
  );
}
