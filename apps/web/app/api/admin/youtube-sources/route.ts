import { createHash } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";
import { loadYouTubeChannelSources, saveYouTubeChannelSources } from "@/lib/server/data-store";
import type { YouTubeChannelSource } from "@/lib/server/types";

export const dynamic = "force-dynamic";

function normalizeText(value: unknown): string {
  return String(value ?? "").trim();
}

function clampInt(value: unknown, fallback: number, minValue: number, maxValue: number): number {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  const n = Number.isFinite(parsed) ? parsed : fallback;
  return Math.max(minValue, Math.min(maxValue, n));
}

function sourceId(channelRef: string): string {
  return createHash("sha256").update(channelRef.toLowerCase()).digest("hex").slice(0, 16);
}

export async function GET(): Promise<NextResponse> {
  try {
    const payload = await loadYouTubeChannelSources();
    return NextResponse.json({ ok: true, data: payload });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  try {
    const body = (await req.json()) as Record<string, unknown>;
    const channelRef = normalizeText(body.channel_ref || body.channelRef);
    const label = normalizeText(body.label) || "YouTube Source";

    if (!channelRef) {
      return NextResponse.json({ ok: false, error: "channel_ref is required" }, { status: 400 });
    }

    const existing = await loadYouTubeChannelSources();
    const id = sourceId(channelRef);
    if (existing.sources.some((source) => source.id === id || source.channel_ref.toLowerCase() === channelRef.toLowerCase())) {
      return NextResponse.json({ ok: false, error: "This YouTube channel is already configured" }, { status: 409 });
    }

    const now = new Date().toISOString();
    const source: YouTubeChannelSource = {
      id,
      label,
      channel_ref: channelRef,
      active: typeof body.active === "boolean" ? body.active : true,
      extraction_limit: clampInt(body.extraction_limit || body.extractionLimit, 10, 1, 50),
      max_pages: clampInt(body.max_pages || body.maxPages, 1, 1, 5),
      enrich_limit: clampInt(body.enrich_limit || body.enrichLimit, 10, 1, 50),
      added_at: now,
      updated_at: now
    };

    const saved = await saveYouTubeChannelSources({
      ...existing,
      sources: [...existing.sources, source]
    });

    if (!saved.saved) {
      return NextResponse.json({ ok: false, error: "Failed to save YouTube source settings" }, { status: 500 });
    }

    return NextResponse.json({ ok: true, data: { source, saved } });
  } catch (err) {
    return NextResponse.json({ ok: false, error: String(err) }, { status: 500 });
  }
}
