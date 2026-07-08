import { createHash } from "node:crypto";

import { downloadGcsJson, uploadGcsJson } from "@/lib/server/gcs-loader";

export const YOUTUBE_CHANNELS_BLOB = "youtube_channel_sources.json";

export type YouTubeChannelConnector = "sec_youtube_video" | "youtube_video";

export type YouTubeChannelConfig = {
  id: string;
  label: string;
  channel_ref: string;
  active: boolean;
  extraction_limit: number;
  enrich_limit: number;
  max_pages: number;
  connector: YouTubeChannelConnector;
  added_at: string;
  updated_at: string;
  last_run_at?: string;
  last_status?: string;
  last_error?: string;
};

export type YouTubeChannelPayload = {
  version: number;
  updated_at: string;
  channels: YouTubeChannelConfig[];
};

const DEFAULT_SEC_CHANNEL: YouTubeChannelConfig = {
  id: "sec_views",
  label: "SEC",
  channel_ref: "https://www.youtube.com/user/SECViews",
  active: true,
  extraction_limit: 2,
  enrich_limit: 2,
  max_pages: 1,
  connector: "sec_youtube_video",
  added_at: "",
  updated_at: "",
};

function nowIso(): string {
  return new Date().toISOString();
}

function positiveInt(value: unknown, fallback: number, min = 1, max = 50): number {
  const parsed = Math.round(Number(value));
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

export function youtubeChannelId(channelRef: string): string {
  return createHash("sha256").update(channelRef.trim().toLowerCase()).digest("hex").slice(0, 16);
}

function normalizeConnector(value: unknown, fallback: YouTubeChannelConnector): YouTubeChannelConnector {
  return value === "sec_youtube_video" || value === "youtube_video" ? value : fallback;
}

function normalizeChannel(raw: unknown, fallbackConnector: YouTubeChannelConnector): YouTubeChannelConfig | null {
  if (!raw || typeof raw !== "object") return null;
  const item = raw as Partial<YouTubeChannelConfig>;
  const channelRef = String(item.channel_ref || "").trim();
  if (!channelRef) return null;
  const timestamp = nowIso();
  return {
    id: String(item.id || youtubeChannelId(channelRef)).trim(),
    label: String(item.label || channelRef).trim(),
    channel_ref: channelRef,
    active: item.active !== false,
    extraction_limit: positiveInt(item.extraction_limit, 2),
    enrich_limit: positiveInt(item.enrich_limit, 2),
    max_pages: positiveInt(item.max_pages, 1, 1, 10),
    connector: normalizeConnector(item.connector, fallbackConnector),
    added_at: String(item.added_at || timestamp),
    updated_at: String(item.updated_at || timestamp),
    last_run_at: item.last_run_at ? String(item.last_run_at) : undefined,
    last_status: item.last_status ? String(item.last_status) : undefined,
    last_error: item.last_error ? String(item.last_error) : undefined,
  };
}

function normalizePayload(raw: unknown): YouTubeChannelPayload {
  const source = raw && typeof raw === "object" ? raw as Partial<YouTubeChannelPayload> : {};
  const timestamp = nowIso();
  const rawChannels = Array.isArray(source.channels) ? source.channels : [];
  const channels = rawChannels
    .map((item) => normalizeChannel(item, "youtube_video"))
    .filter((item): item is YouTubeChannelConfig => Boolean(item));

  if (channels.length === 0) {
    channels.push({
      ...DEFAULT_SEC_CHANNEL,
      added_at: timestamp,
      updated_at: timestamp,
    });
  }

  return {
    version: 1,
    updated_at: String(source.updated_at || timestamp),
    channels,
  };
}

async function savePayload(payload: YouTubeChannelPayload): Promise<YouTubeChannelPayload> {
  const updated = {
    ...payload,
    updated_at: nowIso(),
  };
  const saved = await uploadGcsJson(YOUTUBE_CHANNELS_BLOB, updated);
  if (!saved) {
    throw new Error("Failed to write YouTube channel config to GCS");
  }
  return updated;
}

export async function getYouTubeChannelPayload(): Promise<YouTubeChannelPayload> {
  return normalizePayload(await downloadGcsJson<YouTubeChannelPayload>(YOUTUBE_CHANNELS_BLOB));
}

export async function upsertYouTubeChannel(input: {
  label: string;
  channelRef: string;
  extractionLimit?: unknown;
  enrichLimit?: unknown;
  maxPages?: unknown;
}): Promise<YouTubeChannelPayload> {
  const label = input.label.trim();
  const channelRef = input.channelRef.trim();
  if (!label || !channelRef) {
    throw new Error("label and channelRef are required");
  }

  const payload = await getYouTubeChannelPayload();
  const id = youtubeChannelId(channelRef);
  const timestamp = nowIso();
  const nextChannel: YouTubeChannelConfig = {
    id,
    label,
    channel_ref: channelRef,
    active: true,
    extraction_limit: positiveInt(input.extractionLimit, 2),
    enrich_limit: positiveInt(input.enrichLimit, 2),
    max_pages: positiveInt(input.maxPages, 1, 1, 10),
    connector: "youtube_video",
    added_at: timestamp,
    updated_at: timestamp,
  };

  const existingIndex = payload.channels.findIndex((channel) => channel.id === id || channel.channel_ref.toLowerCase() === channelRef.toLowerCase());
  if (existingIndex >= 0) {
    const existing = payload.channels[existingIndex];
    payload.channels[existingIndex] = {
      ...existing,
      label: nextChannel.label,
      channel_ref: nextChannel.channel_ref,
      active: true,
      extraction_limit: nextChannel.extraction_limit,
      enrich_limit: nextChannel.enrich_limit,
      max_pages: nextChannel.max_pages,
      connector: existing.connector,
      added_at: existing.added_at || timestamp,
      updated_at: timestamp,
    };
  } else {
    payload.channels.push(nextChannel);
  }

  return savePayload(payload);
}

export async function updateYouTubeChannel(channelId: string, patch: {
  active?: boolean;
  label?: string;
  extraction_limit?: unknown;
  enrich_limit?: unknown;
  max_pages?: unknown;
}): Promise<YouTubeChannelPayload> {
  const payload = await getYouTubeChannelPayload();
  const index = payload.channels.findIndex((channel) => channel.id === channelId);
  if (index < 0) {
    throw new Error("YouTube channel not found");
  }
  const current = payload.channels[index];
  payload.channels[index] = {
    ...current,
    active: typeof patch.active === "boolean" ? patch.active : current.active,
    label: typeof patch.label === "string" && patch.label.trim() ? patch.label.trim() : current.label,
    extraction_limit: patch.extraction_limit === undefined ? current.extraction_limit : positiveInt(patch.extraction_limit, current.extraction_limit),
    enrich_limit: patch.enrich_limit === undefined ? current.enrich_limit : positiveInt(patch.enrich_limit, current.enrich_limit),
    max_pages: patch.max_pages === undefined ? current.max_pages : positiveInt(patch.max_pages, current.max_pages, 1, 10),
    updated_at: nowIso(),
  };
  return savePayload(payload);
}

export async function deleteYouTubeChannel(channelId: string): Promise<YouTubeChannelPayload> {
  if (channelId === DEFAULT_SEC_CHANNEL.id) {
    throw new Error("The default SEC channel cannot be removed; deactivate it instead.");
  }
  const payload = await getYouTubeChannelPayload();
  const nextChannels = payload.channels.filter((channel) => channel.id !== channelId);
  if (nextChannels.length === payload.channels.length) {
    throw new Error("YouTube channel not found");
  }
  return savePayload({ ...payload, channels: nextChannels });
}
