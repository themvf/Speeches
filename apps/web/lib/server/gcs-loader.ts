import fs from "node:fs";

import { getDataSourceConfig } from "@/lib/server/env";

let cachedStorage: any = null;
let cachedError = "";

function parseCredentialsFromRaw(raw: string): Record<string, unknown> | null {
  const text = String(raw || "").trim();
  if (!text) {
    return null;
  }

  const candidates = [text];
  if (text.length >= 2 && text[0] === text[text.length - 1] && ["'", '"'].includes(text[0])) {
    candidates.push(text.slice(1, -1).trim());
  }

  for (const candidate of candidates) {
    if (!candidate) {
      continue;
    }
    try {
      const parsed = JSON.parse(candidate);
      if (parsed && typeof parsed === "object") {
        return parsed as Record<string, unknown>;
      }
    } catch {
      // Keep trying candidates.
    }
  }

  try {
    const decoded = Buffer.from(text, "base64").toString("utf-8").trim();
    const parsed = JSON.parse(decoded);
    if (parsed && typeof parsed === "object") {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // Ignore parse failures.
  }

  return null;
}

async function buildStorageClient() {
  const cfg = getDataSourceConfig();
  if (!cfg.gcsBucketName) {
    cachedError = "GCS bucket not configured.";
    return null;
  }

  let credentials: Record<string, unknown> | null = null;

  if (cfg.gcsCredentialsJson) {
    credentials = parseCredentialsFromRaw(cfg.gcsCredentialsJson);
    if (!credentials) {
      cachedError = "Failed to parse GCS_CREDENTIALS_JSON.";
      return null;
    }
  } else if (cfg.gcsCredentialsPath) {
    try {
      const raw = fs.readFileSync(cfg.gcsCredentialsPath, "utf-8");
      credentials = parseCredentialsFromRaw(raw) ?? JSON.parse(raw);
    } catch {
      cachedError = "Failed to read GCS credentials path.";
      return null;
    }
  }

  try {
    const mod = await import("@google-cloud/storage");
    if (credentials) {
      return new mod.Storage({ credentials });
    }
    return new mod.Storage();
  } catch {
    cachedError = "@google-cloud/storage is unavailable or failed to initialize.";
    return null;
  }
}

export async function getStorageClient() {
  if (cachedStorage) {
    return cachedStorage;
  }
  const client = await buildStorageClient();
  if (!client) {
    return null;
  }
  cachedStorage = client;
  cachedError = "";
  return client;
}

export function getLastStorageError(): string {
  return cachedError;
}

export async function downloadGcsJson<T>(blobName: string): Promise<T | null> {
  const cfg = getDataSourceConfig();
  const client = await getStorageClient();
  if (!client || !cfg.gcsBucketName) {
    return null;
  }

  try {
    const bucket = client.bucket(cfg.gcsBucketName);
    const file = bucket.file(blobName);
    const [exists] = await file.exists();
    if (!exists) {
      return null;
    }
    const [text] = await file.download();
    return JSON.parse(text.toString("utf-8")) as T;
  } catch (err) {
    console.error(`[gcs] downloadGcsJson failed for "${blobName}":`, err);
    cachedError = `Download failed for ${blobName}: ${String(err)}`;
    return null;
  }
}

/**
 * Like downloadGcsJson, but distinguishes "blob doesn't exist yet" (safe to
 * treat as empty) from "read failed" (client not configured, network error,
 * malformed JSON, etc). Callers that read-modify-write a blob should use this
 * so a transient read failure can't be silently treated as an empty corpus
 * and then overwrite real remote data on save.
 */
export type GcsJsonReadResult<T> =
  | { status: "found"; data: T; generation: number }
  | { status: "not_found"; generation: 0 }
  | { status: "error"; error: string };

export async function downloadGcsJsonSafe<T>(blobName: string): Promise<GcsJsonReadResult<T>> {
  const cfg = getDataSourceConfig();
  const client = await getStorageClient();
  if (!client || !cfg.gcsBucketName) {
    return { status: "error", error: getLastStorageError() || "GCS storage is not configured." };
  }

  try {
    const bucket = client.bucket(cfg.gcsBucketName);
    const file = bucket.file(blobName);
    // Read a stable object snapshot. A generation change between metadata and
    // content reads is retried so callers never pair stale JSON with a newer
    // generation and then overwrite the newer object.
    for (let attempt = 0; attempt < 2; attempt += 1) {
      try {
        const [before] = await file.getMetadata();
        const beforeGeneration = Number(before.generation ?? 0);
        const [text] = await file.download();
        const [after] = await file.getMetadata();
        const afterGeneration = Number(after.generation ?? 0);
        if (beforeGeneration !== afterGeneration) {
          continue;
        }
        return {
          status: "found",
          data: JSON.parse(text.toString("utf-8")) as T,
          generation: afterGeneration,
        };
      } catch (err) {
        if ((err as { code?: number })?.code === 404) {
          return { status: "not_found", generation: 0 };
        }
        throw err;
      }
    }
    return { status: "error", error: `Concurrent changes prevented a stable read of ${blobName}.` };
  } catch (err) {
    console.error(`[gcs] downloadGcsJsonSafe failed for "${blobName}":`, err);
    return { status: "error", error: String(err) };
  }
}

export type GcsConditionalWriteResult = "ok" | "conflict" | "error";

/**
 * Save only if the blob still has the generation returned by
 * downloadGcsJsonSafe. Generation 0 means create-only when the blob was
 * confirmed absent. This keeps read/modify/write callers from clobbering a
 * concurrent ingestion or admin update.
 */
export async function uploadGcsJsonWithPrecondition(
  blobName: string,
  payload: unknown,
  expectedGeneration: number
): Promise<GcsConditionalWriteResult> {
  const cfg = getDataSourceConfig();
  const client = await getStorageClient();
  if (!client || !cfg.gcsBucketName) {
    return "error";
  }

  try {
    const bucket = client.bucket(cfg.gcsBucketName);
    const file = bucket.file(blobName);
    await file.save(JSON.stringify(payload, null, 2), {
      resumable: false,
      contentType: "application/json; charset=utf-8",
      metadata: {
        cacheControl: "no-cache"
      },
      preconditionOpts: {
        ifGenerationMatch: expectedGeneration
      }
    });
    return "ok";
  } catch (err) {
    if ((err as { code?: number })?.code === 412) {
      return "conflict";
    }
    console.error(`[gcs] conditional upload failed for "${blobName}":`, err);
    return "error";
  }
}

export async function uploadGcsJson(blobName: string, payload: unknown): Promise<boolean> {
  const cfg = getDataSourceConfig();
  const client = await getStorageClient();
  if (!client || !cfg.gcsBucketName) {
    return false;
  }

  try {
    const bucket = client.bucket(cfg.gcsBucketName);
    const file = bucket.file(blobName);
    await file.save(JSON.stringify(payload, null, 2), {
      resumable: false,
      contentType: "application/json; charset=utf-8",
      metadata: {
        cacheControl: "no-cache"
      }
    });
    return true;
  } catch (err) {
    console.error(`[gcs] uploadGcsJson failed for "${blobName}":`, err);
    return false;
  }
}
