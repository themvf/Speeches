import fs from "node:fs";
import path from "node:path";

import { getDataSourceConfig } from "@/lib/server/env";
import { downloadGcsJson } from "@/lib/server/gcs-loader";

const VECTOR_STATE_BLOB = "openai_vector_store_state.json";
const SEC_SPEECHES_LOCAL_FILE = "all_speeches_final.json";

export interface VectorStoreEntry {
  org_label: string;
  vector_store_id: string;
  docs: Record<string, unknown>;
  doc_count_indexed: number;
  updated_at: string;
}

export interface VectorStoreStatePayload {
  version: number;
  updated_at: string;
  stores: Record<string, VectorStoreEntry>;
}

function normalizeString(value: unknown): string {
  return String(value ?? "").trim();
}

function findProjectRootWithData(startDir: string): string {
  let current = path.resolve(startDir);
  for (let i = 0; i < 7; i += 1) {
    const candidate = path.join(current, "data", SEC_SPEECHES_LOCAL_FILE);
    if (fs.existsSync(candidate)) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      break;
    }
    current = parent;
  }
  return path.resolve(startDir);
}

function resolveDataDirPath(): string {
  const cfg = getDataSourceConfig();
  if (cfg.dataDirPath) {
    return path.isAbsolute(cfg.dataDirPath) ? cfg.dataDirPath : path.resolve(process.cwd(), cfg.dataDirPath);
  }
  const root = findProjectRootWithData(process.cwd());
  return path.join(root, "data");
}

function readLocalVectorState(): unknown | null {
  const filePath = path.join(resolveDataDirPath(), VECTOR_STATE_BLOB);
  if (!fs.existsSync(filePath)) {
    return null;
  }
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch {
    return null;
  }
}

function normalizeVectorState(raw: unknown): VectorStoreStatePayload {
  if (!raw || typeof raw !== "object") {
    return { version: 2, updated_at: "", stores: {} };
  }

  const payload = raw as Record<string, unknown>;
  const storesRaw = payload.stores;
  if (storesRaw && typeof storesRaw === "object" && !Array.isArray(storesRaw)) {
    const stores: Record<string, VectorStoreEntry> = {};
    for (const [orgKey, value] of Object.entries(storesRaw as Record<string, unknown>)) {
      const entry = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
      const vectorStoreId = normalizeString(entry.vector_store_id);
      if (!vectorStoreId) {
        continue;
      }
      stores[orgKey] = {
        org_label: normalizeString(entry.org_label) || orgKey.toUpperCase(),
        vector_store_id: vectorStoreId,
        docs: entry.docs && typeof entry.docs === "object" && !Array.isArray(entry.docs) ? (entry.docs as Record<string, unknown>) : {},
        doc_count_indexed: Number.parseInt(String(entry.doc_count_indexed ?? "0"), 10) || 0,
        updated_at: normalizeString(entry.updated_at)
      };
    }
    return {
      version: Number.parseInt(String(payload.version ?? "2"), 10) || 2,
      updated_at: normalizeString(payload.updated_at),
      stores
    };
  }

  const legacyVectorStoreId = normalizeString(payload.vector_store_id);
  if (!legacyVectorStoreId) {
    return { version: 2, updated_at: normalizeString(payload.updated_at), stores: {} };
  }

  return {
    version: 2,
    updated_at: normalizeString(payload.updated_at),
    stores: {
      sec: {
        org_label: "SEC",
        vector_store_id: legacyVectorStoreId,
        docs: payload.docs && typeof payload.docs === "object" && !Array.isArray(payload.docs) ? (payload.docs as Record<string, unknown>) : {},
        doc_count_indexed: Number.parseInt(String(payload.indexed_speeches ?? "0"), 10) || 0,
        updated_at: normalizeString(payload.updated_at)
      }
    }
  };
}

export async function loadVectorStoreState(): Promise<VectorStoreStatePayload> {
  const cfg = getDataSourceConfig();
  let raw: unknown | null = null;

  if (cfg.mode === "gcs" || cfg.mode === "auto") {
    raw = await downloadGcsJson<unknown>(VECTOR_STATE_BLOB);
  }
  if (raw === null && (cfg.mode === "local" || cfg.mode === "auto")) {
    raw = readLocalVectorState();
  }

  return normalizeVectorState(raw);
}

export function listActiveVectorStores(state: VectorStoreStatePayload): Array<{ org_key: string; org_label: string; vector_store_id: string }> {
  const rows: Array<{ org_key: string; org_label: string; vector_store_id: string }> = [];
  for (const [orgKey, value] of Object.entries(state.stores || {})) {
    const vectorStoreId = normalizeString(value?.vector_store_id);
    if (!vectorStoreId) {
      continue;
    }
    rows.push({
      org_key: orgKey,
      org_label: normalizeString(value?.org_label) || orgKey.toUpperCase(),
      vector_store_id: vectorStoreId
    });
  }
  return rows;
}
