import { neon } from "@neondatabase/serverless";
import { DEFAULT_LIST_ID, type SavedItem, type SavedItemsPayload, type SavedList } from "@/lib/saved-items-types";

const DEFAULT_LIST: SavedList = {
  id: DEFAULT_LIST_ID,
  name: "General",
  createdAt: "1970-01-01T00:00:00.000Z",
};

function getSql() {
  const url = process.env.DATABASE_URL;
  if (!url) {
    throw new Error("DATABASE_URL is not configured.");
  }
  return neon(url);
}

function normalizeList(value: unknown): SavedList | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const src = value as Partial<SavedList>;
  const id = String(src.id || "").trim();
  const name = String(src.name || "").replace(/\s+/g, " ").trim();
  if (!id || !name) {
    return null;
  }
  return {
    id,
    name,
    createdAt: String(src.createdAt || new Date().toISOString()),
  };
}

function normalizeItem(value: unknown): SavedItem | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const src = value as Partial<SavedItem>;
  const id = String(src.id || "").trim();
  const title = String(src.title || "").trim();
  if (!id || !title) {
    return null;
  }
  const rawListIds = Array.isArray(src.listIds) ? src.listIds : [DEFAULT_LIST_ID];
  const listIds = Array.from(new Set(rawListIds.map((item) => String(item || "").trim()).filter(Boolean)));
  return {
    id,
    type: src.type === "article" ? "article" : "doc",
    title,
    url: src.url ? String(src.url) : "",
    source: String(src.source || "Unknown"),
    topic: src.topic ? String(src.topic) : undefined,
    savedAt: String(src.savedAt || new Date().toISOString()),
    listIds: listIds.length ? listIds : [DEFAULT_LIST_ID],
    metadata: src.metadata && typeof src.metadata === "object" ? src.metadata : undefined,
  };
}

export function normalizeSavedItemsPayload(value: unknown): SavedItemsPayload {
  const src = value && typeof value === "object" ? value as Partial<SavedItemsPayload> : {};
  const lists = Array.isArray(src.lists)
    ? src.lists.map(normalizeList).filter((item): item is SavedList => Boolean(item))
    : [];
  const items = Array.isArray(src.items)
    ? src.items.map(normalizeItem).filter((item): item is SavedItem => Boolean(item))
    : [];
  const hasDefaultList = lists.some((list) => list.id === DEFAULT_LIST_ID);
  return {
    items,
    lists: hasDefaultList ? lists : [DEFAULT_LIST, ...lists],
  };
}

export async function ensureSavedItemsSchema(): Promise<void> {
  const sql = getSql();
  await sql`
    CREATE TABLE IF NOT EXISTS user_saved_items (
      user_id    TEXT PRIMARY KEY,
      items      JSONB NOT NULL DEFAULT '[]'::jsonb,
      lists      JSONB NOT NULL DEFAULT '[]'::jsonb,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
  `;
}

export async function loadUserSavedItems(userId: string): Promise<SavedItemsPayload> {
  await ensureSavedItemsSchema();
  const sql = getSql();
  const rows = await sql`
    SELECT items, lists
    FROM user_saved_items
    WHERE user_id = ${userId}
    LIMIT 1
  ` as unknown as Array<{ items: unknown; lists: unknown }>;
  if (!rows[0]) {
    return { items: [], lists: [DEFAULT_LIST] };
  }
  return normalizeSavedItemsPayload({
    items: rows[0].items,
    lists: rows[0].lists,
  });
}

export async function saveUserSavedItems(userId: string, payload: SavedItemsPayload): Promise<SavedItemsPayload> {
  const normalized = normalizeSavedItemsPayload(payload);
  await ensureSavedItemsSchema();
  const sql = getSql();
  await sql`
    INSERT INTO user_saved_items (user_id, items, lists, updated_at)
    VALUES (${userId}, ${JSON.stringify(normalized.items)}::jsonb, ${JSON.stringify(normalized.lists)}::jsonb, now())
    ON CONFLICT (user_id) DO UPDATE
    SET items = EXCLUDED.items,
        lists = EXCLUDED.lists,
        updated_at = now()
  `;
  return normalized;
}
