"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

export const DEFAULT_LIST_ID = "general";

export interface SavedItemMetadata {
  documentId?: string;
  organization?: string;
  sourceKind?: string;
  docType?: string;
  speaker?: string;
  date?: string;
  publishedAt?: string;
  wordCount?: number;
  keywords?: string[];
  topics?: string[];
  sentimentLabel?: "positive" | "negative" | "neutral" | "";
  sentimentScore?: number;
  feedKey?: string;
  author?: string;
  toneLabel?: "positive" | "negative" | "neutral" | null;
}

export interface SavedItem {
  id: string;
  type: "article" | "doc";
  title: string;
  url?: string;
  source: string;
  topic?: string;
  savedAt: string;
  listIds: string[];
  metadata?: SavedItemMetadata;
}

export interface SavedList {
  id: string;
  name: string;
  createdAt: string;
}

const ITEMS_STORAGE_KEY = "saved_items_v1";
const LISTS_STORAGE_KEY = "saved_lists_v1";
const DEFAULT_LIST: SavedList = {
  id: DEFAULT_LIST_ID,
  name: "General",
  createdAt: "1970-01-01T00:00:00.000Z",
};

function normalizeListName(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function createListId(name: string): string {
  const slug = normalizeListName(name)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 42);
  return `list:${slug || "saved"}:${Date.now().toString(36)}`;
}

function uniqueListIds(value: unknown): string[] {
  const ids = Array.isArray(value) ? value : [DEFAULT_LIST_ID];
  const clean = ids.map((id) => String(id || "").trim()).filter(Boolean);
  return Array.from(new Set(clean.length ? clean : [DEFAULT_LIST_ID]));
}

function normalizeItem(value: unknown): SavedItem | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const raw = value as Partial<SavedItem>;
  const id = String(raw.id || "").trim();
  const title = String(raw.title || "").trim();
  if (!id || !title) {
    return null;
  }

  return {
    id,
    type: raw.type === "article" ? "article" : "doc",
    title,
    url: raw.url ? String(raw.url) : "",
    source: String(raw.source || "Unknown"),
    topic: raw.topic ? String(raw.topic) : undefined,
    savedAt: String(raw.savedAt || new Date().toISOString()),
    listIds: uniqueListIds(raw.listIds),
    metadata: raw.metadata && typeof raw.metadata === "object" ? raw.metadata : undefined,
  };
}

function normalizeList(value: unknown): SavedList | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const raw = value as Partial<SavedList>;
  const id = String(raw.id || "").trim();
  const name = normalizeListName(String(raw.name || ""));
  if (!id || !name) {
    return null;
  }
  return {
    id,
    name,
    createdAt: String(raw.createdAt || new Date().toISOString()),
  };
}

function readItems(): SavedItem[] {
  try {
    const raw = JSON.parse(localStorage.getItem(ITEMS_STORAGE_KEY) ?? "[]") as unknown[];
    return (Array.isArray(raw) ? raw : []).map(normalizeItem).filter((item): item is SavedItem => Boolean(item));
  } catch {
    return [];
  }
}

function readLists(): SavedList[] {
  try {
    const raw = JSON.parse(localStorage.getItem(LISTS_STORAGE_KEY) ?? "[]") as unknown[];
    const lists = (Array.isArray(raw) ? raw : []).map(normalizeList).filter((item): item is SavedList => Boolean(item));
    const withDefault = lists.some((list) => list.id === DEFAULT_LIST_ID) ? lists : [DEFAULT_LIST, ...lists];
    return withDefault;
  } catch {
    return [DEFAULT_LIST];
  }
}

function writeItems(items: SavedItem[]): void {
  localStorage.setItem(ITEMS_STORAGE_KEY, JSON.stringify(items));
}

function writeLists(lists: SavedList[]): void {
  localStorage.setItem(LISTS_STORAGE_KEY, JSON.stringify(lists));
}

export function useSavedItems() {
  const [items, setItems] = useState<SavedItem[]>([]);
  const [lists, setLists] = useState<SavedList[]>([DEFAULT_LIST]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    setItems(readItems());
    setLists(readLists());
    setLoaded(true);

    const onStorage = (event: StorageEvent) => {
      if (event.key === ITEMS_STORAGE_KEY) {
        setItems(readItems());
      }
      if (event.key === LISTS_STORAGE_KEY) {
        setLists(readLists());
      }
    };

    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const listById = useMemo(() => new Map(lists.map((list) => [list.id, list])), [lists]);

  const isSaved = useCallback((id: string) => items.some((item) => item.id === id), [items]);

  const save = useCallback((item: Omit<SavedItem, "savedAt" | "listIds"> & { listIds?: string[] }, listId = DEFAULT_LIST_ID) => {
    setItems((prev) => {
      const targetListIds = uniqueListIds(item.listIds?.length ? item.listIds : [listId]);
      const existing = prev.find((saved) => saved.id === item.id);
      const next = existing
        ? prev.map((saved) =>
            saved.id === item.id
              ? {
                  ...saved,
                  ...item,
                  savedAt: saved.savedAt,
                  listIds: uniqueListIds([...saved.listIds, ...targetListIds]),
                  metadata: { ...saved.metadata, ...item.metadata },
                }
              : saved
          )
        : [...prev, { ...item, savedAt: new Date().toISOString(), listIds: targetListIds }];
      writeItems(next);
      return next;
    });
  }, []);

  const toggle = useCallback((item: Omit<SavedItem, "savedAt" | "listIds"> & { listIds?: string[] }, listId = DEFAULT_LIST_ID) => {
    setItems((prev) => {
      const exists = prev.some((saved) => saved.id === item.id);
      const next = exists
        ? prev.filter((saved) => saved.id !== item.id)
        : [...prev, { ...item, savedAt: new Date().toISOString(), listIds: uniqueListIds(item.listIds?.length ? item.listIds : [listId]) }];
      writeItems(next);
      return next;
    });
  }, []);

  const updateItem = useCallback((id: string, patch: Partial<Omit<SavedItem, "id" | "savedAt">>) => {
    setItems((prev) => {
      let changed = false;
      const next = prev.map((item) => {
        if (item.id !== id) {
          return item;
        }
        changed = true;
        return {
          ...item,
          ...patch,
          listIds: patch.listIds ? uniqueListIds(patch.listIds) : item.listIds,
          metadata: patch.metadata ? { ...item.metadata, ...patch.metadata } : item.metadata,
        };
      });
      if (changed) {
        writeItems(next);
      }
      return next;
    });
  }, []);

  const setItemLists = useCallback((id: string, listIds: string[]) => {
    updateItem(id, { listIds: uniqueListIds(listIds) });
  }, [updateItem]);

  const remove = useCallback((id: string) => {
    setItems((prev) => {
      const next = prev.filter((item) => item.id !== id);
      writeItems(next);
      return next;
    });
  }, []);

  const clear = useCallback(() => {
    setItems([]);
    writeItems([]);
  }, []);

  const createList = useCallback((name: string): SavedList | null => {
    const cleanName = normalizeListName(name);
    if (!cleanName) {
      return null;
    }

    const existing = lists.find((list) => list.name.toLowerCase() === cleanName.toLowerCase());
    if (existing) {
      return existing;
    }

    const nextList = { id: createListId(cleanName), name: cleanName, createdAt: new Date().toISOString() };
    setLists((prev) => {
      const next = [...prev, nextList];
      writeLists(next);
      return next;
    });
    return nextList;
  }, [lists]);

  const renameList = useCallback((id: string, name: string) => {
    if (id === DEFAULT_LIST_ID) {
      return;
    }
    const cleanName = normalizeListName(name);
    if (!cleanName) {
      return;
    }
    setLists((prev) => {
      const next = prev.map((list) => list.id === id ? { ...list, name: cleanName } : list);
      writeLists(next);
      return next;
    });
  }, []);

  const deleteList = useCallback((id: string) => {
    if (id === DEFAULT_LIST_ID) {
      return;
    }
    setLists((prev) => {
      const next = prev.filter((list) => list.id !== id);
      writeLists(next);
      return next;
    });
    setItems((prev) => {
      const next = prev.map((item) => {
        const nextListIds = item.listIds.filter((listId) => listId !== id);
        return { ...item, listIds: nextListIds.length ? nextListIds : [DEFAULT_LIST_ID] };
      });
      writeItems(next);
      return next;
    });
  }, []);

  return {
    items,
    lists,
    listById,
    loaded,
    isSaved,
    save,
    toggle,
    updateItem,
    setItemLists,
    remove,
    clear,
    createList,
    renameList,
    deleteList,
  };
}
