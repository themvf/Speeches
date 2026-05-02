"use client";

import { useCallback, useEffect, useState } from "react";

export interface SavedItem {
  id: string;
  type: "article" | "doc";
  title: string;
  url?: string;
  source: string;
  topic?: string;
  savedAt: string;
}

const STORAGE_KEY = "saved_items_v1";

function readStorage(): SavedItem[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]") as SavedItem[];
  } catch {
    return [];
  }
}

function writeStorage(items: SavedItem[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
}

export function useSavedItems() {
  const [items, setItems] = useState<SavedItem[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    setItems(readStorage());
    setLoaded(true);

    const onStorage = (event: StorageEvent) => {
      if (event.key === STORAGE_KEY) {
        setItems(readStorage());
      }
    };

    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const isSaved = useCallback((id: string) => items.some((item) => item.id === id), [items]);

  const toggle = useCallback((item: Omit<SavedItem, "savedAt">) => {
    setItems((prev) => {
      const exists = prev.some((s) => s.id === item.id);
      const next = exists
        ? prev.filter((s) => s.id !== item.id)
        : [...prev, { ...item, savedAt: new Date().toISOString() }];
      writeStorage(next);
      return next;
    });
  }, []);

  const remove = useCallback((id: string) => {
    setItems((prev) => {
      const next = prev.filter((s) => s.id !== id);
      writeStorage(next);
      return next;
    });
  }, []);

  const clear = useCallback(() => {
    setItems([]);
    writeStorage([]);
  }, []);

  return { items, loaded, isSaved, toggle, remove, clear };
}
