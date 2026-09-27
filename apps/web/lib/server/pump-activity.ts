import { get } from "node:https";
import {
  parsePumpActivityItem,
  type PumpAction,
  type PumpActivityPage,
  type TraderWatchProfile,
} from "@/lib/trader-watch";
import { tokenLabels } from "@/lib/server/trader-watch";

type PumpResponse = {
  transactions?: unknown;
  pagination?: { has_more?: unknown; next_cursor?: unknown };
};

const CURSOR_PATTERN = /^\d{9,12}#[1-9A-HJ-NP-Za-km-z]{60,100}$/;
const CACHE_MS = 120_000;
const MAX_RESPONSE_BYTES = 2_000_000;
const cache = new Map<string, { expiresAt: number; value: Promise<PumpActivityPage> }>();

export function validPumpCursor(cursor: string): boolean {
  return CURSOR_PATTERN.test(cursor);
}

function readJson(url: URL): Promise<PumpResponse> {
  return new Promise((resolve, reject) => {
    const request = get(url, { headers: { Accept: "application/json" }, timeout: 8_000 }, (response) => {
      if (response.statusCode !== 200) {
        response.resume();
        reject(new Error(`Pump.fun returned ${response.statusCode ?? "an unknown status"}`));
        return;
      }
      let size = 0;
      const chunks: Buffer[] = [];
      response.on("data", (chunk: Buffer) => {
        size += chunk.length;
        if (size > MAX_RESPONSE_BYTES) {
          request.destroy(new Error("Pump.fun response exceeded size limit"));
          return;
        }
        chunks.push(chunk);
      });
      response.on("end", () => {
        try { resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")) as PumpResponse); }
        catch { reject(new Error("Pump.fun returned invalid JSON")); }
      });
      response.on("error", reject);
    });
    request.on("timeout", () => request.destroy(new Error("Pump.fun timed out")));
    request.on("error", reject);
  });
}

export async function readPumpActivity(profile: TraderWatchProfile, action: PumpAction, cursor?: string): Promise<PumpActivityPage> {
  const address = profile.wallet?.address;
  if (!address || !/^[1-9A-HJ-NP-Za-km-z]{32,44}$/.test(address)) throw new Error("Invalid wallet address");
  if (cursor !== undefined && !validPumpCursor(cursor)) throw new Error("Invalid activity cursor");
  const url = new URL(`/transactions/${address}`, "https://profile-api.pump.fun");
  url.searchParams.set("dustFilter", "true");
  url.searchParams.set("includeEvm", "false");
  url.searchParams.set("transactionType", action);
  if (cursor) url.searchParams.set("cursor", cursor);
  const key = url.toString();
  const cached = cache.get(key);
  if (cached && cached.expiresAt > Date.now()) return cached.value;
  const value = readJson(url).then(async (body): Promise<PumpActivityPage> => {
    if (!Array.isArray(body.transactions) || !body.pagination) throw new Error("Invalid Pump.fun activity response");
    let items = body.transactions
      .map((row) => row && typeof row === "object" ? parsePumpActivityItem(row, action) : null)
      .filter((row): row is NonNullable<typeof row> => row !== null);
    const unlabeled = [...new Set(items.filter((item) => !item.name || !item.symbol).map((item) => item.mint))];
    if (unlabeled.length) {
      const labels = await tokenLabels(unlabeled).catch(() => new Map<string, { symbol: string; name: string }>());
      items = items.map((item) => {
        const label = labels.get(item.mint);
        return label ? { ...item, name: item.name ?? label.name, symbol: item.symbol ?? label.symbol, labelSource: item.labelSource ?? "dexscreener" as const } : item;
      });
    }
    const nextCursor = body.pagination.has_more === true && typeof body.pagination.next_cursor === "string" && validPumpCursor(body.pagination.next_cursor)
      ? body.pagination.next_cursor : null;
    return { status: "available", items, nextCursor, note: null };
  });
  cache.set(key, { expiresAt: Date.now() + CACHE_MS, value });
  value.catch(() => cache.delete(key));
  return value;
}

export function unavailablePumpActivity(): PumpActivityPage {
  return { status: "unavailable", items: [], nextCursor: null, note: "Pump.fun activity could not be loaded. Try again later or open the wallet profile." };
}
