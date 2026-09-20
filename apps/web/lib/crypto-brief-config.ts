import { COINS, isCoin } from "./crypto-coins.ts";
import type { CryptoBriefWindow } from "./crypto-brief-types.ts";

export type CryptoBriefConfig = {
  coin: string;
  defaultWindow: CryptoBriefWindow;
  maxTakeaways: 1 | 2 | 3;
  categories: Array<"product" | "governance" | "listing" | "security" | "market_activity" | "community" | "other">;
  modules: { savedPosts: true; marketContext: boolean; accountContext: boolean; namedSignals: boolean };
};

const DEFAULT: Omit<CryptoBriefConfig, "coin"> = {
  defaultWindow: "24h",
  maxTakeaways: 3,
  categories: ["product", "governance", "listing", "security", "market_activity", "community", "other"],
  modules: { savedPosts: true, marketContext: true, accountContext: true, namedSignals: true },
};

// Overrides belong here only when a product capability differs. Every registry entry
// receives the default config, including entries added after this module ships.
const OVERRIDES: Partial<Record<string, Partial<Omit<CryptoBriefConfig, "coin">>>> = {};

export function cryptoBriefConfig(coin: string): CryptoBriefConfig {
  if (!isCoin(coin)) throw new Error(`Unknown coin ${coin}`);
  const override = OVERRIDES[coin] ?? {};
  return { ...DEFAULT, ...override, modules: { ...DEFAULT.modules, ...override.modules }, coin };
}

export const CRYPTO_BRIEF_COINS = COINS.map(({ symbol }) => cryptoBriefConfig(symbol));
