import { COINS, coinConfig, hasContract, matchesCoin, type CoinConfig } from "./crypto-coins.ts";
import { cryptoBriefConfig } from "./crypto-brief-config.ts";
import type { BriefEvidence, BriefTakeaway, CryptoBrief, CryptoBriefMatchMethod, CryptoBriefSourcePost, CryptoBriefWindow } from "./crypto-brief-types.ts";

export const CRYPTO_BRIEF_SCHEMA = "crypto-brief-v1" as const;
export const CRYPTO_BRIEF_POLICY = "identity-v1" as const;
export const CRYPTO_BRIEF_EVIDENCE_LIMIT = 100;
const WINDOW_MS: Record<CryptoBriefWindow, number> = { "24h": 86_400_000, "7d": 7 * 86_400_000, "30d": 30 * 86_400_000 };

const iso = (value: string | Date | null | undefined) => {
  if (!value) return null;
  const date = value instanceof Date ? value : new Date(value);
  return Number.isFinite(date.getTime()) ? date.toISOString() : null;
};
const clean = (value: string) => value.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
const contentKey = (value: string) => clean(value).toLowerCase().replace(/https?:\/\/\S+/g, "").replace(/[^\p{L}\p{N}]+/gu, " ").trim();
const xId = (post: CryptoBriefSourcePost) => post.url?.match(/\/(?:status|statuses)\/(\d+)(?:\/|$|\?)/i)?.[1] ?? (/^\d+$/.test(post.id) ? post.id : null);
const canonicalId = (post: CryptoBriefSourcePost) => xId(post) ? `x:${xId(post)}` : `crypto_archive:${post.id}`;
const safeKind = (kind: string | null): BriefEvidence["kind"] => ["original", "reply", "quote", "repost"].includes(kind ?? "") ? kind as BriefEvidence["kind"] : "unknown";

export function resolveCryptoBriefCoin(input: string): { status: "resolved"; coin: CoinConfig } | { status: "ambiguous" | "untracked"; matches: CoinConfig[] } {
  const query = input.trim();
  if (!query) return { status: "untracked", matches: [] };
  const upper = query.replace(/^[$#]/, "").toUpperCase();
  try { return { status: "resolved", coin: coinConfig(upper) }; } catch {}
  const lower = query.toLowerCase();
  const matches = COINS.filter((coin) => coin.name.toLowerCase() === lower || coin.label.toLowerCase() === lower || (coin.address && coin.address === query));
  return matches.length === 1 ? { status: "resolved", coin: matches[0] } : { status: matches.length > 1 ? "ambiguous" : "untracked", matches };
}

function matchMethod(post: CryptoBriefSourcePost, coin: CoinConfig): { method: CryptoBriefMatchMethod; provisional: boolean; reason: string } | null {
  if (hasContract(post.text, coin.symbol)) return { method: "contract", provisional: false, reason: `Exact ${coin.networkLabel} contract address` };
  if (!matchesCoin(post.text, coin.symbol, "words")) return null;
  const escaped = coin.symbol.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  if (new RegExp(`[$#]${escaped}(?![\\p{L}\\p{N}_])`, "iu").test(post.text)) return { method: "cashtag", provisional: !!coin.address, reason: `Configured ${coin.symbol} cashtag` };
  if (post.handle && coin.official.includes(post.handle.toLowerCase())) return { method: "declared_account", provisional: !!coin.address, reason: `Configured project account with matching post text` };
  return { method: "name_context", provisional: !!coin.address, reason: `Configured name and context match` };
}

function excerpt(value: string, max = 260): string {
  const text = clean(value);
  if (text.length <= max) return text;
  const slice = text.slice(0, max - 1);
  const stop = Math.max(slice.lastIndexOf(". "), slice.lastIndexOf("! "), slice.lastIndexOf("? "));
  return `${slice.slice(0, stop >= 80 ? stop + 1 : Math.max(80, slice.lastIndexOf(" "))).trim()}…`;
}

function category(text: string): string {
  if (/\b(hack|exploit|security|scam|phish|drain)\b/i.test(text)) return "security";
  if (/\b(list|listing|exchange|pair|market)\b/i.test(text)) return "listing";
  if (/\b(govern|proposal|vote|dao)\b/i.test(text)) return "governance";
  if (/\b(launch|release|ship|upgrade|product|app|mainnet|testnet)\b/i.test(text)) return "product";
  if (/\b(price|volume|liquidity|supply|burn|holder)\b/i.test(text)) return "market_activity";
  return "other";
}

export function buildCryptoBrief(args: {
  coin: string;
  window: CryptoBriefWindow;
  asOf: string;
  posts: readonly CryptoBriefSourcePost[];
  archiveStatus: "available" | "partial" | "unavailable";
  lastCollectedAt?: string | null;
  collectionComplete?: boolean | null;
  coverageReasons?: string[];
  contractAnchorAt?: string | null;
}): CryptoBrief {
  const coin = coinConfig(args.coin);
  const config = cryptoBriefConfig(args.coin);
  const endMs = Date.parse(args.asOf);
  const startMs = endMs - WINDOW_MS[args.window];
  if (!Number.isFinite(endMs)) throw new Error("Invalid asOf");
  const anchorMs = args.contractAnchorAt ? Date.parse(args.contractAnchorAt) : null;
  let excludedUndatedCount = 0;
  const matched: BriefEvidence[] = [];
  for (const post of args.posts) {
    const publishedAt = iso(post.posted_at);
    if (!publishedAt) { excludedUndatedCount++; continue; }
    const time = Date.parse(publishedAt);
    if (time < startMs || time >= endMs) continue;
    const match = matchMethod(post, coin);
    if (!match) continue;
    if (coin.address && match.method !== "contract" && (!anchorMs || time < anchorMs)) continue;
    matched.push({
      id: canonicalId(post), source: "crypto_archive", sourceRecordIds: [String(post.id)], authorId: post.author_id,
      handle: post.handle, text: clean(post.text), url: post.url, publishedAt, observedAt: iso(post.first_seen_at), kind: safeKind(post.kind),
      match: { coin: coin.symbol, ...match },
    });
  }
  matched.sort((a, b) => (b.publishedAt ?? "").localeCompare(a.publishedAt ?? "") || a.id.localeCompare(b.id));
  const unique = new Map<string, BriefEvidence>();
  for (const evidence of matched) {
    const found = unique.get(evidence.id);
    if (found) found.sourceRecordIds.push(...evidence.sourceRecordIds.filter((id) => !found.sourceRecordIds.includes(id)));
    else unique.set(evidence.id, evidence);
  }
  const all = [...unique.values()];
  const evidence = all.slice(0, CRYPTO_BRIEF_EVIDENCE_LIMIT);
  const originals = evidence.filter((post) => post.kind !== "repost");
  const selected: BriefEvidence[] = [];
  const usedContent = new Set<string>(), usedAuthors = new Set<string>();
  for (const pass of [true, false]) for (const post of originals) {
    const key = contentKey(post.text), author = post.authorId ?? post.handle ?? post.id;
    if (!key || usedContent.has(key) || (pass && usedAuthors.has(author))) continue;
    selected.push(post); usedContent.add(key); usedAuthors.add(author);
    if (selected.length === config.maxTakeaways) break;
  }
  const takeaways: BriefTakeaway[] = selected.slice(0, config.maxTakeaways).map((post, index) => ({
    id: `takeaway:${index + 1}:${post.id}`, category: category(post.text), text: excerpt(post.text), basis: "post_excerpt",
    whyItMatters: null, watchNext: null, evidenceIds: [post.id], ruleId: null,
    limitations: post.match.provisional ? ["Identity match is provisional; inspect the match reason and source post."] : [],
  }));
  const authors = new Set(evidence.map((post) => post.authorId ?? post.handle).filter(Boolean));
  const reasons = [...(args.coverageReasons ?? [])];
  if (all.length > CRYPTO_BRIEF_EVIDENCE_LIMIT) reasons.push(`Only the newest ${CRYPTO_BRIEF_EVIDENCE_LIMIT} matching posts are included.`);
  if (args.archiveStatus !== "available" && !reasons.length) reasons.push("The saved archive is not fully available.");
  const coverage = {
    status: args.archiveStatus, windowStart: new Date(startMs).toISOString(), windowEnd: new Date(endMs).toISOString(),
    latestPublishedAt: evidence[0]?.publishedAt ?? null, lastCollectedAt: iso(args.lastCollectedAt), matchingPostCount: all.length,
    includedPostCount: evidence.length, includedAuthorCount: authors.size, evidenceLimit: CRYPTO_BRIEF_EVIDENCE_LIMIT,
    truncated: all.length > CRYPTO_BRIEF_EVIDENCE_LIMIT, excludedUndatedCount, collectionComplete: args.collectionComplete ?? null,
    comparableBaseline: false, reasons,
  } as const;
  const windowLabel = args.window === "24h" ? "24 hours" : args.window === "7d" ? "7 days" : "30 days";
  const lead = takeaways[0]?.text ?? "No clear development in the saved posts.";
  const overview = `${coin.name} — ${windowLabel}. ${lead} Based on ${evidence.length} included saved post${evidence.length === 1 ? "" : "s"} from ${authors.size} account${authors.size === 1 ? "" : "s"}.`;
  return { schemaVersion: CRYPTO_BRIEF_SCHEMA, asset: { symbol: coin.symbol, name: coin.name, network: coin.network, networkLabel: coin.networkLabel, address: coin.address, identityNote: coin.identityNote }, window: args.window, asOf: new Date(endMs).toISOString(), coverage, overview, takeaways, evidence };
}
