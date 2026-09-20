import { neon } from "@neondatabase/serverless";
import { coinConfig } from "../crypto-coins.ts";
import type { CryptoBriefSourcePost, CryptoBriefWindow } from "../crypto-brief-types.ts";

const WINDOW_MS: Record<CryptoBriefWindow, number> = { "24h": 86_400_000, "7d": 7 * 86_400_000, "30d": 30 * 86_400_000 };

export type CryptoBriefStoredData = {
  posts: CryptoBriefSourcePost[];
  archiveStatus: "available" | "partial" | "unavailable";
  lastCollectedAt: string | null;
  collectionComplete: boolean | null;
  coverageReasons: string[];
  contractAnchorAt: string | null;
};

export async function readCryptoBriefStoredData(coin: string, window: CryptoBriefWindow, asOf: string): Promise<CryptoBriefStoredData> {
  if (!process.env.DATABASE_URL) return { posts: [], archiveStatus: "unavailable", lastCollectedAt: null, collectionComplete: null, coverageReasons: ["The crypto archive is not connected in this environment."], contractAnchorAt: null };
  const sql = neon(process.env.DATABASE_URL);
  const end = new Date(asOf), start = new Date(end.getTime() - WINDOW_MS[window]);
  const config = coinConfig(coin);
  const exists = await sql`SELECT to_regclass('public.crypto_social_posts') AS posts,to_regclass('public.crypto_social_windows') AS windows`;
  if (!exists[0]?.posts || !exists[0]?.windows) return { posts: [], archiveStatus: "unavailable", lastCollectedAt: null, collectionComplete: null, coverageReasons: ["Collection has not started for this archive."], contractAnchorAt: null };
  const [posts, windows, anchor] = await Promise.all([
    sql`SELECT DISTINCT p.id::text,p.author_id::text,a.handle,p.text,p.url,p.posted_at,p.first_seen_at,p.kind
      FROM crypto_social_posts p JOIN crypto_social_accounts a ON a.id=p.author_id
      JOIN crypto_social_matches m ON m.post_id=p.id JOIN crypto_social_windows w ON w.id=m.window_id
      WHERE w.coin=${coin} AND p.posted_at>=${start.toISOString()}::timestamptz AND p.posted_at<${end.toISOString()}::timestamptz
      ORDER BY p.posted_at DESC,p.id DESC`,
    sql`SELECT count(*)::int AS windows,count(*) FILTER(WHERE pages>0)::int AS searched,
      count(*) FILTER(WHERE status='search_exhausted')::int AS exhausted,
      (SELECT max(r.requested_at) FROM crypto_social_requests r JOIN crypto_social_windows rw ON rw.id=r.window_id WHERE rw.coin=${coin}) AS last_collected
      FROM crypto_social_windows WHERE coin=${coin} AND start_at<${end.toISOString()}::timestamptz AND end_at>${start.toISOString()}::timestamptz AND query<>'timeline text match'`,
    config.address
      ? (config.address.startsWith("0x")
        ? sql`SELECT min(posted_at) AS at FROM crypto_social_posts WHERE position(${config.address.toLowerCase()} in lower(text))>0`
        : sql`SELECT min(posted_at) AS at FROM crypto_social_posts WHERE position(${config.address} in text)>0`)
      : Promise.resolve([]),
  ]);
  const summary = windows[0] ?? {};
  const windowCount = Number(summary.windows ?? 0), searched = Number(summary.searched ?? 0), exhausted = Number(summary.exhausted ?? 0);
  const reasons: string[] = [];
  let archiveStatus: CryptoBriefStoredData["archiveStatus"] = "available";
  if (!windowCount) { archiveStatus = "unavailable"; reasons.push("No collection windows exist for this asset and interval."); }
  else if (searched < windowCount) { archiveStatus = "partial"; reasons.push(`${searched} of ${windowCount} configured collection windows have saved results.`); }
  const collectionComplete = windowCount ? exhausted === windowCount : null;
  if (windowCount && !collectionComplete) reasons.push("The saved sample is not a complete view of X.");
  return {
    posts: posts.map((row) => ({ id: String(row.id), author_id: row.author_id == null ? null : String(row.author_id), handle: row.handle == null ? null : String(row.handle), text: String(row.text ?? ""), url: row.url == null ? null : String(row.url), posted_at: row.posted_at as string | Date | null, first_seen_at: row.first_seen_at as string | Date | null, kind: row.kind == null ? null : String(row.kind) })),
    archiveStatus, lastCollectedAt: summary.last_collected ? new Date(summary.last_collected as string).toISOString() : null,
    collectionComplete, coverageReasons: reasons, contractAnchorAt: anchor[0]?.at ? new Date(anchor[0].at as string).toISOString() : null,
  };
}
