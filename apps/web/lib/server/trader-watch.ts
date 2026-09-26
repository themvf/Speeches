import { getRecentArticles } from "@/lib/server/neon";
import { xTimelineFeedKey } from "@/lib/server/x-syndication";
import {
  type TraderPost,
  type TraderWatchProfile,
  type WalletObservation,
  walletTokenChanges,
} from "@/lib/trader-watch";

type RpcEnvelope<T> = { result?: T; error?: { message?: string } };
type SignatureRow = { signature: string; blockTime: number | null; err: unknown };
type TransactionRow = {
  blockTime?: number | null;
  meta?: {
    err?: unknown;
    preTokenBalances?: Parameters<typeof walletTokenChanges>[1];
    postTokenBalances?: Parameters<typeof walletTokenChanges>[2];
  } | null;
};

function rpcEndpoint(): string {
  const configured = process.env.SOLANA_RPC_URL?.trim();
  if (!configured) return "https://api.mainnet-beta.solana.com";
  const parsed = new URL(configured);
  if (parsed.protocol !== "https:") throw new Error("SOLANA_RPC_URL must be HTTPS");
  return parsed.toString();
}

async function rpc<T>(method: string, params: unknown[]): Promise<T> {
  const response = await fetch(rpcEndpoint(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }),
    cache: "no-store",
    signal: AbortSignal.timeout(8_000),
  });
  if (!response.ok) throw new Error(`Solana RPC returned ${response.status}`);
  const payload = await response.json() as RpcEnvelope<T>;
  if (payload.error) throw new Error(payload.error.message || "Solana RPC error");
  if (payload.result === undefined) throw new Error("Solana RPC result missing");
  return payload.result;
}

export async function readTraderPosts(profile: TraderWatchProfile): Promise<TraderPost[]> {
  if (!process.env.DATABASE_URL) throw new Error("Post archive database is not configured");
  const rows = await getRecentArticles({ feedKey: xTimelineFeedKey(profile.xHandle), limit: 20 });
  return rows.map((row) => ({
    id: row.guid,
    text: row.description || row.title,
    url: row.url,
    publishedAt: row.published_at ? new Date(row.published_at).toISOString() : null,
    observedAt: new Date(row.fetched_at).toISOString(),
  }));
}

export async function readWalletActivity(profile: TraderWatchProfile): Promise<WalletObservation[]> {
  const address = profile.wallet?.address;
  if (!address) return [];
  const signatures = await rpc<SignatureRow[]>("getSignaturesForAddress", [address, { limit: 8, commitment: "confirmed" }]);
  if (!Array.isArray(signatures)) throw new Error("Invalid Solana signature response");
  const observations = await Promise.all(signatures.map(async (row): Promise<WalletObservation> => {
    const base: WalletObservation = {
      signature: row.signature,
      timestamp: row.blockTime ? new Date(row.blockTime * 1000).toISOString() : null,
      status: row.err ? "failed" : "details_unavailable",
      tokenChanges: [],
      url: `https://explorer.solana.com/tx/${row.signature}`,
    };
    if (row.err) return base;
    try {
      const transaction = await rpc<TransactionRow | null>("getTransaction", [row.signature, {
        commitment: "confirmed",
        encoding: "jsonParsed",
        maxSupportedTransactionVersion: 0,
      }]);
      if (!transaction?.meta) return base;
      return {
        ...base,
        timestamp: transaction.blockTime ? new Date(transaction.blockTime * 1000).toISOString() : base.timestamp,
        status: transaction.meta.err ? "failed" : "confirmed",
        tokenChanges: transaction.meta.err ? [] : walletTokenChanges(
          address,
          transaction.meta.preTokenBalances,
          transaction.meta.postTokenBalances,
        ),
      };
    } catch {
      return base;
    }
  }));
  return observations;
}
