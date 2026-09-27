import {
  type TraderWatchProfile,
  type WalletHolding,
  type WalletHoldings,
  type WalletObservation,
  type ParsedTokenAccount,
  formatTokenAmount,
  labelWalletObservations,
  walletHoldingsFromAccounts,
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
type TokenAccountsResult = { context?: { slot?: number }; value?: ParsedTokenAccount[] };
type BalanceResult = { value?: number };
type DexPair = {
  chainId?: string;
  baseToken?: { address?: string; symbol?: string; name?: string };
  quoteToken?: { address?: string; symbol?: string; name?: string };
  liquidity?: { usd?: number };
};

const TOKEN_PROGRAMS = [
  "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
  "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",
] as const;

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

export async function tokenLabels(mints: string[]): Promise<Map<string, { symbol: string; name: string }>> {
  const labels = new Map<string, { symbol: string; name: string }>();
  const batches: string[][] = [];
  for (let index = 0; index < mints.length; index += 30) batches.push(mints.slice(index, index + 30));
  const responses = await Promise.allSettled(batches.map(async (batch) => {
    const response = await fetch(`https://api.dexscreener.com/tokens/v1/solana/${batch.join(",")}`, {
      next: { revalidate: 120 },
      signal: AbortSignal.timeout(8_000),
    });
    if (!response.ok) throw new Error(`DEX Screener returned ${response.status}`);
    const pairs = await response.json() as DexPair[];
    return Array.isArray(pairs) ? pairs : [];
  }));
  const best = new Map<string, { symbol: string; name: string; liquidity: number }>();
  const wanted = new Set(mints);
  for (const response of responses) {
    if (response.status !== "fulfilled") continue;
    for (const pair of response.value) {
      if (pair.chainId !== "solana") continue;
      for (const token of [pair.baseToken, pair.quoteToken]) {
        const mint = token?.address;
        if (!mint || !wanted.has(mint)) continue;
        const symbol = token.symbol?.trim().slice(0, 32);
        const name = token.name?.trim().slice(0, 80);
        if (!symbol || !name) continue;
        const liquidity = Number.isFinite(pair.liquidity?.usd) ? pair.liquidity!.usd! : 0;
        if (liquidity > (best.get(mint)?.liquidity ?? -1)) best.set(mint, { symbol, name, liquidity });
      }
    }
  }
  for (const [mint, label] of best) labels.set(mint, { symbol: label.symbol, name: label.name });
  return labels;
}

export async function labelWalletActivity(
  observations: WalletObservation[],
  holdings: WalletHolding[],
): Promise<WalletObservation[]> {
  const known = new Map(holdings.filter((item) => item.symbol && item.name).map((item) => [item.mint, { symbol: item.symbol!, name: item.name! }]));
  const missing = [...new Set(observations.flatMap((item) => item.tokenChanges.map((change) => change.mint)))].filter((mint) => !known.has(mint));
  const fetched = await tokenLabels(missing).catch(() => new Map<string, { symbol: string; name: string }>());
  for (const [mint, label] of fetched) known.set(mint, label);
  return labelWalletObservations(observations, known);
}

export async function readWalletHoldings(profile: TraderWatchProfile): Promise<WalletHoldings> {
  const address = profile.wallet?.address;
  if (!address) return { status: "unavailable", items: [], sol: null, observedAt: null, note: "No wallet address is configured." };
  const [legacy, token2022, balance] = await Promise.allSettled([
    rpc<TokenAccountsResult>("getTokenAccountsByOwner", [address, { programId: TOKEN_PROGRAMS[0] }, { encoding: "jsonParsed", commitment: "confirmed" }]),
    rpc<TokenAccountsResult>("getTokenAccountsByOwner", [address, { programId: TOKEN_PROGRAMS[1] }, { encoding: "jsonParsed", commitment: "confirmed" }]),
    rpc<BalanceResult>("getBalance", [address, { commitment: "confirmed" }]),
  ]);
  const tokenResults = [legacy, token2022].filter((result): result is PromiseFulfilledResult<TokenAccountsResult> => result.status === "fulfilled" && Array.isArray(result.value.value));
  if (!tokenResults.length) return { status: "unavailable", items: [], sol: null, observedAt: new Date().toISOString(), note: "Token balances could not be read from Solana RPC." };

  const accounts = tokenResults.flatMap((result) => result.value.value!);
  let items: WalletHolding[] = walletHoldingsFromAccounts(address, accounts);
  const labels = await tokenLabels(items.map((item) => item.mint)).catch(() => new Map<string, { symbol: string; name: string }>());
  items = items.map((item) => {
    const label = labels.get(item.mint);
    return label ? { ...item, ...label, labelSource: "dexscreener" as const } : item;
  });
  const lamports = balance.status === "fulfilled" ? balance.value.value : undefined;
  const sol = Number.isSafeInteger(lamports) && lamports! >= 0 ? formatTokenAmount(BigInt(lamports!), 9) : null;
  const partial = tokenResults.length < TOKEN_PROGRAMS.length || sol === null;
  return {
    status: partial ? "partial" : "available",
    items,
    sol,
    observedAt: new Date().toISOString(),
    note: partial ? "Some Solana balance reads failed; this list may omit holdings." : null,
  };
}
