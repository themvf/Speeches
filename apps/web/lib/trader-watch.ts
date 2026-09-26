export type TraderWatchProfile = {
  id: string;
  name: string;
  xHandle: string;
  description: string;
  wallet: {
    chain: "solana";
    address: string;
    attribution: "verified" | "unverified";
    evidenceUrl: string | null;
  } | null;
};

// Attribution is deliberately separate from the address. A matching profile
// name does not establish that the X account controls this wallet.
export const TRADER_WATCH_PROFILES: readonly TraderWatchProfile[] = [
  {
    id: "lbexplorer",
    name: "LB",
    xHandle: "lbexplorer",
    description: "$50K to $1M public trading challenge",
    wallet: {
      chain: "solana",
      address: "64w4qRu9VGio7U1Asc6B68QDpS8L1McmSn2yyExC6Fii",
      attribution: "unverified",
      evidenceUrl: null,
    },
  },
];

export function traderWatchProfile(id: string): TraderWatchProfile | undefined {
  return TRADER_WATCH_PROFILES.find((profile) => profile.id === id);
}

export type TraderPost = {
  id: string;
  text: string;
  url: string;
  publishedAt: string | null;
  observedAt: string;
};

export type WalletTokenChange = {
  mint: string;
  delta: string;
  decimals: number;
};

export type WalletObservation = {
  signature: string;
  timestamp: string | null;
  status: "confirmed" | "failed" | "details_unavailable";
  tokenChanges: WalletTokenChange[];
  url: string;
};

export type TraderWatchData = {
  profile: TraderWatchProfile;
  posts: { status: "available" | "unavailable"; items: TraderPost[]; note: string | null };
  walletActivity: { status: "available" | "unavailable"; items: WalletObservation[]; note: string | null };
  generatedAt: string;
};

type TokenBalance = {
  mint?: string;
  owner?: string;
  uiTokenAmount?: { amount?: string; decimals?: number };
};

export function walletTokenChanges(
  address: string,
  pre: TokenBalance[] | undefined,
  post: TokenBalance[] | undefined,
): WalletTokenChange[] {
  const balances = new Map<string, { before: bigint; after: bigint; decimals: number }>();
  for (const [side, rows] of [["before", pre], ["after", post]] as const) {
    for (const row of rows ?? []) {
      if (row.owner !== address || !row.mint || !/^\d+$/.test(row.uiTokenAmount?.amount ?? "")) continue;
      const decimals = row.uiTokenAmount?.decimals;
      if (!Number.isInteger(decimals) || decimals! < 0 || decimals! > 18) continue;
      const current = balances.get(row.mint) ?? { before: 0n, after: 0n, decimals: decimals! };
      if (current.decimals !== decimals) continue;
      current[side] += BigInt(row.uiTokenAmount!.amount!);
      balances.set(row.mint, current);
    }
  }
  return [...balances.entries()]
    .map(([mint, balance]) => ({ mint, delta: formatTokenDelta(balance.after - balance.before, balance.decimals), decimals: balance.decimals }))
    .filter((change) => change.delta !== "0")
    .sort((a, b) => a.mint.localeCompare(b.mint));
}

function formatTokenDelta(value: bigint, decimals: number): string {
  if (value === 0n) return "0";
  const sign = value < 0n ? "-" : "+";
  const absolute = value < 0n ? -value : value;
  const divisor = 10n ** BigInt(decimals);
  const whole = absolute / divisor;
  const fractional = decimals ? (absolute % divisor).toString().padStart(decimals, "0").replace(/0+$/, "") : "";
  return `${sign}${whole}${fractional ? `.${fractional}` : ""}`;
}
