export type TraderWatchProfile = {
  id: string;
  name: string;
  description: string;
  wallet: {
    chain: "solana";
    address: string;
    profileUrl: string;
  } | null;
};

// This registry follows wallet addresses. Display names identify the source
// profile; they do not assert who controls an address.
export const TRADER_WATCH_PROFILES: readonly TraderWatchProfile[] = [
  {
    id: "lbexplorer",
    name: "lbexplorer",
    description: "Pump.fun wallet profile",
    wallet: {
      chain: "solana",
      address: "64w4qRu9VGio7U1Asc6B68QDpS8L1McmSn2yyExC6Fii",
      profileUrl: "https://pump.fun/profile/64w4qRu9VGio7U1Asc6B68QDpS8L1McmSn2yyExC6Fii",
    },
  },
];

export function traderWatchProfile(id: string): TraderWatchProfile | undefined {
  return TRADER_WATCH_PROFILES.find((profile) => profile.id === id);
}

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

export type WalletHolding = {
  mint: string;
  amount: string;
  decimals: number;
  symbol: string | null;
  name: string | null;
  labelSource: "dexscreener" | null;
};

export type WalletHoldings = {
  status: "available" | "partial" | "unavailable";
  items: WalletHolding[];
  sol: string | null;
  observedAt: string | null;
  note: string | null;
};

export type TraderWatchData = {
  profile: TraderWatchProfile;
  walletActivity: { status: "available" | "unavailable"; items: WalletObservation[]; note: string | null };
  walletHoldings: WalletHoldings;
  generatedAt: string;
};

export type ParsedTokenAccount = {
  account?: { data?: { parsed?: { info?: {
    owner?: string;
    mint?: string;
    tokenAmount?: { amount?: string; decimals?: number };
  } } } };
};

export function walletHoldingsFromAccounts(address: string, accounts: ParsedTokenAccount[]): WalletHolding[] {
  const byMint = new Map<string, { amount: bigint; decimals: number }>();
  for (const account of accounts) {
    const info = account.account?.data?.parsed?.info;
    const raw = info?.tokenAmount?.amount;
    const decimals = info?.tokenAmount?.decimals;
    if (info?.owner !== address || !info.mint || !/^\d+$/.test(raw ?? "") || !Number.isInteger(decimals) || decimals! < 0 || decimals! > 18) continue;
    const current = byMint.get(info.mint);
    if (current && current.decimals !== decimals) continue;
    byMint.set(info.mint, { amount: (current?.amount ?? 0n) + BigInt(raw!), decimals: decimals! });
  }
  return [...byMint.entries()]
    .filter(([, value]) => value.amount > 0n)
    .map(([mint, value]) => ({
      mint,
      amount: formatTokenAmount(value.amount, value.decimals),
      decimals: value.decimals,
      symbol: null,
      name: null,
      labelSource: null,
    }))
    .sort((a, b) => a.mint.localeCompare(b.mint));
}

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
  return `${sign}${formatTokenAmount(absolute, decimals)}`;
}

export function formatTokenAmount(absolute: bigint, decimals: number): string {
  const divisor = 10n ** BigInt(decimals);
  const whole = absolute / divisor;
  const fractional = decimals ? (absolute % divisor).toString().padStart(decimals, "0").replace(/0+$/, "") : "";
  return `${whole}${fractional ? `.${fractional}` : ""}`;
}
