import registry from "@/lib/generated/finra-member-firms.json";

export type FinraMemberFirmMatch = {
  name: string;
};

type RegistryFirm = {
  name: string;
  normalizedName: string;
};

type FirmAlias = {
  alias: string;
  firmName: string;
  score: number;
};

const MAX_MATCHES = 5;
const MIN_ALIAS_CHARS = 8;
const TRAILING_SUFFIX_RE = /\b(?:incorporated|inc|llc|l l c|corp|corporation|co|company|limited|ltd|lp|l p|llp|plc|pbc)\b$/;
const TRAILING_BUSINESS_RE = /\b(?:and co|securities|capital markets|wealth management|financial services|brokerage services|broker dealer|broker dealers|investments|investment services|distributors|advisors|advisor services|advisory services)\b$/;
const GENERIC_ALIAS_RE = /^(?:and partners|capital|global|partners|securities|investments|financial|markets|wealth|advisors|brokerage|group|strategic|institutional|management)$/;

function normalizeText(value: string): string {
  return String(value || "")
    .normalize("NFKC")
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenCount(value: string): number {
  return value ? value.split(" ").length : 0;
}

function stripTrailingPatterns(value: string): string {
  let out = value;
  let changed = true;
  while (changed) {
    const next = out
      .replace(TRAILING_SUFFIX_RE, "")
      .replace(TRAILING_BUSINESS_RE, "")
      .replace(/\s+/g, " ")
      .trim();
    changed = next !== out;
    out = next;
  }
  return out;
}

function compactInitialAlias(value: string): string {
  const compact = value.replace(/\b([a-z])\s+([a-z])\s+([a-z][a-z0-9]+)\b/g, "$1$2$3");
  return compact === value ? "" : compact;
}

function isUsableAlias(alias: string): boolean {
  if (!alias || alias.length < MIN_ALIAS_CHARS) return false;
  if (/^(?:and|the)\s+/.test(alias)) return false;
  if (GENERIC_ALIAS_RE.test(alias)) return false;
  const tokens = tokenCount(alias);
  if (tokens >= 2) return true;
  return alias.length >= 12 && !GENERIC_ALIAS_RE.test(alias);
}

function aliasesForFirm(firm: RegistryFirm): string[] {
  const normalized = normalizeText(firm.normalizedName || firm.name);
  const stripped = stripTrailingPatterns(normalized);
  const compact = compactInitialAlias(stripped || normalized);
  return Array.from(new Set([normalized, stripped, compact].filter(isUsableAlias)));
}

const FIRM_ALIASES: FirmAlias[] = (registry.firms as RegistryFirm[])
  .flatMap((firm) => aliasesForFirm(firm).map((alias) => ({
    alias,
    firmName: firm.name,
    score: tokenCount(alias) * 100 + alias.length,
  })))
  .sort((a, b) => b.score - a.score || a.firmName.localeCompare(b.firmName));

export function finraMemberFirmCount(): number {
  return Number(registry.count || (registry.firms as RegistryFirm[]).length || 0);
}

export function finraMemberFirmSourceUrl(): string {
  return String(registry.sourceUrl || "https://www.finra.org/about/entities-we-regulate/broker-dealer-firms-we-regulate");
}

export function findFinraMemberFirmMatches(input: {
  title?: string | null;
  description?: string | null;
  author?: string | null;
  url?: string | null;
}, maxMatches = MAX_MATCHES): FinraMemberFirmMatch[] {
  const haystack = ` ${normalizeText([
    input.title,
    input.description,
    input.author,
    input.url,
  ].filter(Boolean).join(" "))} `;
  if (haystack.trim().length < MIN_ALIAS_CHARS) return [];

  const seen = new Set<string>();
  const matches: FinraMemberFirmMatch[] = [];
  for (const item of FIRM_ALIASES) {
    if (seen.has(item.firmName)) continue;
    if (!haystack.includes(` ${item.alias} `)) continue;
    seen.add(item.firmName);
    matches.push({ name: item.firmName });
    if (matches.length >= maxMatches) break;
  }
  return matches;
}
