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
  tokens: number;
  explicit: boolean;
};

type AffiliateAliasDefinition = {
  firmName: string;
  aliases: string[];
};

const MAX_MATCHES = 5;
const MIN_ALIAS_CHARS = 8;
const TRAILING_SUFFIX_RE = /\b(?:incorporated|inc|llc|l l c|corp|corporation|co|company|limited|ltd|lp|l p|llp|plc|pbc)\b$/;
const TRAILING_BUSINESS_RE = /\b(?:and co|securities|capital markets|wealth management|financial services|brokerage services|broker dealer|broker dealers|investments|investment services|distributors|advisors|advisor services|advisory services)\b$/;
const GENERIC_ALIAS_RE = /^(?:and partners|capital|global|partners|securities|investments|financial|markets|wealth|advisors|brokerage|group|strategic|institutional|management)$/;
const AFFILIATE_ALIAS_DEFINITIONS: AffiliateAliasDefinition[] = [
  {
    firmName: "ETORO USA SECURITIES INC.",
    aliases: ["eToro", "eToro USA", "eToro Securities"],
  },
  {
    firmName: "KRAKEN SECURITIES",
    aliases: ["Kraken", "Kraken Securities LLC", "Payward", "Payward Inc.", "Payward, Inc."],
  },
];

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

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
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

function dedupeAliases(aliases: FirmAlias[]): FirmAlias[] {
  const byKey = new Map<string, FirmAlias>();
  for (const alias of aliases) {
    const key = `${alias.firmName}:${alias.alias}`;
    const existing = byKey.get(key);
    if (!existing || alias.score > existing.score || (alias.explicit && !existing.explicit)) {
      byKey.set(key, alias);
    }
  }
  return [...byKey.values()];
}

const REGISTRY_FIRM_NAMES = new Set((registry.firms as RegistryFirm[]).map((firm) => firm.name));

const REGISTRY_ALIASES: FirmAlias[] = (registry.firms as RegistryFirm[])
  .flatMap((firm) => aliasesForFirm(firm).map((alias) => ({
    alias,
    firmName: firm.name,
    score: tokenCount(alias) * 100 + alias.length,
    tokens: tokenCount(alias),
    explicit: false,
  })));

const AFFILIATE_ALIASES: FirmAlias[] = AFFILIATE_ALIAS_DEFINITIONS
  .filter((definition) => REGISTRY_FIRM_NAMES.has(definition.firmName))
  .flatMap((definition) => definition.aliases.map((alias) => normalizeText(alias))
    .filter(Boolean)
    .map((alias) => ({
      alias,
      firmName: definition.firmName,
      score: 10_000 + tokenCount(alias) * 100 + alias.length,
      tokens: tokenCount(alias),
      explicit: true,
    })));

const FIRM_ALIASES: FirmAlias[] = dedupeAliases([...REGISTRY_ALIASES, ...AFFILIATE_ALIASES])
  .sort((a, b) => b.score - a.score || a.firmName.localeCompare(b.firmName));

function rawTextContainsEntityAlias(rawText: string, alias: string): boolean {
  const re = new RegExp(`(?:^|[^A-Za-z0-9])(${escapeRegex(alias)})(?=$|[^A-Za-z0-9])`, "gi");
  for (const match of rawText.matchAll(re)) {
    const candidate = match[1] || "";
    if (candidate === candidate.toUpperCase()) return true;
    if (/^[A-Z][A-Za-z0-9]*$/.test(candidate)) return true;
    if (/^[a-z]+[A-Z][A-Za-z0-9]*$/.test(candidate)) return true;
  }
  return false;
}

export function finraMemberFirmNewsSearchTerms(firmName: string): string[] {
  const aliases = AFFILIATE_ALIAS_DEFINITIONS
    .filter((definition) => definition.firmName === firmName)
    .flatMap((definition) => definition.aliases);
  return Array.from(new Set([firmName, ...aliases].map((term) => term.trim()).filter(Boolean)));
}

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
  const rawVisibleText = [
    input.title,
    input.description,
    input.author,
  ].filter(Boolean).join(" ");
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
    if (item.tokens === 1 && !rawTextContainsEntityAlias(rawVisibleText, item.alias)) continue;
    seen.add(item.firmName);
    matches.push({ name: item.firmName });
    if (matches.length >= maxMatches) break;
  }
  return matches;
}
