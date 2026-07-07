import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

function getArg(flag: string): string | null {
  const index = process.argv.indexOf(flag);
  if (index < 0) return null;
  return process.argv[index + 1] ?? null;
}

function inferNamespace(theme: string): string {
  const upper = theme.trim().toUpperCase();
  if (!upper) return "";
  if (!upper.includes("_")) return "CORE";
  return upper.split("_")[0] ?? "CORE";
}

const FINANCE_GROUP_PATTERNS: ReadonlyArray<{ group: string; terms: readonly string[] }> = [
  {
    group: "macro",
    terms: [
      "ECON",
      "ECONOMY",
      "GDP",
      "INFLATION",
      "CPI",
      "PRICES",
      "RATE",
      "RATES",
      "CENTRAL_BANK",
      "MONETARY",
      "FISCAL",
      "RECESSION",
      "SLOWDOWN",
      "GROWTH",
      "UNEMPLOYMENT",
      "WAGES",
      "DOLLAR",
      "CURRENCY",
      "FOREX",
      "FX"
    ]
  },
  {
    group: "banking_credit",
    terms: [
      "BANK",
      "BANKING",
      "CREDIT",
      "DEBT",
      "BOND",
      "LOAN",
      "YIELD",
      "LIQUIDITY",
      "FUNDING",
      "DEFAULT",
      "MORTGAGE",
      "REFINANCING",
      "TREASURY"
    ]
  },
  {
    group: "markets",
    terms: [
      "MARKET",
      "MARKETS",
      "STOCK",
      "EQUITY",
      "EQUITIES",
      "SECURITIES",
      "INDEX",
      "ETF",
      "EXCHANGE",
      "TRADER",
      "TRADERS",
      "VOLATILITY",
      "VIX",
      "OPTIONS",
      "FUTURES",
      "DERIVATIVES",
      "COMMODITY",
      "COMMODITIES",
      "GOLD",
      "OIL",
      "GAS"
    ]
  },
  {
    group: "capital_formation",
    terms: [
      "IPO",
      "PUBLIC_OFFERING",
      "EQUITY_OFFERING",
      "DEBT_OFFERING",
      "SECONDARY_OFFERING",
      "LISTING",
      "SPAC",
      "PRIVATE_EQUITY",
      "PRIVATE_CREDIT",
      "VENTURE_CAPITAL",
      "FUNDRAISING",
      "UNDERWRITER",
      "CAPITAL_MARKETS",
      "CAPITAL_FORMATION",
      "PLACEMENT",
      "MERGER",
      "ACQUISITION",
      "BUYOUT",
      "TAKEOVER"
    ]
  },
  {
    group: "regulation",
    terms: [
      "REGULATION",
      "REGULATORY",
      "SEC",
      "COMPLIANCE",
      "ENFORCEMENT",
      "AML",
      "BSA",
      "KYC",
      "FINCEN",
      "SANCTIONS",
      "OFAC",
      "ANTITRUST",
      "LITIGATION",
      "INVESTIGATION"
    ]
  },
  {
    group: "crypto",
    terms: [
      "CRYPTO",
      "CRYPTOCURRENCY",
      "BITCOIN",
      "BLOCKCHAIN",
      "TOKEN",
      "STABLECOIN",
      "DEFI"
    ]
  }
];

const FINANCE_NAMESPACE_HINTS = new Set([
  "ECON",
  "EPU"
]);

function normalizeTheme(theme: string): string {
  return theme.trim().toUpperCase();
}

function themeTokens(theme: string): string[] {
  return normalizeTheme(theme).split("_").filter(Boolean);
}

function containsTokenSequence(haystack: readonly string[], needle: readonly string[]): boolean {
  if (needle.length === 0 || needle.length > haystack.length) {
    return false;
  }

  for (let index = 0; index <= haystack.length - needle.length; index += 1) {
    if (needle.every((token, offset) => haystack[index + offset] === token)) {
      return true;
    }
  }

  return false;
}

function matchedFinanceSignals(theme: string, namespace: string): { finance: boolean; groups: string[]; reasons: string[] } {
  const normalizedTheme = normalizeTheme(theme);
  const tokens = themeTokens(theme);
  const groups = new Set<string>();
  const reasons = new Set<string>();

  if (FINANCE_NAMESPACE_HINTS.has(namespace)) {
    groups.add(namespace === "ECON" ? "macro" : "regulation");
    reasons.add(`namespace:${namespace}`);
  }

  if (normalizedTheme.startsWith("TAX_FNCACT")) {
    groups.add("markets");
    reasons.add("pattern:TAX_FNCACT");
  }

  for (const patternGroup of FINANCE_GROUP_PATTERNS) {
    const matchedTerms = patternGroup.terms.filter((term) => containsTokenSequence(tokens, themeTokens(term)));
    if (matchedTerms.length === 0) {
      continue;
    }

    groups.add(patternGroup.group);
    for (const term of matchedTerms) {
      reasons.add(`term:${term}`);
    }
  }

  return {
    finance: groups.size > 0,
    groups: [...groups],
    reasons: [...reasons]
  };
}

function escapeCsv(value: string): string {
  return `"${String(value ?? "").replace(/"/g, "\"\"")}"`;
}

function main() {
  const inputPath = resolve(getArg("--input") ?? "tmp/reviews/gdelt-theme-lookup.txt");
  const outputPath = resolve(getArg("--output") ?? "tmp/reviews/gdelt-theme-lookup-finance-marked.csv");
  const raw = readFileSync(inputPath, "utf8");

  const lines = raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const rows = lines
    .map((line) => {
      const [theme = "", count = ""] = line.split(/\t+/);
      const normalizedTheme = theme.trim();
      const namespace = inferNamespace(normalizedTheme);
      const finance = matchedFinanceSignals(normalizedTheme, namespace);
      return {
        theme: normalizedTheme,
        count: count.trim(),
        namespace,
        finance: finance.finance ? "Y" : "N",
        financeGroups: finance.groups.join(", "),
        financeReasons: finance.reasons.join(", ")
      };
    })
    .filter((row) => row.theme.length > 0);

  const csv = [
    "theme,count,namespace,finance_relevant,finance_groups,finance_reasons",
    ...rows.map((row) =>
      [
        escapeCsv(row.theme),
        escapeCsv(row.count),
        escapeCsv(row.namespace),
        escapeCsv(row.finance),
        escapeCsv(row.financeGroups),
        escapeCsv(row.financeReasons)
      ].join(",")
    )
  ].join("\n");

  writeFileSync(outputPath, csv, "utf8");
  console.log(`Wrote ${rows.length} themes to ${outputPath}`);
}

main();
