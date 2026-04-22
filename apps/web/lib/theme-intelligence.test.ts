import assert from "node:assert/strict";
import test from "node:test";

import { INTELLIGENCE_PROFILES } from "./intelligence-seed.ts";
import { buildGdeltDocQueries, buildGdeltDocQuery, mapGdeltDocArticlesToEvidence } from "./server/gdelt-doc.ts";
import { buildGdeltGkgArchiveUrls, parseGdeltGkgCsv, parseGdeltGkgManifest } from "./server/gdelt-gkg.ts";
import { PRODUCT_CATEGORY_ORDER, focusAreasForProductCategory, parseRawThemes, scoreThemeArticle, themesForProductCategory } from "./theme-intelligence.ts";

test("maps raw GDELT themes to deduplicated normalized themes", () => {
  const signal = scoreThemeArticle({
    raw_themes: ["ECON_INFLATION", "latest CPI prices", "OIL"]
  });

  assert.deepEqual(signal.normalized_themes, ["ENERGY", "INFLATION"]);
  assert.deepEqual(signal.full_theme_list, ["ECON_INFLATION", "latest CPI prices", "OIL"]);
  assert.deepEqual(signal.normalized_theme_list, ["ENERGY", "INFLATION"]);
  assert.deepEqual(signal.theme_weights, { ENERGY: 10, INFLATION: 10 });
  assert.equal(signal.total_score, 5.2);
  assert.equal(signal.severity, "NORMAL");
});

test("detects multiple themes from one raw theme string", () => {
  const signal = scoreThemeArticle({
    raw_themes: "TRADE_SANCTIONS; central_bank policy; ECON_CREDIT; BOND_MARKET; RESTRICTIONS"
  });

  assert.deepEqual(signal.normalized_themes, ["CENTRAL_BANK", "CREDIT_MARKETS", "SANCTIONS", "TRADE", "REGULATION"]);
  assert.deepEqual(signal.normalized_theme_list, ["CENTRAL_BANK", "CREDIT_MARKETS", "SANCTIONS", "TRADE", "REGULATION"]);
  assert.equal(signal.matches.find((match) => match.normalized_theme === "CREDIT_MARKETS")?.matched_raw_themes.length, 2);
  assert.equal(signal.matches.find((match) => match.normalized_theme === "SANCTIONS")?.matched_raw_themes.length, 2);
});

test("scores severity thresholds accurately", () => {
  const normal = scoreThemeArticle({ raw_themes: ["CRYPTOCURRENCY"] });
  const high = scoreThemeArticle({
    raw_themes: ["ECON_BANKING", "LIQUIDITY"],
    context: {
      theme_mentions: {
        BANKING: { current_mentions: 20, baseline_mentions: 20 },
        LIQUIDITY: { current_mentions: 15, baseline_mentions: 15 }
      }
    }
  });
  const critical = scoreThemeArticle({
    raw_themes: ["BANK_FAILURE", "ECON_CREDIT", "OIL"],
    context: {
      theme_mentions: {
        BANKING: { current_mentions: 40, baseline_mentions: 20 },
        CREDIT_MARKETS: { current_mentions: 35, baseline_mentions: 15 },
        ENERGY: { current_mentions: 30, baseline_mentions: 10 }
      }
    }
  });

  assert.equal(normal.total_score, 1.8);
  assert.equal(normal.severity, "NORMAL");
  assert.equal(high.severity, "HIGH");
  assert.equal(critical.severity, "CRITICAL");
});

test("parses semicolon, comma, pipe, and newline theme inputs", () => {
  assert.deepEqual(parseRawThemes("GDP; wages, BITCOIN|OIL\nECB"), ["GDP", "wages", "BITCOIN", "OIL", "ECB"]);
});

test("keeps short acronym matching token-aware to avoid supply-chain false positives", () => {
  const signal = scoreThemeArticle({ raw_themes: ["SUPPLY_CHAIN"] });

  assert.deepEqual(signal.normalized_themes, ["SUPPLY_CHAIN"]);
  assert.equal(signal.theme_weights.TECHNOLOGY, undefined);
});

test("adds relative trend context and primary driver", () => {
  const signal = scoreThemeArticle({
    raw_themes: "ECON_INFLATION; OIL; CENTRAL_BANK",
    context: {
      window_label: "last 2 hours",
      previous_score: 16,
      baseline_score: 14,
      confidence: 85,
      theme_mentions: {
        INFLATION: { current_mentions: 120, baseline_mentions: 40, previous_mentions: 55 },
        ENERGY: { current_mentions: 70, baseline_mentions: 28, previous_mentions: 32 },
        CENTRAL_BANK: { current_mentions: 60, baseline_mentions: 30, previous_mentions: 40 }
      }
    }
  });

  assert.equal(signal.total_score, 286);
  assert.equal(signal.trend.delta, 270);
  assert.equal(signal.trend.direction, "ACCELERATING");
  assert.equal(signal.primary_driver?.normalized_theme, "INFLATION");
  assert.equal(signal.primary_driver?.spike_pct, 200);
  assert.equal(signal.primary_driver?.contribution_pct, 45.5);
  assert.deepEqual(signal.primary_drivers.map((driver) => driver.normalized_theme), ["INFLATION", "ENERGY", "CENTRAL_BANK"]);
  assert.equal(signal.secondary_drivers.length, 0);
  assert.equal(signal.signal_model.confidence_score, 85);
});

test("creates interpretation and market impacts for inflation plus energy", () => {
  const signal = scoreThemeArticle({
    raw_themes: "ECON_INFLATION; OIL",
    context: {
      previous_score: 8,
      theme_mentions: {
        INFLATION: { current_mentions: 60, baseline_mentions: 20 },
        ENERGY: { current_mentions: 45, baseline_mentions: 15 }
      }
    }
  });

  assert.equal(signal.interpretation.headline, "Inflation Shock");
  assert.ok(signal.combination_signals.some((combo) => combo.id === "inflation_energy_shock"));
  assert.ok(signal.market_impacts.some((impact) => impact.asset === "XLE" && impact.direction === "UP"));
  assert.ok(signal.market_impacts.some((impact) => impact.asset === "QQQ" && impact.direction === "DOWN"));
});

test("keeps AI independent from technology and ranks all contributing themes", () => {
  const signal = scoreThemeArticle({
    raw_themes: "GENERATIVE_AI; AI; SEMICONDUCTOR; SOFTWARE; CRYPTOCURRENCY; BITCOIN; EARNINGS; LAYOFFS; SEC; FUNDING_STRESS",
    context: {
      theme_mentions: {
        AI: { current_mentions: 74, baseline_mentions: 26 },
        TECHNOLOGY: { current_mentions: 58, baseline_mentions: 43 },
        CRYPTO: { current_mentions: 37, baseline_mentions: 21 },
        CORPORATE_ACTIVITY: { current_mentions: 29, baseline_mentions: 24 },
        REGULATION: { current_mentions: 18, baseline_mentions: 15 },
        LIQUIDITY: { current_mentions: 14, baseline_mentions: 10 }
      }
    }
  });

  assert.ok(signal.normalized_theme_list.includes("AI"));
  assert.ok(signal.normalized_theme_list.includes("TECHNOLOGY"));
  assert.notEqual(signal.matches.find((match) => match.normalized_theme === "AI")?.matched_raw_themes.length, 0);
  assert.notEqual(signal.matches.find((match) => match.normalized_theme === "TECHNOLOGY")?.matched_raw_themes.length, 0);
  assert.equal(signal.frequency_signals.length, 6);
  assert.deepEqual(signal.primary_drivers.map((driver) => driver.normalized_theme), ["AI", "CRYPTO", "TECHNOLOGY"]);
  assert.deepEqual(signal.secondary_drivers.map((driver) => driver.normalized_theme), ["LIQUIDITY", "CORPORATE_ACTIVITY", "REGULATION"]);
  assert.equal(Math.round(signal.frequency_signals.reduce((sum, driver) => sum + driver.contribution_pct, 0)), 100);
  assert.ok(signal.market_impacts.some((impact) => impact.themes.includes("AI")));
});

test("rolls normalized themes into the visible product taxonomy", () => {
  const signal = scoreThemeArticle({
    raw_themes: "SEC; COMPLIANCE; IPO; CAPITAL_RAISE; AML; KYC; SANCTIONS; ENFORCEMENT; FRAUD; AI; SEMICONDUCTOR; BITCOIN; ECON_CREDIT; STOCK_MARKET; GDP",
    context: {
      theme_mentions: {
        REGULATION: { current_mentions: 18, baseline_mentions: 12 },
        CORPORATE_ACTIVITY: { current_mentions: 12, baseline_mentions: 9 },
        SANCTIONS: { current_mentions: 14, baseline_mentions: 8 },
        AI: { current_mentions: 20, baseline_mentions: 10 },
        TECHNOLOGY: { current_mentions: 16, baseline_mentions: 10 },
        CRYPTO: { current_mentions: 15, baseline_mentions: 9 },
        CREDIT_MARKETS: { current_mentions: 17, baseline_mentions: 11 },
        FINANCIAL_MARKETS: { current_mentions: 13, baseline_mentions: 10 },
        ECONOMIC_GROWTH: { current_mentions: 10, baseline_mentions: 8 }
      }
    }
  });

  const categories = signal.product_categories.map((item) => item.category);
  assert.deepEqual(PRODUCT_CATEGORY_ORDER, [
    "SECURITIES_REGULATION",
    "CAPITAL_FORMATION",
    "AML",
    "ENFORCEMENT",
    "AI_TECH",
    "CRYPTO",
    "CREDIT_MARKETS",
    "FINANCIAL_MARKETS",
    "ECONOMIC_GROWTH"
  ]);
  assert.ok(categories.includes("SECURITIES_REGULATION"));
  assert.ok(categories.includes("CAPITAL_FORMATION"));
  assert.ok(categories.includes("AML"));
  assert.ok(categories.includes("ENFORCEMENT"));
  assert.ok(categories.includes("AI_TECH"));
  assert.ok(categories.includes("CRYPTO"));
  assert.ok(categories.includes("CREDIT_MARKETS"));
  assert.ok(categories.includes("FINANCIAL_MARKETS"));
  assert.ok(categories.includes("ECONOMIC_GROWTH"));
  assert.ok(signal.product_categories.every((category) => category.subcategories.length > 0));
  assert.ok(signal.primary_product_category);
  assert.equal(Math.round(signal.product_categories.reduce((sum, category) => sum + category.contribution_pct, 0)), 100);
});

test("keeps product category theme filters distinct", () => {
  assert.deepEqual(themesForProductCategory("AI_TECH"), ["AI", "TECHNOLOGY"]);
  assert.deepEqual(themesForProductCategory("AML"), ["SANCTIONS"]);
  assert.ok(themesForProductCategory("CRYPTO").includes("CRYPTO"));
  assert.ok(themesForProductCategory("CREDIT_MARKETS").includes("INTEREST_RATES"));
  assert.ok(themesForProductCategory("ECONOMIC_GROWTH").includes("INFLATION"));
});

test("keeps AML focus areas narrow and avoids broad crypto or regulation chips", () => {
  const focusAreas = focusAreasForProductCategory("AML");

  assert.deepEqual(focusAreas.map((item) => item.label), ["Sanctions", "AML / BSA", "KYC / Ownership", "Illicit Finance"]);
  assert.deepEqual(focusAreas.find((item) => item.id === "aml_sanctions")?.raw_patterns, ["SANCTIONS", "OFAC"]);
  assert.deepEqual(focusAreas.flatMap((item) => item.raw_patterns), [
    "SANCTIONS",
    "OFAC",
    "AML",
    "BSA",
    "MONEY_LAUNDERING",
    "ANTI_MONEY_LAUNDERING",
    "KYC",
    "CIP",
    "CUSTOMER_IDENTIFICATION",
    "BENEFICIAL_OWNERSHIP",
    "ILLICIT_FINANCE",
    "SUSPICIOUS_ACTIVITY",
    "SAR",
    "TERRORIST_FINANCING"
  ]);
  assert.equal(themesForProductCategory("AML").includes("CRYPTO"), false);
  assert.equal(themesForProductCategory("AML").includes("REGULATION"), false);
});

test("does not feed AML from embargo, restrictions, generic regulation, or generic crypto", () => {
  const broad = scoreThemeArticle({
    raw_themes: "EMBARGO; RESTRICTIONS; REGULATION; CRYPTOCURRENCY; BITCOIN"
  });
  const explicit = scoreThemeArticle({
    raw_themes: "SANCTIONS; OFAC; MONEY_LAUNDERING; KYC; BENEFICIAL_OWNERSHIP; SAR"
  });

  assert.equal(broad.product_categories.some((category) => category.category === "AML"), false);
  assert.equal(explicit.product_categories.some((category) => category.category === "AML"), true);
});

test("seed intelligence profiles produce complete API-backed signal models", () => {
  const profiles = INTELLIGENCE_PROFILES.map((profile) => ({
    ...profile,
    signal: scoreThemeArticle({
      id: profile.id,
      title: profile.label,
      raw_themes: profile.rawThemes,
      context: profile.context
    })
  }));
  const modern = profiles.find((profile) => profile.id === "modern");

  assert.equal(profiles.length, 4);
  assert.ok(profiles.every((profile) => profile.evidence.length > 0));
  assert.ok(profiles.every((profile) => profile.clusters.length > 0));
  assert.ok(profiles.every((profile) => profile.signal.frequency_signals.length > 0));
  assert.ok(profiles.every((profile) => profile.signal.product_categories.length > 0));
  assert.ok(profiles.every((profile) => profile.signal.primary_product_category));
  assert.ok(profiles.every((profile) => profile.evidence.every((article) => !article.url || !article.url.includes("/search"))));
  assert.deepEqual(modern?.signal.primary_drivers.map((driver) => driver.normalized_theme), ["AI", "CRYPTO", "TECHNOLOGY"]);
  assert.deepEqual(modern?.signal.secondary_drivers.map((driver) => driver.normalized_theme), ["LIQUIDITY", "CORPORATE_ACTIVITY", "REGULATION"]);
});

test("maps GDELT DOC articles to evidence with real URLs", () => {
  const baseProfile = INTELLIGENCE_PROFILES[0];
  const profile = {
    ...baseProfile,
    signal: scoreThemeArticle({
      id: baseProfile.id,
      title: baseProfile.label,
      raw_themes: baseProfile.rawThemes,
      context: baseProfile.context
    })
  };

  const query = buildGdeltDocQuery(profile);
  const queries = buildGdeltDocQueries(profile);
  const evidence = mapGdeltDocArticlesToEvidence(profile, [
    {
      url: "https://www.reuters.com/markets/commodities/oil-prices-rise-2026-04-21/",
      title: "Oil prices rise as inflation concerns intensify",
      seendate: "20260421T183000Z",
      domain: "reuters.com",
      language: "English",
      sourcecountry: "United States"
    },
    {
      url: "https://www.reuters.com/markets/commodities/oil-prices-rise-2026-04-21/",
      title: "Duplicate URL should be removed",
      seendate: "20260421T183000Z",
      domain: "reuters.com"
    },
    {
      url: "not-a-url",
      title: "Invalid URL should be removed"
    }
  ]);

  assert.equal(query, "oil inflation");
  assert.deepEqual(queries, ["oil inflation", "energy inflation", "central bank inflation"]);
  assert.equal(evidence.length, 1);
  assert.equal(evidence[0].url, "https://www.reuters.com/markets/commodities/oil-prices-rise-2026-04-21/");
  assert.equal(evidence[0].source, "reuters.com");
  assert.ok(evidence[0].relatedThemes.includes("INFLATION"));
  assert.ok(evidence[0].relatedThemes.includes("ENERGY"));
});

test("parses GDELT GKG manifest and maps live rows through normalized themes", () => {
  const manifest = [
    "97681 hash http://data.gdeltproject.org/gdeltv2/20260421211500.export.CSV.zip",
    "171911 hash http://data.gdeltproject.org/gdeltv2/20260421211500.mentions.CSV.zip",
    "7716254 hash http://data.gdeltproject.org/gdeltv2/20260421211500.gkg.csv.zip"
  ].join("\n");

  const latestArchive = parseGdeltGkgManifest(manifest);
  assert.equal(latestArchive, "http://data.gdeltproject.org/gdeltv2/20260421211500.gkg.csv.zip");
  assert.deepEqual(buildGdeltGkgArchiveUrls(latestArchive!, 2), [
    "http://data.gdeltproject.org/gdeltv2/20260421211500.gkg.csv.zip",
    "http://data.gdeltproject.org/gdeltv2/20260421210000.gkg.csv.zip"
  ]);

  const rows = [
    [
      "20260421211500-1",
      "20260421211500",
      "1",
      "reuters.com",
      "https://www.reuters.com/markets/commodities/oil-prices-rise-2026-04-21/",
      "",
      "",
      "ECON_INFLATION;OIL;CENTRAL_BANK;",
      "ECON_INFLATION,23;OIL,44;CENTRAL_BANK,71;"
    ].join("\t"),
    [
      "20260421211500-2",
      "20260421211500",
      "1",
      "example.com",
      "https://example.com/local/soil-samples/",
      "",
      "",
      "ENV_SOIL;",
      "ENV_SOIL,12;"
    ].join("\t")
  ].join("\n");

  const records = parseGdeltGkgCsv(rows);
  assert.equal(records.length, 1);
  assert.equal(records[0].url, "https://www.reuters.com/markets/commodities/oil-prices-rise-2026-04-21/");
  assert.deepEqual(records[0].normalizedThemes, ["CENTRAL_BANK", "ENERGY", "INFLATION"]);
});
