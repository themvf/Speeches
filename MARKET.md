# Market Page

Live market intelligence dashboard at `/market`, providing real-time financial data across five tabs — all sourced from free-tier APIs with no paid subscriptions required.

---

## Tabs

### Overview
- **US Indices** — 2×2 card grid showing S&P 500, Dow Jones, NASDAQ, and Russell 2000 with current price, absolute change, % change, and open/closed status
- **VIX / Fear & Greed Meter** — volatility index value with a gradient bar from Greed (green) → Calm (cyan) → Panic (red), labeled dynamically based on VIX level
- **Global Indexes** — table of 6 international markets with price, change, % change, and a proportional bar chart

### Sectors
- All 11 S&P GICS sectors sorted by daily % change
- Each sector is collapsible to reveal 10 representative stocks with ticker, name, price, and % change
- Sector-level performance uses Yahoo Finance ETF candles; stock quotes are fetched in a single parallel batch
- Clicking a company row lazily reveals up to 5 recent English-language U.S. Google News articles inline
- Company news is relevance-ranked, labeled with rule-based catalysts, expandable to 10 cached results, and manually refreshable with a 60-second cooldown
- Loaded results remain available in a 15-minute browser-session cache and add closed-row coverage/catalyst badges without prefetching
- Each expanded sector summarizes loaded coverage, article volume, dominant catalyst, and same-day price direction
- Deterministic controls sort/filter companies by price move, news recency, relevance, and catalyst; article filters support source tiers and press-release suppression
- Articles show publisher quality, likely-paywall and clustered-story indicators; recent categorized stories beside a 1%+ move are marked as a non-causal "Possible catalyst"

### Movers
- **Top 10 Gainers** and **Top 10 Losers** from a curated 35-stock watchlist
- Sorted by % change; displayed with rank, ticker, company, price, and a proportional bar

### Crypto
- Top 20 coins by market cap with rank, ticker, name, price, 24h %, market cap, and 24h volume
- Table header on desktop; responsive (market cap and volume hidden on mobile)

### Exchanges
- 16 major exchanges grouped by region: Americas, Europe, Asia Pacific
- Each row shows exchange code, name, timezone, and a live status pill: **OPEN** (green) / **CLOSED** (red) / **PRE** (amber) / **AFTER** (amber)

---

## API Connections

### Finnhub (free tier — 60 calls/min)
Used for US indices, VIX, global index proxies, sector performance, stock quotes, movers, and exchange status.

**Key:** `FINNHUB_API_KEY` environment variable

| Endpoint | Used for | Revalidation |
|---|---|---|
| `GET /api/v1/quote?symbol=SPY\|DIA\|QQQ\|IWM` | US index cards (ETF proxies) | 60s |
| `GET /api/v1/quote?symbol=^VIX` | VIX fear & greed meter | 60s |
| `GET /api/v1/quote?symbol=EWU\|EWG\|EWJ\|EWH\|EWA\|EWQ` | Global index ETF proxies | 60s |
| Yahoo Finance ETF chart (×11 batch) | Sector-level performance | 300s |
| Yahoo Finance quote (×110 batch) | Per-sector stock quotes | 300s |
| `GET /api/v1/quote?symbol=<ticker>` (×35 batch) | Movers watchlist | 120s |
| `GET /api/v1/stock/market-status?exchange=<code>` (×16) | Exchange open/closed status | 60s |

#### Global Index ETF Proxies
Because direct international index symbols (e.g. `^FTSE`) are paywalled on Finnhub's free plan, country ETFs are used as proxies:

| ETF | Tracks |
|---|---|
| EWU | FTSE 100 (UK) |
| EWG | DAX (Germany) |
| EWJ | Nikkei 225 (Japan) |
| EWH | Hang Seng (Hong Kong) |
| EWA | ASX 200 (Australia) |
| EWQ | CAC 40 (France) |

#### VIX Behavior
`^VIX` is attempted on the free plan. If Finnhub returns a null price, the VIX meter is hidden rather than showing incorrect data. Fear & Greed labels are computed server-side:

| VIX range | Label |
|---|---|
| < 15 | GREED |
| 15 – 25 | CALM |
| 25 – 35 | CONCERN |
| > 35 | PANIC |

---

### CoinGecko (free — no API key)
Used for the Crypto tab. Rate limit: ~10–50 calls/min on the keyless free tier.

| Endpoint | Used for | Revalidation |
|---|---|---|
| `GET /api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=20` | Top 20 coins by market cap | 120s |

---

## Deferred: Stock Social Signals

Hold off for now, but a future stock-signal feed could normalize X/Twitter posts, Reddit posts, news headlines, and filings into a common object for the market page:

```json
{
  "type": "stock_signal",
  "symbol": "TSLA",
  "company": "Tesla Inc.",
  "source": "x",
  "source_id": "1728108619189874825",
  "url": "https://x.com/elonmusk/status/1728108619189874825",
  "text": "More than 10 per human on average",
  "sentiment": "BULLISH",
  "sentiment_score": 0.72,
  "confidence": 0.81,
  "signal_reason": "High-engagement post from company-linked account with positive product-demand framing.",
  "engagement": {
    "likes": 104121,
    "reposts": 11311,
    "replies": 6526,
    "quotes": 2915,
    "views": 291500
  },
  "author": {
    "name": "Elon Musk",
    "username": "elonmusk",
    "followers": 172669889,
    "verified": true
  },
  "market_context": {
    "price": 184.21,
    "change_pct": 2.14,
    "market_state": "OPEN"
  },
  "created_at": "2023-11-24T17:49:36Z",
  "processed_at": "2026-06-29T00:00:00Z"
}
```

Recommended prototype path:

1. Use `vladkens/twscrape` as an internal Python ingestion worker, not inside the Next.js request path.
2. Query configured cashtags, ticker keywords, company accounts, and executive accounts.
3. Normalize results into `StockSignal` records and persist them to GCS, JSON, or Neon.
4. Add a route such as `GET /api/market/signals?symbol=TSLA`.
5. Surface the result later as a "Signals" tab or per-ticker drawer on the market page.

Notes:

- `twscrape` is a strong prototype candidate because it supports X search, tweet details, user timelines, parsed tweet/user models, engagement fields, media, quoted tweets, account sessions, and SQLite-backed account rotation.
- It depends on authorized X account cookies or sessions and private X endpoints, so reliability and terms-of-service risk need review before production use.
- Keep the app source-agnostic so the collector can be swapped later for `twikit`, `Scweet`, official X API access, Reddit/news sources, or a paid provider such as TwitterAPI.io without changing the market UI/data model.

---

## Internal API Routes

All routes live under `app/api/market/` and follow the standard `{ ok, data, request_id }` envelope pattern.

| Route | Source | Cache TTL |
|---|---|---|
| `GET /api/market/overview` | Finnhub (14 symbols) | 60s |
| `GET /api/market/sectors` | Yahoo ETF candles + 110 stock quotes | 300s |
| `GET /api/market/company-news?symbol=<ticker>&limit=5|10&refresh=0|1` | Google News RSS on demand | 900s per ticker; 60s manual-refresh cooldown |
| `GET /api/market/movers` | Finnhub (35 symbol batch) | 120s |
| `GET /api/market/crypto` | CoinGecko | 120s |
| `GET /api/market/exchanges` | Finnhub (16 market-status calls) | 60s |

Caching is handled via Next.js `fetch` cache with `{ next: { revalidate: N } }` — Vercel deduplicates concurrent requests within the same revalidation window.

---

## Component Structure

```
app/market/page.tsx                         Server Component (metadata + shell)
└── components/market-dashboard.tsx         "use client" — tab state, lazy fetch, polling
    ├── components/market/overview-tab.tsx  IndexCard × 4, VixMeter, GlobalIndexTable
    ├── components/market/sectors-tab.tsx   SectorRow (collapsible) → StockRow
    ├── components/market/movers-tab.tsx    MoversList × 2 (gainers / losers)
    ├── components/market/crypto-tab.tsx    CryptoTable with column headers
    └── components/market/exchanges-tab.tsx RegionGroup × 3 → ExchangeRow
```

**Fetch strategy:** Each tab fetches lazily on first activation (not at page load). Polling intervals run while the tab is active and are cleared on tab switch.

| Tab | Poll interval |
|---|---|
| Overview | 60s |
| Sectors | 300s for prices; 900s per opened company for news |
| Movers | 120s |
| Crypto | 120s |
| Exchanges | 60s |

---

## Type Definitions

All market types are in `apps/web/lib/server/types.ts`:

- `MarketStatus` — `"OPEN" | "CLOSED" | "PRE" | "AFTER"`
- `FearGreedLabel` — `"GREED" | "CALM" | "CONCERN" | "PANIC"`
- `MarketIndexQuote` — symbol, name, price, change, pct, up, status
- `VixQuote` — value, change, pct, label, gradientPct (0–100 bar position)
- `MarketOverviewData` — indices, vix, globalIndices, generatedAt
- `SectorData` / `SectorStock` — sector name + pct + nested stocks
- `MarketSectorsData`
- `CompanyNewsArticle` / `MarketCompanyNewsData` — normalized on-demand Google News RSS results with source-quality, story-cluster, relevance, and catalyst metadata
- `MoverQuote` — rank, symbol, name, price, pct, change, up
- `MarketMoversData` — gainers[], losers[], generatedAt
- `CryptoCoin` — rank, id, symbol, name, price, pct24h, marketCap, volume24h, up
- `MarketCryptoData`
- `ExchangeInfo` — code, name, timezone, status
- `ExchangeRegionGroup` — region + exchanges[]
- `MarketExchangesData`

---

## Rate Limit Budget

On the Finnhub free plan (60 calls/min), the worst-case burst on a cold cache is 65 requests. The Sectors tab uses Yahoo Finance separately:

| Route | Provider | Calls |
|---|---|---|
| /api/market/overview | Finnhub | 14 |
| /api/market/movers | Finnhub | 35 |
| /api/market/exchanges | Finnhub | 16 |
| **Finnhub total** |  | **65** |
| /api/market/sectors | Yahoo Finance | 121 (11 ETF candle requests + 110 stock quotes) |
| /api/market/company-news | Google News RSS | 0 cached; 1 typical miss; 2 maximum sparse-result miss |

In practice this is spread across multiple minutes because:
1. Tabs fetch lazily — only the active tab fires on load
2. Next.js caches responses for the full revalidation window
3. Sectors (the heaviest tab) revalidates only every 5 minutes
4. Company news is never prefetched or persisted; only an opened row can trigger its per-ticker RSS lookup

## Sector enrichment backlog

1. **Market cap and sector weight** — show each company's market capitalization, weight within its sector ETF, and rank so users can distinguish leaders from smaller constituents.
2. **Volume and liquidity signals** — add daily volume, 20-day average volume, and relative volume to make unusual trading activity visible.
3. **Valuation snapshot** — include forward P/E, price-to-sales, dividend yield, and sector-relative percentile with clear as-of dates.
4. **Price context** — add 1-week, 1-month, YTD, and 52-week-range performance plus a compact sparkline for every company.
5. **News and regulatory signals** — connect each ticker to recent corpus articles, SEC filings/enforcement mentions, sentiment, and the latest material catalyst with source links.
