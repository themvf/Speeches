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
- Each sector is collapsible to reveal 5 representative stocks with ticker, name, price, and % change
- Sector-level % change sourced from Finnhub; stock quotes fetched in a single parallel batch

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
| `GET /api/v1/stock/sector-performance` | Sector-level % change | 300s |
| `GET /api/v1/quote?symbol=<ticker>` (×55 batch) | Per-sector stock quotes | 300s |
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

## Internal API Routes

All routes live under `app/api/market/` and follow the standard `{ ok, data, request_id }` envelope pattern.

| Route | Source | Cache TTL |
|---|---|---|
| `GET /api/market/overview` | Finnhub (14 symbols) | 60s |
| `GET /api/market/sectors` | Finnhub sector-perf + 55 stock quotes | 300s |
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
| Sectors | 300s |
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
- `MoverQuote` — rank, symbol, name, price, pct, change, up
- `MarketMoversData` — gainers[], losers[], generatedAt
- `CryptoCoin` — rank, id, symbol, name, price, pct24h, marketCap, volume24h, up
- `MarketCryptoData`
- `ExchangeInfo` — code, name, timezone, status
- `ExchangeRegionGroup` — region + exchanges[]
- `MarketExchangesData`

---

## Rate Limit Budget

On the Finnhub free plan (60 calls/min), the worst-case burst on a cold cache is:

| Route | Calls |
|---|---|
| /api/market/overview | 14 |
| /api/market/sectors | 56 (1 sector-perf + 55 stock quotes) |
| /api/market/movers | 35 |
| /api/market/exchanges | 16 |
| **Total (all tabs cold)** | **121** |

In practice this is spread across multiple minutes because:
1. Tabs fetch lazily — only the active tab fires on load
2. Next.js caches responses for the full revalidation window
3. Sectors (the heaviest tab) revalidates only every 5 minutes
