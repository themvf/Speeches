import { NextResponse } from "next/server";

export const revalidate = 60;

// ETFs are used for reliable free-tier support on Finnhub.
// ^GSPC / ^DJI / ^IXIC / ^N225 / ^FTSE return null on many free accounts.
const INDICES = [
  { symbol: "SPY", name: "S&P 500" },
  { symbol: "DIA", name: "Dow Jones" },
  { symbol: "QQQ", name: "Nasdaq 100" },
  { symbol: "EWJ", name: "Nikkei (EWJ)" },
  { symbol: "EWU", name: "FTSE (EWU)" },
];

type FinnhubQuote = {
  c: number | null;
  d: number | null;
  dp: number | null;
};

export async function GET() {
  const apiKey = process.env.FINNHUB_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "FINNHUB_API_KEY not set" }, { status: 500 });
  }

  const settled = await Promise.allSettled(
    INDICES.map(async ({ symbol, name }) => {
      const res = await fetch(
        `https://finnhub.io/api/v1/quote?symbol=${encodeURIComponent(symbol)}&token=${apiKey}`,
        { next: { revalidate: 60 } }
      );
      const data: FinnhubQuote = await res.json();
      if (!data.c) throw new Error(`No price for ${symbol}`);
      return {
        symbol,
        name,
        price: data.c,
        change: data.d ?? 0,
        pct: data.dp ?? 0,
        up: (data.d ?? 0) >= 0,
      };
    })
  );

  const results = settled
    .filter((r) => r.status === "fulfilled")
    .map((r) => (r as PromiseFulfilledResult<typeof results[0]>).value);

  return NextResponse.json(results);
}
