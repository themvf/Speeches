import { NextResponse } from "next/server";

export const revalidate = 60;

const INDICES = [
  { symbol: "^GSPC", name: "S&P 500" },
  { symbol: "^DJI", name: "Dow Jones" },
  { symbol: "^IXIC", name: "Nasdaq" },
  { symbol: "^N225", name: "Nikkei 225" },
  { symbol: "^FTSE", name: "FTSE 100" },
];

type FinnhubQuote = {
  c: number;
  d: number;
  dp: number;
};

export async function GET() {
  const apiKey = process.env.FINNHUB_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "FINNHUB_API_KEY not set" }, { status: 500 });
  }

  try {
    const results = await Promise.all(
      INDICES.map(async ({ symbol, name }) => {
        const res = await fetch(
          `https://finnhub.io/api/v1/quote?symbol=${encodeURIComponent(symbol)}&token=${apiKey}`,
          { next: { revalidate: 60 } }
        );
        const data: FinnhubQuote = await res.json();
        return {
          symbol,
          name,
          price: data.c,
          change: data.d,
          pct: data.dp,
          up: data.d >= 0,
        };
      })
    );
    return NextResponse.json(results);
  } catch {
    return NextResponse.json({ error: "Failed to fetch market data" }, { status: 500 });
  }
}
