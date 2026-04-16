import { NextResponse } from "next/server";
import { getTickerConfig } from "@/lib/ticker-config";

export const revalidate = 60;

type FinnhubQuote = { c: number | null; d: number | null; dp: number | null };

type QuoteResult = {
  symbol: string;
  name: string;
  price: number;
  change: number;
  pct: number;
  up: boolean;
};

export async function GET() {
  const apiKey = process.env.FINNHUB_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "FINNHUB_API_KEY not set" }, { status: 500 });
  }

  const indices = await getTickerConfig();

  const settled = await Promise.allSettled(
    indices.map(async ({ symbol, name }) => {
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
    .map((r) => (r as PromiseFulfilledResult<QuoteResult>).value);

  return NextResponse.json(results);
}
