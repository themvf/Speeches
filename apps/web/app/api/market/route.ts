import { NextResponse } from "next/server";
import { getTickerConfig } from "@/lib/ticker-config";
import { fetchYahooQuote } from "@/lib/server/yahoo";

export const revalidate = 60;

type QuoteResult = {
  symbol: string;
  name: string;
  price: number;
  change: number;
  pct: number;
  up: boolean;
};

export async function GET() {
  const tickers = await getTickerConfig();

  const settled = await Promise.allSettled(
    tickers.map(async ({ symbol, name }) => {
      const q = await fetchYahooQuote(symbol, 60);
      if (!q?.price) throw new Error(`No price for ${symbol}`);
      return {
        symbol,
        name,
        price: q.price,
        change: q.change,
        pct: q.pct,
        up: q.change >= 0,
      } satisfies QuoteResult;
    })
  );

  const results = settled
    .filter((r) => r.status === "fulfilled")
    .map((r) => (r as PromiseFulfilledResult<QuoteResult>).value);

  return NextResponse.json(results);
}
