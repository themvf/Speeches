import { NextResponse } from "next/server";
import { fetchYahooQuote } from "@/lib/server/yahoo";

export async function POST(req: Request) {
  const { symbol } = await req.json();
  if (!symbol || typeof symbol !== "string") {
    return NextResponse.json({ error: "symbol required" }, { status: 400 });
  }

  const upper = symbol.trim().toUpperCase();
  const q = await fetchYahooQuote(upper, 0);

  if (!q?.price) {
    return NextResponse.json({ valid: false, error: `No price data found for "${upper}"` });
  }

  return NextResponse.json({
    valid: true,
    symbol: upper,
    name: q.name,
    price: q.price,
    change: q.change,
    pct: q.pct,
    up: q.change >= 0,
  });
}
