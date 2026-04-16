import { NextResponse } from "next/server";

type FinnhubQuote = { c: number | null; d: number | null; dp: number | null };
type FinnhubProfile = { name?: string };

export async function POST(req: Request) {
  const apiKey = process.env.FINNHUB_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "FINNHUB_API_KEY not set" }, { status: 500 });
  }

  const { symbol } = await req.json();
  if (!symbol || typeof symbol !== "string") {
    return NextResponse.json({ error: "symbol required" }, { status: 400 });
  }

  const upper = symbol.trim().toUpperCase();

  // Fetch quote and company profile in parallel
  const [quoteRes, profileRes] = await Promise.all([
    fetch(`https://finnhub.io/api/v1/quote?symbol=${encodeURIComponent(upper)}&token=${apiKey}`),
    fetch(`https://finnhub.io/api/v1/stock/profile2?symbol=${encodeURIComponent(upper)}&token=${apiKey}`),
  ]);

  const quote: FinnhubQuote = await quoteRes.json();
  const profile: FinnhubProfile = await profileRes.json();

  if (!quote.c) {
    return NextResponse.json({ valid: false, error: `No price data found for "${upper}"` });
  }

  return NextResponse.json({
    valid: true,
    symbol: upper,
    name: profile.name ?? upper,
    price: quote.c,
    change: quote.d ?? 0,
    pct: quote.dp ?? 0,
    up: (quote.d ?? 0) >= 0,
  });
}
