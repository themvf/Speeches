import { auth } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";
import { loadUserPortfolios, saveUserPortfolios } from "@/lib/server/portfolio-store";

export const dynamic = "force-dynamic";

function clerkConfigured(): boolean {
  return Boolean(process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY && process.env.CLERK_SECRET_KEY);
}

async function currentUserId(): Promise<string | null> {
  if (!clerkConfigured()) return null;
  return (await auth()).userId || null;
}

export async function GET(): Promise<NextResponse> {
  const userId = await currentUserId();
  if (!userId) return NextResponse.json({ ok: false, error: "Sign in to sync portfolios." }, { status: 401 });
  try {
    return NextResponse.json({ ok: true, data: await loadUserPortfolios(userId) });
  } catch (error) {
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : "Portfolio sync is unavailable." }, { status: 503 });
  }
}
export async function PUT(req: Request): Promise<NextResponse> {
  const userId = await currentUserId();
  if (!userId) return NextResponse.json({ ok: false, error: "Sign in to sync portfolios." }, { status: 401 });
  try {
    return NextResponse.json({ ok: true, data: await saveUserPortfolios(userId, await req.json()) });
  } catch (error) {
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : "Portfolio sync failed." }, { status: 503 });
  }
}
// End portfolio persistence route.
