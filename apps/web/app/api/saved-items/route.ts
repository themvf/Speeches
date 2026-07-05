import { auth } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";
import {
  loadUserSavedItems,
  normalizeSavedItemsPayload,
  saveUserSavedItems,
} from "@/lib/server/saved-items-store";

export const dynamic = "force-dynamic";

function clerkConfigured(): boolean {
  return Boolean(process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY && process.env.CLERK_SECRET_KEY);
}

async function currentUserId(): Promise<string | null> {
  if (!clerkConfigured()) {
    return null;
  }
  const { userId } = await auth();
  return userId || null;
}

export async function GET(): Promise<NextResponse> {
  const userId = await currentUserId();
  if (!userId) {
    return NextResponse.json({ ok: false, error: "Sign in to sync saved items." }, { status: 401 });
  }

  try {
    const data = await loadUserSavedItems(userId);
    return NextResponse.json({ ok: true, data });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Saved item sync is unavailable." },
      { status: 503 }
    );
  }
}

export async function PUT(req: Request): Promise<NextResponse> {
  const userId = await currentUserId();
  if (!userId) {
    return NextResponse.json({ ok: false, error: "Sign in to sync saved items." }, { status: 401 });
  }

  try {
    const payload = normalizeSavedItemsPayload(await req.json());
    const data = await saveUserSavedItems(userId, payload);
    return NextResponse.json({ ok: true, data });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Saved item sync failed." },
      { status: 503 }
    );
  }
}
