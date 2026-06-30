import { createHash } from "node:crypto";
import { type NextRequest, NextResponse } from "next/server";

const COOKIE = "agentic_chat_auth";
const COOKIE_MAX_AGE = 60 * 60 * 24 * 7; // 7 days
const DEFAULT_CHAT_SECRET_HASH = "3ce6674b3c68fdb13b5c4e7e8a148452e21052f68dc106a485e61b79115d3b5b";

function hashSecret(secret: string): string {
  return createHash("sha256").update(secret).digest("hex");
}

function expectedHash(): string {
  const configuredHash = process.env.CHAT_SECRET_SHA256?.trim();
  if (configuredHash) {
    return configuredHash;
  }
  const configuredSecret = process.env.CHAT_SECRET?.trim();
  if (configuredSecret) {
    return hashSecret(configuredSecret);
  }
  return DEFAULT_CHAT_SECRET_HASH;
}

export async function POST(req: NextRequest) {
  let body: { secret?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid request body" }, { status: 400 });
  }

  const submittedHash = hashSecret(String(body.secret || ""));
  const targetHash = expectedHash();
  if (submittedHash !== targetHash) {
    return NextResponse.json({ ok: false, error: "Invalid password" }, { status: 401 });
  }

  const res = NextResponse.json({ ok: true });
  res.cookies.set(COOKIE, targetHash, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    maxAge: COOKIE_MAX_AGE,
    path: "/",
  });
  return res;
}
