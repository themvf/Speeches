import { clerkMiddleware } from "@clerk/nextjs/server";
import { type NextRequest, NextResponse } from "next/server";

const COOKIE = "admin_auth";
const LOGIN_PATH = "/admin/login";
const CHAT_COOKIE = "agentic_chat_auth";
const CHAT_LOGIN_PATH = "/chats/login";
const DEFAULT_CHAT_SECRET_HASH = "3ce6674b3c68fdb13b5c4e7e8a148452e21052f68dc106a485e61b79115d3b5b";

function isAdminPath(pathname: string) {
  return (pathname.startsWith("/admin") || pathname.startsWith("/api/admin")) &&
    pathname !== "/api/admin/login";
}

function isChatPath(pathname: string) {
  return (pathname === "/chats" || pathname.startsWith("/api/chats")) &&
    pathname !== "/api/chats/login";
}

async function hashSecret(secret: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(secret));
  return Array.from(new Uint8Array(buf)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function chatSecretHash(): Promise<string> {
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

async function legacyMiddleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  if (pathname === CHAT_LOGIN_PATH) {
    return NextResponse.next();
  }

  if (isChatPath(pathname)) {
    const cookie = req.cookies.get(CHAT_COOKIE);
    if (cookie?.value && cookie.value === await chatSecretHash()) {
      return NextResponse.next();
    }

    if (pathname.startsWith("/api/")) {
      return NextResponse.json({ ok: false, error: "Unauthorized" }, { status: 401 });
    }

    const loginUrl = req.nextUrl.clone();
    loginUrl.pathname = CHAT_LOGIN_PATH;
    loginUrl.searchParams.set("next", pathname);
    return NextResponse.redirect(loginUrl);
  }

  if (!isAdminPath(pathname) || pathname === LOGIN_PATH) {
    return NextResponse.next();
  }

  const secret = process.env.ADMIN_SECRET;
  if (!secret) {
    return new NextResponse("Admin access is not configured.", { status: 503 });
  }

  const cookie = req.cookies.get(COOKIE);
  if (cookie?.value && cookie.value === await hashSecret(secret)) {
    return NextResponse.next();
  }

  if (pathname.startsWith("/api/")) {
    return NextResponse.json({ ok: false, error: "Unauthorized" }, { status: 401 });
  }

  const loginUrl = req.nextUrl.clone();
  loginUrl.pathname = LOGIN_PATH;
  loginUrl.searchParams.set("next", pathname);
  return NextResponse.redirect(loginUrl);
}

const clerkConfigured = Boolean(process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY && process.env.CLERK_SECRET_KEY);

export default clerkConfigured
  ? clerkMiddleware((_auth, req) => legacyMiddleware(req))
  : legacyMiddleware;

export const config = {
  matcher: [
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    "/(api|trpc)(.*)",
  ],
};
