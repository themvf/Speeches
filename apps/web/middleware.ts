import { type NextRequest, NextResponse } from "next/server";

const COOKIE = "admin_auth";
const LOGIN_PATH = "/admin/login";

function isAdminPath(pathname: string) {
  return (pathname.startsWith("/admin") || pathname.startsWith("/api/admin")) &&
    pathname !== "/api/admin/login";
}

async function hashSecret(secret: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(secret));
  return Array.from(new Uint8Array(buf)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

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

  // API routes get 401 instead of a redirect
  if (pathname.startsWith("/api/")) {
    return NextResponse.json({ ok: false, error: "Unauthorized" }, { status: 401 });
  }

  const loginUrl = req.nextUrl.clone();
  loginUrl.pathname = LOGIN_PATH;
  loginUrl.searchParams.set("next", pathname);
  return NextResponse.redirect(loginUrl);
}

export const config = {
  matcher: ["/admin", "/admin/:path*", "/api/admin/:path*"],
};
