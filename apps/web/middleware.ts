import { type NextRequest, NextResponse } from "next/server";

const COOKIE = "admin_auth";
const LOGIN_PATH = "/admin/login";

function isAdminPath(pathname: string) {
  return (pathname.startsWith("/admin") || pathname.startsWith("/api/admin")) &&
    pathname !== "/api/admin/login";
}

export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  if (!isAdminPath(pathname) || pathname === LOGIN_PATH) {
    return NextResponse.next();
  }

  const secret = process.env.ADMIN_SECRET;
  if (!secret) {
    // If no secret is configured, block access entirely rather than allow it open
    return new NextResponse("Admin access is not configured.", { status: 503 });
  }

  const cookie = req.cookies.get(COOKIE);
  if (cookie?.value === secret) {
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
