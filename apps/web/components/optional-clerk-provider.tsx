import { ClerkProvider } from "@clerk/nextjs";

export function isClerkConfigured(): boolean {
  return Boolean(process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY && process.env.CLERK_SECRET_KEY);
}

export function OptionalClerkProvider({ children }: { children: React.ReactNode }) {
  if (!isClerkConfigured()) {
    return <>{children}</>;
  }

  return <ClerkProvider>{children}</ClerkProvider>;
}
