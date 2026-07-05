"use client";

import { SignInButton, UserButton, useUser } from "@clerk/nextjs";
import Link from "next/link";

function signInClass(): string {
  return "rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.95)] px-3 py-2 text-sm font-semibold text-[color:var(--ink)] hover:border-[color:var(--line-strong)]";
}

function ClerkAuthControls() {
  const { isLoaded, isSignedIn } = useUser();

  if (!isLoaded) {
    return <div className="h-9 w-16 rounded-xl border border-[color:var(--line)] bg-[color:rgba(9,21,34,0.7)]" />;
  }

  return (
    <div className="flex items-center gap-2">
      {!isSignedIn ? (
        <SignInButton mode="redirect">
          <button type="button" className={signInClass()}>
            Sign in
          </button>
        </SignInButton>
      ) : (
        <UserButton />
      )}
    </div>
  );
}

export function AuthControls({ enabled }: { enabled: boolean }) {
  if (!enabled) {
    return (
      <Link href={"/sign-in" as any} className={signInClass()}>
        Sign in
      </Link>
    );
  }

  return <ClerkAuthControls />;
}
