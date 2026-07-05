import { SignUp } from "@clerk/nextjs";
import { AuthNotConfigured } from "@/components/auth-not-configured";
import { isClerkConfigured } from "@/components/optional-clerk-provider";

export default function SignUpPage() {
  if (!isClerkConfigured()) {
    return <AuthNotConfigured />;
  }

  return (
    <main className="mx-auto grid min-h-[70vh] w-full max-w-xl place-items-center px-4 py-12">
      <SignUp path="/sign-up" routing="path" signInUrl="/sign-in" />
    </main>
  );
}
