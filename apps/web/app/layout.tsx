import type { Metadata } from "next";
import localFont from "next/font/local";
import { AppNav } from "@/components/app-nav";
import { isClerkConfigured, OptionalClerkProvider } from "@/components/optional-clerk-provider";
import "./globals.css";

const bodyFont = localFont({
  variable: "--font-body",
  display: "swap",
  src: [
    {path: "../node_modules/@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-400-normal.woff2", weight: "400"},
    {path: "../node_modules/@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-500-normal.woff2", weight: "500"},
    {path: "../node_modules/@fontsource/ibm-plex-sans/files/ibm-plex-sans-latin-600-normal.woff2", weight: "600"}
  ]
});

const displayFont = localFont({
  variable: "--font-display",
  display: "swap",
  src: [
    {path: "../node_modules/@fontsource/space-grotesk/files/space-grotesk-latin-500-normal.woff2", weight: "500"},
    {path: "../node_modules/@fontsource/space-grotesk/files/space-grotesk-latin-600-normal.woff2", weight: "600"},
    {path: "../node_modules/@fontsource/space-grotesk/files/space-grotesk-latin-700-normal.woff2", weight: "700"}
  ]
});

export const metadata: Metadata = {
  title: "Policy Research Hub",
  description: "Regulatory intelligence dashboard for policy and enforcement research workflows."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const authEnabled = isClerkConfigured();

  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${bodyFont.variable} ${displayFont.variable}`}>
        <OptionalClerkProvider>
          <a href="#main-content" className="skip-link">
            Skip to content
          </a>
          <div className="min-h-screen">
            <AppNav authEnabled={authEnabled} />
            <main id="main-content" className="pb-8">{children}</main>
          </div>
        </OptionalClerkProvider>
      </body>
    </html>
  );
}
