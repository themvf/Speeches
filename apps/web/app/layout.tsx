import type { Metadata } from "next";
import { IBM_Plex_Sans, Space_Grotesk } from "next/font/google";
import { AppNav } from "@/components/app-nav";
import { TickerBar } from "@/components/ticker-bar";
import "./globals.css";

const bodyFont = IBM_Plex_Sans({
  variable: "--font-body",
  subsets: ["latin"],
  weight: ["400", "500", "600"]
});

const displayFont = Space_Grotesk({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["500", "600", "700"]
});

export const metadata: Metadata = {
  title: "Policy Research Hub",
  description: "Regulatory intelligence dashboard for policy and enforcement research workflows."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${bodyFont.variable} ${displayFont.variable}`}>
        <a href="#main-content" className="skip-link">
          Skip to content
        </a>
        <div className="min-h-screen">
          <AppNav />
          <main id="main-content" className="pb-8">{children}</main>
          <TickerBar />
        </div>
      </body>
    </html>
  );
}
