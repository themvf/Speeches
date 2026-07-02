import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const BASE_URL = "https://www.finra.org/about/entities-we-regulate/broker-dealer-firms-we-regulate";
const OUTPUT_PATH = path.join(process.cwd(), "lib", "generated", "finra-member-firms.json");

type FinraMemberFirm = {
  name: string;
  normalizedName: string;
  pageUrl: string;
  rssUrl: string;
};

function decodeHtml(value: string): string {
  return value
    .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) => String.fromCharCode(Number.parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_, dec) => String.fromCharCode(Number.parseInt(dec, 10)))
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&nbsp;/g, " ")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">");
}

function normalizeFirmName(value: string): string {
  return decodeHtml(value)
    .normalize("NFKC")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizedKey(value: string): string {
  return normalizeFirmName(value)
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function googleNewsRssUrl(firmName: string): string {
  const query = `"${firmName.replace(/"/g, "")}" when:7d`;
  return `https://news.google.com/rss/search?q=${encodeURIComponent(query)}&hl=en-US&gl=US&ceid=US:en`;
}

function extractFirmNames(html: string): string[] {
  const bodyMatch = html.match(/<div class="views-field views-field-body"><div class="field-content">([\s\S]*?)<\/div><\/div>/);
  const body = bodyMatch?.[1] ?? "";
  const names: string[] = [];
  const paragraphRe = /<p>([\s\S]*?)<\/p>/g;
  let match: RegExpExecArray | null;
  while ((match = paragraphRe.exec(body)) !== null) {
    const firstLine = match[1].split(/<br\s*\/?>/i)[0] ?? "";
    const name = normalizeFirmName(firstLine.replace(/<[^>]+>/g, ""));
    if (name) {
      names.push(name);
    }
  }
  return names;
}

async function fetchPage(index: number): Promise<{ pageUrl: string; names: string[] }> {
  const pageUrl = `${BASE_URL}/${index}`;
  const response = await fetch(pageUrl, {
    headers: {
      "user-agent": "PolicyResearchHub/1.0 FINRA member firm registry sync",
    },
  });
  if (!response.ok) {
    throw new Error(`FINRA firm list fetch failed for ${pageUrl}: ${response.status}`);
  }
  return { pageUrl, names: extractFirmNames(await response.text()) };
}

async function main() {
  const byKey = new Map<string, FinraMemberFirm>();
  for (let pageIndex = 0; pageIndex <= 26; pageIndex += 1) {
    const { pageUrl, names } = await fetchPage(pageIndex);
    for (const name of names) {
      const key = normalizedKey(name);
      if (!key || byKey.has(key)) continue;
      byKey.set(key, {
        name,
        normalizedName: key,
        pageUrl,
        rssUrl: googleNewsRssUrl(name),
      });
    }
  }

  const firms = [...byKey.values()].sort((a, b) => a.name.localeCompare(b.name));
  await mkdir(path.dirname(OUTPUT_PATH), { recursive: true });
  await writeFile(
    OUTPUT_PATH,
    `${JSON.stringify({
      source: "FINRA Broker-Dealer Firms We Regulate",
      sourceUrl: BASE_URL,
      generatedAt: new Date().toISOString(),
      count: firms.length,
      firms,
    }, null, 2)}\n`,
    "utf8"
  );
  console.log(`Wrote ${firms.length} FINRA member firms to ${OUTPUT_PATH}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
