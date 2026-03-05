import { buildDocumentListItems, loadCorpusDocuments, loadEnrichmentState, parseComparableDate } from "@/lib/server/data-store";
import { createRequestId, fail, normalizeText, ok } from "@/lib/server/api-utils";

export const runtime = "nodejs";

interface CitationItem {
  document_id: string;
  title: string;
  organization: string;
  source_kind: string;
  published_at: string;
  url: string;
  score: number;
  snippet: string;
}

function clampInt(value: unknown, fallback: number, minValue: number, maxValue: number): number {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  const n = Number.isFinite(parsed) ? parsed : fallback;
  return Math.max(minValue, Math.min(maxValue, n));
}

function tokenize(text: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const token of text.toLowerCase().split(/[^a-z0-9]+/)) {
    const trimmed = token.trim();
    if (trimmed.length < 3 || seen.has(trimmed)) {
      continue;
    }
    seen.add(trimmed);
    out.push(trimmed);
    if (out.length >= 24) {
      break;
    }
  }
  return out;
}

function countOccurrences(text: string, token: string): number {
  if (!text || !token) {
    return 0;
  }
  let count = 0;
  let index = text.indexOf(token);
  while (index >= 0) {
    count += 1;
    index = text.indexOf(token, index + token.length);
    if (count >= 12) {
      break;
    }
  }
  return count;
}

function extractSnippet(fullText: string, tokens: string[]): string {
  const sentences = fullText
    .replace(/\s+/g, " ")
    .split(/(?<=[.?!])\s+/)
    .map((item) => item.trim())
    .filter(Boolean);

  for (const sentence of sentences) {
    const lower = sentence.toLowerCase();
    if (tokens.some((token) => lower.includes(token))) {
      return sentence.slice(0, 260);
    }
  }

  return sentences[0]?.slice(0, 260) || "";
}

function buildAnswer(prompt: string, citations: CitationItem[]): string {
  if (citations.length === 0) {
    return `No strong matches were found for "${prompt}". Try adding specific entities, topics, or source names.`;
  }

  const lines = citations.map((item, idx) => {
    const dateLabel = item.published_at ? `, ${item.published_at}` : "";
    return `${idx + 1}. ${item.title} (${item.organization}${dateLabel})`;
  });

  return `Retrieved ${citations.length} relevant documents for "${prompt}".\n${lines.join("\n")}`;
}

export async function POST(request: Request) {
  const requestId = createRequestId();

  try {
    let body: Record<string, unknown> = {};
    try {
      body = (await request.json()) as Record<string, unknown>;
    } catch {
      body = {};
    }

    const prompt = normalizeText(body.prompt);
    if (!prompt) {
      return fail("Prompt is required.", "CHAT_PROMPT_REQUIRED", 400, requestId);
    }

    const topK = clampInt(body.top_k, 5, 1, 8);
    const tokens = tokenize(prompt);

    const [corpusDocs, enrichment] = await Promise.all([loadCorpusDocuments(), loadEnrichmentState()]);
    const items = buildDocumentListItems(corpusDocs, enrichment);
    const contentById = new Map<string, string>();
    for (const doc of corpusDocs) {
      const docId = String(doc.metadata?.document_id || "").trim();
      if (!docId) {
        continue;
      }
      contentById.set(docId, String(doc.content?.full_text || "").replace(/\s+/g, " ").trim());
    }

    const scored = items
      .map((item) => {
        const text = [
          item.title,
          item.organization,
          item.doc_type,
          item.source_kind,
          item.speaker,
          ...(item.tags || []),
          ...(item.topics || []),
          ...(item.keywords || [])
        ]
          .join(" ")
          .toLowerCase();
        const fullText = String(contentById.get(item.document_id) || "");
        const fullLower = fullText.toLowerCase();

        let score = 0;
        for (const token of tokens) {
          const titleHits = countOccurrences(item.title.toLowerCase(), token);
          const metaHits = countOccurrences(text, token);
          const fullHits = countOccurrences(fullLower, token);
          score += titleHits * 6 + metaHits * 2 + fullHits;
        }
        if (tokens.length === 0) {
          score = 1;
        }

        return {
          item,
          fullText,
          score
        };
      })
      .filter((entry) => entry.score > 0)
      .sort((a, b) => {
        if (b.score !== a.score) {
          return b.score - a.score;
        }
        return parseComparableDate(b.item.published_at || b.item.date) - parseComparableDate(a.item.published_at || a.item.date);
      })
      .slice(0, topK);

    const citations: CitationItem[] = scored.map((entry) => ({
      document_id: entry.item.document_id,
      title: entry.item.title || "Untitled",
      organization: entry.item.organization,
      source_kind: entry.item.source_kind,
      published_at: entry.item.published_at || entry.item.date,
      url: entry.item.url,
      score: entry.score,
      snippet: extractSnippet(entry.fullText, tokens)
    }));

    return ok(
      {
        answer: buildAnswer(prompt, citations),
        citations,
        retrieved_count: citations.length,
        model: "heuristic-retrieval"
      },
      requestId
    );
  } catch (error) {
    return fail(
      `Failed to answer chat request: ${error instanceof Error ? error.message : "Unknown error"}`,
      "CHAT_ANSWER_FAILED",
      500,
      requestId
    );
  }
}
