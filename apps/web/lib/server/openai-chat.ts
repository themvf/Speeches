import { getOpenAiConfig } from "@/lib/server/env";
import type { DocumentListItem } from "@/lib/server/types";

type ChatRole = "user" | "assistant";

export interface ChatHistoryMessage {
  role: ChatRole;
  content: string;
}

type OpenAiResponseContentItem = {
  type?: string;
  text?: string;
};

type OpenAiOutputItem = {
  type?: string;
  content?: OpenAiResponseContentItem[];
  results?: Array<Record<string, unknown>>;
};

type OpenAiResponsePayload = {
  output_text?: string;
  output?: OpenAiOutputItem[];
  error?: { message?: string };
};

type FileSearchResult = {
  filename: string;
  file_id: string;
  score: number;
  snippet: string;
};

export interface VectorChatCitation {
  document_id: string;
  title: string;
  organization: string;
  source_kind: string;
  published_at: string;
  url: string;
  score: number;
  snippet: string;
}

export interface VectorChatAnswer {
  answer: string;
  citations: VectorChatCitation[];
  retrieved_count: number;
  model: string;
}

interface AskVectorChatArgs {
  prompt: string;
  history: ChatHistoryMessage[];
  topK: number;
  vectorStoreIds: string[];
  documents: DocumentListItem[];
  latestIndexedDate?: string;
  model?: string;
}

function normalizeText(value: unknown): string {
  return String(value ?? "").replace(/\s+/g, " ").trim();
}

function normalizeResponseText(value: unknown): string {
  return String(value ?? "")
    .replace(/\r\n?/g, "\n")
    .replace(/\u00a0/g, " ")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function clampInt(value: number, fallback: number, minValue: number, maxValue: number): number {
  const parsed = Number.isFinite(value) ? value : fallback;
  return Math.max(minValue, Math.min(maxValue, parsed));
}

function chunkList<T>(items: T[], size: number): T[][] {
  const safeSize = Math.max(1, size);
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += safeSize) {
    out.push(items.slice(i, i + safeSize));
  }
  return out;
}

function extractDocIdFromFilename(filename: string): string {
  const text = normalizeText(filename);
  if (!text) {
    return "";
  }
  const lowered = text.split(/[\\/]/).pop() || text;
  const bracketed = lowered.match(/\[([a-f0-9]{24})\]\.txt$/i);
  if (bracketed?.[1]) {
    return bracketed[1].toLowerCase();
  }
  const plain = lowered.match(/([a-f0-9]{24})\.txt$/i);
  if (plain?.[1]) {
    return plain[1].toLowerCase();
  }
  return "";
}

function normalizeSnippet(value: unknown, maxChars = 320): string {
  const text = normalizeText(value);
  if (!text) {
    return "";
  }
  return text.slice(0, maxChars);
}

function extractResponseText(payload: OpenAiResponsePayload): string {
  const direct = normalizeResponseText(payload.output_text);
  if (direct) {
    return direct;
  }
  const output = Array.isArray(payload.output) ? payload.output : [];
  const pieces: string[] = [];
  for (const item of output) {
    if (item?.type !== "message" || !Array.isArray(item.content)) {
      continue;
    }
    for (const contentItem of item.content) {
      const text = normalizeResponseText(contentItem?.text);
      if (text) {
        pieces.push(text);
      }
    }
  }
  return pieces.join("\n\n").trim();
}

function extractResultSnippet(result: Record<string, unknown>): string {
  const directText = normalizeSnippet(result.text);
  if (directText) {
    return directText;
  }

  const content = Array.isArray(result.content) ? result.content : [];
  const parts: string[] = [];
  for (const item of content) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const text = normalizeSnippet((item as Record<string, unknown>).text);
    if (text) {
      parts.push(text);
    }
  }
  return normalizeSnippet(parts.join(" "));
}

function extractFileSearchResults(payload: OpenAiResponsePayload): FileSearchResult[] {
  const output = Array.isArray(payload.output) ? payload.output : [];
  const rows: FileSearchResult[] = [];
  for (const item of output) {
    if (item?.type !== "file_search_call" || !Array.isArray(item.results)) {
      continue;
    }
    for (const raw of item.results) {
      const result = raw && typeof raw === "object" ? raw : {};
      const numericScore = Number.parseFloat(String((result as Record<string, unknown>).score ?? "0"));
      rows.push({
        filename: normalizeText((result as Record<string, unknown>).filename),
        file_id: normalizeText((result as Record<string, unknown>).file_id),
        score: Number.isFinite(numericScore) ? numericScore : 0,
        snippet: extractResultSnippet(result as Record<string, unknown>)
      });
    }
  }
  return rows;
}

function mergeFileSearchResults(batches: FileSearchResult[], maxResults: number): FileSearchResult[] {
  const dedup = new Map<string, FileSearchResult>();
  for (const result of batches) {
    const snippetKey = normalizeText(result.snippet).toLowerCase().slice(0, 220);
    const key = `${result.file_id || result.filename}::${snippetKey}`;
    const existing = dedup.get(key);
    if (!existing) {
      dedup.set(key, { ...result });
      continue;
    }
    if (result.score > existing.score) {
      existing.score = result.score;
    }
    if (!existing.filename && result.filename) {
      existing.filename = result.filename;
    }
    if (!existing.file_id && result.file_id) {
      existing.file_id = result.file_id;
    }
    if (result.snippet.length > existing.snippet.length) {
      existing.snippet = result.snippet;
    }
  }
  return [...dedup.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.max(1, maxResults));
}

function trimHistory(history: ChatHistoryMessage[], maxMessages = 6, maxChars = 5000): ChatHistoryMessage[] {
  const filtered = (history || [])
    .filter((item) => item && (item.role === "user" || item.role === "assistant"))
    .map((item) => ({ role: item.role, content: normalizeText(item.content) }))
    .filter((item) => item.content);
  const recent = filtered.slice(-maxMessages);
  const kept: ChatHistoryMessage[] = [];
  let usedChars = 0;
  for (let i = recent.length - 1; i >= 0; i -= 1) {
    const item = recent[i];
    if (usedChars + item.content.length > maxChars && kept.length > 0) {
      continue;
    }
    kept.unshift(item);
    usedChars += item.content.length;
  }
  return kept;
}

function buildRetrievalPrompt(prompt: string, history: ChatHistoryMessage[]): string {
  const recentHistory = trimHistory(history, 4, 2200);
  if (!recentHistory.length) {
    return prompt;
  }
  const historyText = recentHistory
    .map((item) => `${item.role === "assistant" ? "Assistant" : "User"}: ${item.content}`)
    .join("\n");
  return `Conversation context:\n${historyText}\n\nCurrent user question:\n${prompt}`;
}

function buildResponseInput(prompt: string, history: ChatHistoryMessage[], evidenceContext: string): string {
  const recentHistory = trimHistory(history);
  const historyText = recentHistory.length
    ? `${recentHistory.map((item) => `${item.role === "assistant" ? "Assistant" : "User"}: ${item.content}`).join("\n")}\n\n`
    : "";
  return `${historyText}Current question:\n${prompt}\n\nEvidence Context:\n${evidenceContext || "No retrieved evidence available."}`;
}

function buildChatInstructions(latestIndexedDate?: string): string {
  const todayText = new Date().toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric"
  });
  const latestText = normalizeText(latestIndexedDate) || "the latest indexed date available";
  return [
    "You are a retrieval-grounded policy research assistant writing for an analyst who wants synthesis, not a source dump.",
    "Use only the retrieved corpus evidence provided in the Evidence Context for factual claims.",
    "Write in clear markdown-like plain text with short sections and bullets where useful.",
    "When evidence is sufficient, structure the answer with these headings in this order: '## Bottom line', '## What the evidence shows', '## Important nuance or disagreement', and '## Gaps or follow-up'.",
    "Under 'What the evidence shows', synthesize across sources instead of repeating snippets one by one.",
    "Cite evidence inline using [Source N] references when making factual claims. Reuse the provided source numbers exactly and do not invent citations.",
    "Do not append a raw source list inside the answer body. The UI will render sources separately.",
    "If evidence is insufficient, say exactly what is missing instead of guessing.",
    "If the user uses ambiguous temporal language like recent, latest, current, now, or today without a date range, ask one concise clarification question first.",
    `Today's date is ${todayText}. Latest indexed coverage appears to run through ${latestText}.`
  ].join(" ");
}

function buildEvidenceContext(results: FileSearchResult[], documentsById: Map<string, DocumentListItem>, maxItems = 14, maxChars = 20_000): string {
  let usedChars = 0;
  const blocks: string[] = [];
  for (let idx = 0; idx < results.length && blocks.length < maxItems; idx += 1) {
    const result = results[idx];
    const docId = extractDocIdFromFilename(result.filename);
    const doc = docId ? documentsById.get(docId) : undefined;
    const title = normalizeText(doc?.title) || normalizeText(result.filename) || "Unknown document";
    const organization = normalizeText(doc?.organization);
    const publishedAt = normalizeText(doc?.published_at || doc?.date);
    const sourceKind = normalizeText(doc?.source_kind);
    const snippet = normalizeSnippet(result.snippet, 700);
    if (!snippet) {
      continue;
    }
    const block = [
      `[Source ${blocks.length + 1}]`,
      `Title: ${title}`,
      organization ? `Organization: ${organization}` : "",
      publishedAt ? `Date: ${publishedAt}` : "",
      sourceKind ? `Source Kind: ${sourceKind}` : "",
      `Snippet: ${snippet}`
    ]
      .filter(Boolean)
      .join("\n");
    if (usedChars + block.length > maxChars) {
      break;
    }
    blocks.push(block);
    usedChars += block.length;
  }
  return blocks.join("\n\n").trim();
}

async function callOpenAiResponses(payload: Record<string, unknown>): Promise<OpenAiResponsePayload> {
  const cfg = getOpenAiConfig();
  if (!cfg.apiKey) {
    throw new Error("OPENAI_API_KEY is not configured for the web app.");
  }

  const response = await fetch(`${cfg.baseUrl.replace(/\/$/, "")}/responses`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${cfg.apiKey}`
    },
    body: JSON.stringify(payload),
    cache: "no-store"
  });

  const text = await response.text();
  let json: OpenAiResponsePayload | null = null;
  try {
    json = JSON.parse(text) as OpenAiResponsePayload;
  } catch {
    json = null;
  }

  if (!response.ok) {
    const message = normalizeText(json?.error?.message) || normalizeText(text) || `OpenAI request failed with status ${response.status}.`;
    throw new Error(message);
  }

  if (!json) {
    throw new Error("OpenAI returned a non-JSON response.");
  }
  return json;
}

async function runFileSearchCall(model: string, question: string, vectorStoreIds: string[], maxNumResults: number): Promise<FileSearchResult[]> {
  const payload = {
    model,
    input: question,
    tools: [
      {
        type: "file_search",
        vector_store_ids: vectorStoreIds,
        max_num_results: maxNumResults
      }
    ],
    include: ["file_search_call.results"]
  };
  const response = await callOpenAiResponses(payload);
  return extractFileSearchResults(response);
}

function buildCitations(results: FileSearchResult[], documentsById: Map<string, DocumentListItem>, maxItems: number): VectorChatCitation[] {
  const out: VectorChatCitation[] = [];
  const seen = new Set<string>();
  for (const result of results) {
    const docId = extractDocIdFromFilename(result.filename);
    const doc = docId ? documentsById.get(docId) : undefined;
    const dedupeKey = docId || result.file_id || result.filename;
    if (!dedupeKey || seen.has(dedupeKey)) {
      continue;
    }
    seen.add(dedupeKey);
    out.push({
      document_id: docId || dedupeKey,
      title: normalizeText(doc?.title) || normalizeText(result.filename) || "Untitled",
      organization: normalizeText(doc?.organization),
      source_kind: normalizeText(doc?.source_kind),
      published_at: normalizeText(doc?.published_at || doc?.date),
      url: normalizeText(doc?.url),
      score: result.score,
      snippet: normalizeSnippet(result.snippet, 280)
    });
    if (out.length >= maxItems) {
      break;
    }
  }
  return out;
}

export async function askVectorStoreChat(args: AskVectorChatArgs): Promise<VectorChatAnswer> {
  const cfg = getOpenAiConfig();
  const model = normalizeText(args.model) || normalizeText(cfg.model) || "gpt-5.1";
  const topK = clampInt(args.topK, 8, 1, 12);
  const vectorStoreIds = [...new Set((args.vectorStoreIds || []).map((item) => normalizeText(item)).filter(Boolean))];
  if (!vectorStoreIds.length) {
    throw new Error("No active vector stores are available for web chat. Build/Sync the knowledge index first.");
  }

  const documentsById = new Map<string, DocumentListItem>();
  for (const doc of args.documents || []) {
    const docId = normalizeText(doc.document_id);
    if (docId) {
      documentsById.set(docId, doc);
    }
  }

  const retrievalPrompt = buildRetrievalPrompt(normalizeText(args.prompt), args.history || []);
  const retrievalBatches = chunkList(vectorStoreIds, 2);
  const allResults: FileSearchResult[] = [];
  for (const batch of retrievalBatches) {
    const batchResults = await runFileSearchCall(model, retrievalPrompt, batch, topK);
    allResults.push(...batchResults);
  }
  const mergedResults = mergeFileSearchResults(allResults, Math.max(topK * 4, 16));
  if (!mergedResults.length) {
    return {
      answer: `I could not retrieve relevant indexed documents for "${normalizeText(args.prompt)}". Try adding specific entities, agencies, dates, or source names.`,
      citations: [],
      retrieved_count: 0,
      model
    };
  }

  const evidenceContext = buildEvidenceContext(mergedResults, documentsById);
  const synthesisPayload = {
    model,
    instructions: buildChatInstructions(args.latestIndexedDate),
    input: buildResponseInput(normalizeText(args.prompt), args.history || [], evidenceContext)
  };
  const synthesisResponse = await callOpenAiResponses(synthesisPayload);
  const answer = extractResponseText(synthesisResponse) || "No answer returned.";

  return {
    answer,
    citations: buildCitations(mergedResults, documentsById, topK),
    retrieved_count: mergedResults.length,
    model
  };
}
