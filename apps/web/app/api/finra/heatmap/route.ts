import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import { loadCorpusDocuments } from "@/lib/server/data-store";

export const runtime = "nodejs";
export const revalidate = 300;

/* ─── Rule label lookup ──────────────────────────────────────────────────── */
const RULE_LABELS: Record<string, string> = {
  "2010": "Standards of Commercial Honor",
  "2020": "Manipulative & Deceptive Devices",
  "2040": "Payments to Unregistered Persons",
  "2111": "Suitability",
  "2232": "Customer Confirmations",
  "2310": "Recommendations to Customers",
  "2330": "Investment Analysis Tools",
  "2360": "Options",
  "3110": "Supervision",
  "3120": "Supervisory Control System",
  "3270": "Outside Business Activities",
  "3280": "Selling Away / Private Securities",
  "4210": "Margin Requirements",
  "4370": "Business Continuity Plans",
  "4511": "Books and Records",
  "4512": "Customer Account Information",
  "4513": "Records of Written Complaints",
  "4530": "Reporting Requirements",
  "5130": "Restrictions on IPO Equity Securities",
  "5210": "Publication of Transactions",
};

/* ─── Regex patterns ─────────────────────────────────────────────────────── */
// Matches: FINRA Rule 2010, NASD Rule 2110, NASD Conduct Rule 2310
const RULE_RE = /(?:FINRA|NASD)\s+(?:Conduct\s+)?Rule\s+(\d{3,4}[A-Z]?)/gi;

/* ─── Types ──────────────────────────────────────────────────────────────── */
export interface FINRAHeatmapRule {
  rule: string;
  label: string;
  total: number;
  by_month: number[];
}

export interface FINRARecentCase {
  document_id: string;
  title: string;
  date: string;
  doc_type: string;
  url: string;
}

export interface FINRAHeatmapPayload {
  generated_at: string;
  total_cases: number;
  doc_type_counts: Record<string, number>;
  months: string[]; // "YYYY-MM" strings, oldest first
  rules: FINRAHeatmapRule[];
  recent_cases: FINRARecentCase[];
  max_cell_value: number;
}

/* ─── Handler ────────────────────────────────────────────────────────────── */
export async function GET() {
  const requestId = createRequestId();

  try {
    const corpus = await loadCorpusDocuments();
    const awcDocs = corpus.filter(
      (doc) => String(doc.metadata?.source_kind ?? "") === "finra_awc"
    );

    // 18-month window ending this month
    const now = new Date();
    const months: string[] = [];
    for (let i = 17; i >= 0; i--) {
      const d = new Date(now.getFullYear(), now.getMonth() - i, 1);
      months.push(
        `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`
      );
    }
    const monthIndex = new Map(months.map((m, i) => [m, i]));

    const docTypeCounts: Record<string, number> = {};
    const ruleCounts = new Map<string, number[]>();
    const recentCases: FINRARecentCase[] = [];

    for (const doc of awcDocs) {
      const m = doc.metadata ?? {};
      const docType = String(m.doc_type ?? "AWC");
      docTypeCounts[docType] = (docTypeCounts[docType] ?? 0) + 1;

      // Resolve month bucket
      const dateStr = String(m.date ?? "").trim();
      let mIdx = -1;
      if (dateStr) {
        const parsed = new Date(
          dateStr.length <= 7 ? `${dateStr}-01` : dateStr
        );
        if (!Number.isNaN(parsed.getTime())) {
          const key = `${parsed.getFullYear()}-${String(parsed.getMonth() + 1).padStart(2, "0")}`;
          mIdx = monthIndex.get(key) ?? -1;
        }
      }

      // Extract FINRA rule citations (one count per rule per document)
      const text = String(doc.content?.full_text ?? "");
      const seenRules = new Set<string>();
      RULE_RE.lastIndex = 0;
      let match: RegExpExecArray | null;
      while ((match = RULE_RE.exec(text)) !== null) {
        const rule = match[1];
        if (seenRules.has(rule)) continue;
        seenRules.add(rule);
        if (!ruleCounts.has(rule)) {
          ruleCounts.set(rule, new Array(months.length).fill(0));
        }
        if (mIdx >= 0) {
          // biome-ignore lint/style/noNonNullAssertion: just set above
          ruleCounts.get(rule)![mIdx]++;
        }
      }

      recentCases.push({
        document_id: String(m.document_id ?? ""),
        title: String(m.title ?? ""),
        date: dateStr,
        doc_type: docType,
        url: String(m.url ?? ""),
      });
    }

    recentCases.sort((a, b) => {
      const ta = a.date ? new Date(a.date).getTime() : 0;
      const tb = b.date ? new Date(b.date).getTime() : 0;
      return tb - ta;
    });

    const rules: FINRAHeatmapRule[] = [...ruleCounts.entries()]
      .map(([rule, byMonth]) => ({
        rule,
        label: RULE_LABELS[rule] ?? "",
        total: byMonth.reduce((s, n) => s + n, 0),
        by_month: byMonth,
      }))
      .filter((r) => r.total > 0)
      .sort((a, b) => b.total - a.total)
      .slice(0, 15);

    const maxCellValue = rules.length
      ? Math.max(...rules.flatMap((r) => r.by_month))
      : 1;

    const payload: FINRAHeatmapPayload = {
      generated_at: new Date().toISOString(),
      total_cases: awcDocs.length,
      doc_type_counts: docTypeCounts,
      months,
      rules,
      recent_cases: recentCases.slice(0, 25),
      max_cell_value: maxCellValue,
    };

    return ok(payload, requestId);
  } catch {
    return fail(
      "Failed to load FINRA heatmap data",
      "HEATMAP_ERROR",
      500,
      requestId
    );
  }
}
