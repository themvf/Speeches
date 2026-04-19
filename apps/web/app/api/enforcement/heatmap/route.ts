import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import { loadCorpusDocuments } from "@/lib/server/data-store";

export const runtime = "nodejs";
export const revalidate = 300;

/* ─── FINRA rule labels ──────────────────────────────────────────────────── */
const FINRA_RULE_LABELS: Record<string, string> = {
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

/* ─── SEC statute / rule labels ──────────────────────────────────────────── */
const SEC_RULE_LABELS: Record<string, string> = {
  "10(b)":   "Securities Fraud",
  "10b-5":   "Anti-Fraud Rule 10b-5",
  "17(a)":   "Fraudulent Transactions",
  "206(4)":  "IA Fraud – Rule 206(4)",
  "206(2)":  "IA Fraud – Negligent",
  "9(a)":    "Market Manipulation",
  "13b2-1":  "Books & Records Falsification",
  "13b2-2":  "Officer Certification Fraud",
  "15(a)":   "Unregistered Broker-Dealer",
  "21C":     "Cease-and-Desist",
};

/* ─── Regex patterns ─────────────────────────────────────────────────────── */
const FINRA_RULE_RE =
  /(?:FINRA|NASD)\s+(?:Conduct\s+)?Rule\s+(\d{3,4}[A-Z]?)/gi;

const SEC_RULE_RE =
  /(?:(?:Section|§)\s+)?(10b-5[a-c]?|10\([ab]\)|17\([a-c]\)(?:\(\d\))?|206\(\d\)|9\([a-c]\)(?:\(\d\))?|13b2-[12]|15\([ab]\)|21C)/g;

/* ─── Shared types ────────────────────────────────────────────────────────── */
export interface HeatmapRule {
  rule: string;
  label: string;
  total: number;
  by_month: number[];
}

export interface RecentCase {
  document_id: string;
  title: string;
  date: string;
  doc_type: string;
  url: string;
  agency: "FINRA" | "SEC";
}

export interface AgencyHeatmap {
  total_cases: number;
  doc_type_counts: Record<string, number>;
  months: string[];
  rules: HeatmapRule[];
  recent_cases: RecentCase[];
  max_cell_value: number;
}

export interface EnforcementHeatmapPayload {
  generated_at: string;
  months: string[];
  finra: AgencyHeatmap;
  sec: AgencyHeatmap;
}

/* ─── Build 18-month window ──────────────────────────────────────────────── */
function buildMonths(): { months: string[]; monthIndex: Map<string, number> } {
  const now = new Date();
  const months: string[] = [];
  for (let i = 17; i >= 0; i--) {
    const d = new Date(now.getFullYear(), now.getMonth() - i, 1);
    months.push(
      `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`
    );
  }
  return { months, monthIndex: new Map(months.map((m, i) => [m, i])) };
}

function parseDateToMonthKey(dateStr: string): string | null {
  const s = (dateStr || "").trim();
  if (!s) return null;
  const d = new Date(s.length <= 7 ? `${s}-01` : s);
  if (Number.isNaN(d.getTime())) return null;
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
}

function buildAgencyHeatmap(
  // biome-ignore lint/suspicious/noExplicitAny: corpus doc
  docs: any[],
  agency: "FINRA" | "SEC",
  ruleRe: RegExp,
  ruleLabels: Record<string, string>,
  months: string[],
  monthIndex: Map<string, number>
): AgencyHeatmap {
  const docTypeCounts: Record<string, number> = {};
  const ruleCounts = new Map<string, number[]>();
  const recentCases: RecentCase[] = [];

  for (const doc of docs) {
    const m = doc.metadata ?? {};
    const docType = String(m.doc_type ?? (agency === "FINRA" ? "AWC" : "Litigation Release"));
    docTypeCounts[docType] = (docTypeCounts[docType] ?? 0) + 1;

    const dateStr = String(m.published_date ?? m.date ?? "").trim();
    const monthKey = parseDateToMonthKey(dateStr);
    const mIdx = monthKey ? (monthIndex.get(monthKey) ?? -1) : -1;

    const text = String(doc.content?.full_text ?? "");
    ruleRe.lastIndex = 0;
    const seenRules = new Set<string>();
    let match: RegExpExecArray | null;
    // biome-ignore lint/suspicious/noAssignInExpressions: regex loop idiom
    while ((match = ruleRe.exec(text)) !== null) {
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
      agency,
    });
  }

  recentCases.sort((a, b) => {
    const ta = a.date ? new Date(a.date).getTime() : 0;
    const tb = b.date ? new Date(b.date).getTime() : 0;
    return tb - ta;
  });

  const rules: HeatmapRule[] = [...ruleCounts.entries()]
    .map(([rule, byMonth]) => ({
      rule,
      label: ruleLabels[rule] ?? "",
      total: byMonth.reduce((s, n) => s + n, 0),
      by_month: byMonth,
    }))
    .filter((r) => r.total > 0)
    .sort((a, b) => b.total - a.total)
    .slice(0, 15);

  const maxCellValue = rules.length
    ? Math.max(...rules.flatMap((r) => r.by_month))
    : 1;

  return {
    total_cases: docs.length,
    doc_type_counts: docTypeCounts,
    months,
    rules,
    recent_cases: recentCases.slice(0, 25),
    max_cell_value: maxCellValue,
  };
}

/* ─── Handler ────────────────────────────────────────────────────────────── */
export async function GET() {
  const requestId = createRequestId();

  try {
    const corpus = await loadCorpusDocuments();
    const { months, monthIndex } = buildMonths();

    const finraDocs = corpus.filter(
      (doc) => String(doc.metadata?.source_kind ?? "") === "finra_awc"
    );
    const secDocs = corpus.filter(
      (doc) =>
        String(doc.metadata?.source_kind ?? "") === "sec_enforcement_litigation"
    );

    const finra = buildAgencyHeatmap(
      finraDocs,
      "FINRA",
      FINRA_RULE_RE,
      FINRA_RULE_LABELS,
      months,
      monthIndex
    );
    const sec = buildAgencyHeatmap(
      secDocs,
      "SEC",
      SEC_RULE_RE,
      SEC_RULE_LABELS,
      months,
      monthIndex
    );

    const payload: EnforcementHeatmapPayload = {
      generated_at: new Date().toISOString(),
      months,
      finra,
      sec,
    };

    return ok(payload, requestId);
  } catch {
    return fail(
      "Failed to load enforcement heatmap data",
      "HEATMAP_ERROR",
      500,
      requestId
    );
  }
}
