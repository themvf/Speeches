import { createRequestId, fail, ok } from "@/lib/server/api-utils";
import { loadCorpusDocuments, parseComparableDate } from "@/lib/server/data-store";
import type { CustomDocumentRecord } from "@/lib/server/types";

export const runtime = "nodejs";
export const revalidate = 300;

type Agency = "SEC" | "FINRA";
type AgencyKey = "sec" | "finra";

const FINRA_RULE_LABELS: Record<string, string> = {
  "2010": "Standards of Commercial Honor",
  "2020": "Manipulative and Deceptive Devices",
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
  "5210": "Publication of Transactions"
};

const SEC_RULE_LABELS: Record<string, string> = {
  "Rule 10b-5": "Anti-Fraud Rule 10b-5",
  "Rule 13b2-1": "Books and Records Falsification",
  "Rule 13b2-2": "Officer Certification Fraud",
  "Rule 206(4)-7": "Advisers Act Compliance Rule",
  "Rule 206(4)-8": "Private Fund Anti-Fraud Rule",
  "Rule 15c3-1": "Net Capital Rule",
  "Rule 15c3-3": "Customer Protection Rule",
  "Exchange Act 10(b)": "Exchange Act Section 10(b)",
  "Exchange Act 13(a)": "Exchange Act Section 13(a)",
  "Exchange Act 15(a)": "Exchange Act Section 15(a)",
  "Exchange Act 21C": "Cease-and-Desist Authority",
  "Securities Act 17(a)": "Securities Act Section 17(a)",
  "Advisers Act 204(a)": "Advisers Act Section 204(a)",
  "Advisers Act 206(1)": "Advisers Act Section 206(1)",
  "Advisers Act 206(2)": "Advisers Act Section 206(2)",
  "Advisers Act 206(4)": "Advisers Act Section 206(4)",
  "Advisers Act 207": "Advisers Act Section 207"
};

const FINRA_RULE_BLOCK_RE =
  /\b(?:FINRA|NASD)\s+(?:Conduct\s+)?Rules?\s+([0-9A-Z,\sand\/.-]{3,120})/gi;
const FINRA_VIOLATION_RULE_BLOCK_RE =
  /\bviolat(?:ed|ing|ions?\s+of)\s+(?:FINRA\s+|NASD\s+)?(?:Conduct\s+)?Rules?\s+([0-9A-Z,\sand\/.-]{3,120})/gi;
const FINRA_RULE_TOKEN_RE = /\b\d{3,4}[A-Z]?\b/g;
const SEC_RULE_RE = /\bRule\s+(10b-5[a-c]?|13b2-[12]|206\(4\)-[0-9]+|15c3-[0-9]+|17a-[0-9]+|17Ad-[0-9]+)\b/gi;
const SEC_SECTION_BLOCK_RE =
  /\bSections?\s+([^.;:\n]{1,180}?)\s+of\s+the\s+((?:Securities|Exchange|Investment Advisers|Advisers|Investment Company)[^.;:\n]{0,90}?Act(?:\s+of\s+\d{4})?)/gi;
const SEC_STANDALONE_SECTION_RE =
  /\b(?:Section|Sec\.?|section|\u00a7)\s+(10\([ab]\)|13\([a-z]\)|15\([ab]\)|17\([a-c]\)|17A\([a-z]\)|21C|204\(a\)|206\(\d\)|207)\b/g;
const SECTION_TOKEN_RE = /\b(?:\d{1,3}[A-Za-z]?|21C)(?:\([a-z0-9]+\)){0,3}(?:-[0-9]+)?\b/g;
const MONEY_RE = /\$[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*(?:million|billion))?/gi;

export interface EnforcementBetaCitation {
  citation: string;
  label: string;
  agency: Agency;
}

export interface EnforcementBetaCitationActivity extends EnforcementBetaCitation {
  total: number;
  by_month: number[];
}

export interface EnforcementBetaAction {
  document_id: string;
  agency: Agency;
  source_kind: string;
  title: string;
  date: string;
  month: string;
  doc_type: string;
  url: string;
  release_no: string;
  action_type: string;
  forum: string;
  outcome_status: string;
  citations: EnforcementBetaCitation[];
  entities: string[];
  sanctions: string[];
  summary: string;
  data_quality: {
    has_date: boolean;
    has_full_text: boolean;
    has_citations: boolean;
  };
}

export interface EnforcementBetaAgencyPayload {
  agency: Agency;
  total_actions: number;
  dated_actions: number;
  full_text_actions: number;
  cited_actions: number;
  citation_coverage_pct: number;
  missing_date_count: number;
  missing_full_text_count: number;
  missing_citation_count: number;
  doc_type_counts: Record<string, number>;
  action_type_counts: Record<string, number>;
  outcome_counts: Record<string, number>;
  citation_activity: EnforcementBetaCitationActivity[];
  top_citations: EnforcementBetaCitationActivity[];
  actions: EnforcementBetaAction[];
  recent_actions: EnforcementBetaAction[];
  max_cell_value: number;
}

export interface EnforcementBetaPayload {
  generated_at: string;
  months: string[];
  totals: {
    combined_actions: number;
    sec_actions: number;
    finra_actions: number;
    cited_actions: number;
    citation_coverage_pct: number;
    latest_action_date: string;
  };
  agencies: {
    sec: EnforcementBetaAgencyPayload;
    finra: EnforcementBetaAgencyPayload;
  };
  combined_actions: EnforcementBetaAction[];
  combined_recent_actions: EnforcementBetaAction[];
  filters: {
    agencies: Agency[];
    doc_types: string[];
    action_types: string[];
    outcomes: string[];
    citations: EnforcementBetaCitation[];
  };
}

function text(value: unknown): string {
  return String(value ?? "").trim();
}

function metadataRecord(doc: CustomDocumentRecord): Record<string, unknown> {
  return (doc.metadata || {}) as unknown as Record<string, unknown>;
}

function metadataBlob(doc: CustomDocumentRecord): string {
  const metadata = metadataRecord(doc);
  return [
    metadata.source_kind,
    metadata.source_family,
    metadata.organization,
    metadata.doc_type,
    metadata.url,
    metadata.pdf_url,
    metadata.source_filename,
    metadata.tags,
    doc.content?.full_text?.slice(0, 800)
  ]
    .map((value) => text(value).toLowerCase())
    .join(" ");
}

function isSecEnforcementDoc(doc: CustomDocumentRecord): boolean {
  const metadata = metadataRecord(doc);
  const sourceKind = text(metadata.source_kind).toLowerCase();
  const sourceFamily = text(metadata.source_family).toLowerCase();
  const url = text(metadata.url).toLowerCase();
  return (
    sourceKind === "sec_enforcement_litigation" ||
    sourceFamily === "sec_enforcement_litigation" ||
    url.includes("/enforcement-litigation/litigation-releases/")
  );
}

function isFinraAwcDoc(doc: CustomDocumentRecord): boolean {
  const metadata = metadataRecord(doc);
  const sourceKind = text(metadata.source_kind).toLowerCase();
  const sourceFamily = text(metadata.source_family).toLowerCase();
  const organization = text(metadata.organization).toLowerCase();
  const docType = text(metadata.doc_type).toLowerCase();
  const url = text(metadata.url).toLowerCase();
  const pdfUrl = text(metadata.pdf_url).toLowerCase();
  const sourceFilename = text(metadata.source_filename).toLowerCase();
  const blob = metadataBlob(doc);

  if (sourceKind === "finra_awc" || sourceFamily === "finra_awc") {
    return true;
  }
  if (organization !== "finra") {
    return false;
  }
  if (docType === "awc" || docType.includes("acceptance, waiver") || docType.includes("acceptance waiver")) {
    return true;
  }
  if (
    (url.includes("/fda_documents/") && url.includes("awc")) ||
    (sourceFilename.includes("awc") && (url.includes("/fda_documents/") || pdfUrl.includes("/fda_documents/")))
  ) {
    return true;
  }
  return blob.includes("document type: awc") || blob.includes("acceptance, waiver & consent");
}

function arrayStrings(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => text(item)).filter(Boolean);
  }
  if (typeof value === "string") {
    return value
      .split(",")
      .map((item) => text(item))
      .filter(Boolean);
  }
  return [];
}

function dedupeByKey<T>(items: T[], keyFn: (item: T) => string): T[] {
  const out: T[] = [];
  const seen = new Set<string>();
  for (const item of items) {
    const key = keyFn(item);
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    out.push(item);
  }
  return out;
}

function buildMonths(): { months: string[]; monthIndex: Map<string, number> } {
  const now = new Date();
  const months: string[] = [];
  for (let i = 17; i >= 0; i -= 1) {
    const d = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() - i, 1));
    months.push(`${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, "0")}`);
  }
  return { months, monthIndex: new Map(months.map((month, index) => [month, index])) };
}

function parseMonth(value: string): string {
  const date = new Date(value.length <= 7 ? `${value}-01T00:00:00Z` : value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}`;
}

function countBy(values: string[]): Record<string, number> {
  return values.reduce<Record<string, number>>((acc, value) => {
    const key = value || "Unknown";
    acc[key] = (acc[key] ?? 0) + 1;
    return acc;
  }, {});
}

function agencyKey(agency: Agency): AgencyKey {
  return agency === "SEC" ? "sec" : "finra";
}

function shortActName(actName: string): string {
  const act = actName.toLowerCase();
  if (act.includes("advisers")) {
    return "Advisers Act";
  }
  if (act.includes("exchange")) {
    return "Exchange Act";
  }
  if (act.includes("investment company")) {
    return "Investment Company Act";
  }
  if (act.includes("securities")) {
    return "Securities Act";
  }
  return "SEC";
}

function normalizeSectionToken(raw: string): string {
  return raw.replace(/\s+/g, "").replace(/[.,;:]+$/g, "");
}

function secLabel(citation: string): string {
  if (SEC_RULE_LABELS[citation]) {
    return SEC_RULE_LABELS[citation];
  }
  if (citation.startsWith("Rule ")) {
    return citation;
  }
  const [act, section] = citation.split(/ (?=\S+$)/);
  return section ? `${act} Section ${section}` : citation;
}

function extractSecCitations(fullText: string): EnforcementBetaCitation[] {
  const citations: EnforcementBetaCitation[] = [];

  SEC_RULE_RE.lastIndex = 0;
  let ruleMatch: RegExpExecArray | null;
  while ((ruleMatch = SEC_RULE_RE.exec(fullText)) !== null) {
    const rule = `Rule ${ruleMatch[1]}`;
    citations.push({ citation: rule, label: secLabel(rule), agency: "SEC" });
  }

  SEC_SECTION_BLOCK_RE.lastIndex = 0;
  let blockMatch: RegExpExecArray | null;
  while ((blockMatch = SEC_SECTION_BLOCK_RE.exec(fullText)) !== null) {
    const block = blockMatch[1];
    const act = shortActName(blockMatch[2]);
    SECTION_TOKEN_RE.lastIndex = 0;
    let sectionMatch: RegExpExecArray | null;
    while ((sectionMatch = SECTION_TOKEN_RE.exec(block)) !== null) {
      const section = normalizeSectionToken(sectionMatch[0]);
      if (/^\d+$/.test(section)) {
        continue;
      }
      const citation = `${act} ${section}`;
      citations.push({ citation, label: secLabel(citation), agency: "SEC" });
    }
  }

  SEC_STANDALONE_SECTION_RE.lastIndex = 0;
  let sectionMatch: RegExpExecArray | null;
  while ((sectionMatch = SEC_STANDALONE_SECTION_RE.exec(fullText)) !== null) {
    const section = normalizeSectionToken(sectionMatch[1]);
    const act = section.startsWith("206") || section === "204(a)" || section === "207" ? "Advisers Act" : "SEC";
    const citation = `${act} ${section}`;
    citations.push({ citation, label: secLabel(citation), agency: "SEC" });
  }

  return dedupeByKey(citations, (citation) => citation.citation);
}

function extractFinraCitations(fullText: string): EnforcementBetaCitation[] {
  const citations: EnforcementBetaCitation[] = [];

  const collectFromBlock = (block: string) => {
    FINRA_RULE_TOKEN_RE.lastIndex = 0;
    let tokenMatch: RegExpExecArray | null;
    while ((tokenMatch = FINRA_RULE_TOKEN_RE.exec(block)) !== null) {
      const rule = tokenMatch[0].toUpperCase();
      citations.push({
        citation: `FINRA Rule ${rule}`,
        label: FINRA_RULE_LABELS[rule] ?? `FINRA Rule ${rule}`,
        agency: "FINRA"
      });
    }
  };

  for (const pattern of [FINRA_RULE_BLOCK_RE, FINRA_VIOLATION_RULE_BLOCK_RE]) {
    pattern.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(fullText)) !== null) {
      collectFromBlock(match[1]);
    }
  }

  return dedupeByKey(citations, (citation) => citation.citation);
}

function inferReleaseNo(metadata: Record<string, unknown>, fullText: string): string {
  const explicit = text(metadata.release_no);
  if (explicit) {
    return explicit;
  }
  const urlMatch = text(metadata.url).match(/\blr[-/](\d{5})\b/i);
  if (urlMatch) {
    return `LR-${urlMatch[1]}`;
  }
  const textMatch = fullText.match(/Litigation Release No\.\s*(\d{5})/i);
  return textMatch ? `LR-${textMatch[1]}` : "";
}

function inferActionType(agency: Agency, metadata: Record<string, unknown>, fullText: string): string {
  const explicit = text(metadata.action_type).toLowerCase();
  if (explicit) {
    return explicit;
  }
  const docType = text(metadata.doc_type).toLowerCase();
  const blob = `${text(metadata.title)} ${fullText.slice(0, 1800)}`.toLowerCase();

  if (agency === "FINRA") {
    if (docType.includes("awc")) {
      return "settlement";
    }
    if (docType.includes("oho") || docType.includes("nac")) {
      return "adjudicated";
    }
    return "disciplinary";
  }
  if (blob.includes("dismiss")) {
    return "dismissal";
  }
  if (blob.includes("final judgment") || blob.includes("judgment")) {
    return "judgment";
  }
  if (blob.includes("settlement") || blob.includes("settled")) {
    return "settlement";
  }
  if (blob.includes("complaint") || blob.includes("charged") || blob.includes("filed charges")) {
    return "filing";
  }
  if (blob.includes("order")) {
    return "order";
  }
  return "unknown";
}

function inferOutcomeStatus(metadata: Record<string, unknown>, fullText: string, actionType: string): string {
  const explicit = text(metadata.outcome_status).toLowerCase();
  if (explicit) {
    return explicit;
  }
  const blob = `${text(metadata.title)} ${fullText.slice(0, 1800)}`.toLowerCase();
  if (["dismissal", "judgment", "settlement"].includes(actionType) || blob.includes("final judgment")) {
    return "resolved";
  }
  if (blob.includes("filed") || blob.includes("charged") || blob.includes("complaint")) {
    return "filed";
  }
  return "unknown";
}

function inferForum(agency: Agency, metadata: Record<string, unknown>, fullText: string): string {
  const explicit = text(metadata.forum);
  if (explicit) {
    return explicit;
  }
  const docType = text(metadata.doc_type);
  const blob = `${docType} ${fullText.slice(0, 1200)}`.toLowerCase();
  if (agency === "FINRA") {
    if (docType.toLowerCase().includes("awc")) {
      return "FINRA AWC";
    }
    if (docType.toLowerCase().includes("oho")) {
      return "FINRA OHO";
    }
    if (docType.toLowerCase().includes("nac")) {
      return "FINRA NAC";
    }
    return "FINRA";
  }
  if (blob.includes("u.s. district court") || blob.includes("district of") || /\b\d+:\d{2}-cv-/.test(blob)) {
    return "Federal court";
  }
  if (blob.includes("administrative proceeding")) {
    return "Administrative proceeding";
  }
  return "SEC";
}

function splitTitleEntities(title: string): string[] {
  const cleaned = title
    .replace(/^securities and exchange commission\s+v\.?\s+/i, "")
    .replace(/\s+/g, " ")
    .trim();
  if (!cleaned) {
    return [];
  }
  return cleaned
    .split(/\s+(?:and|&)\s+|,\s+/i)
    .map((item) => item.trim())
    .filter((item) => item.length > 2)
    .slice(0, 5);
}

function extractSanctions(fullText: string): string[] {
  return dedupeByKey(fullText.match(MONEY_RE) ?? [], (item) => item.toLowerCase()).slice(0, 4);
}

function summarize(doc: CustomDocumentRecord): string {
  const metadata = metadataRecord(doc);
  const explicit = text(metadata.summary);
  if (explicit) {
    return explicit.slice(0, 280);
  }
  const caseSummary = text(metadata.case_summary);
  if (caseSummary) {
    return caseSummary.slice(0, 280);
  }
  const paragraph = (doc.content?.paragraphs || []).find((item) => {
    const value = text(item);
    return value.length >= 90 && !/^u\.s\. securities and exchange commission$/i.test(value);
  });
  return text(paragraph || doc.content?.full_text || "").slice(0, 280);
}

function normalizeAction(doc: CustomDocumentRecord, agency: Agency): EnforcementBetaAction {
  const metadata = metadataRecord(doc);
  const fullText = text(doc.content?.full_text);
  const date = text(metadata.published_date) || text(metadata.date);
  const citations = agency === "SEC" ? extractSecCitations(fullText) : extractFinraCitations(fullText);
  const actionType = inferActionType(agency, metadata, fullText);
  const entities = dedupeByKey(
    [
      ...arrayStrings(metadata.entities),
      ...arrayStrings(metadata.respondents),
      text(metadata.subject_text),
      ...splitTitleEntities(text(metadata.title))
    ],
    (item) => item.toLowerCase()
  ).slice(0, 6);

  return {
    document_id: text(metadata.document_id),
    agency,
    source_kind: text(metadata.source_kind) || (agency === "SEC" ? "sec_enforcement_litigation" : "finra_awc"),
    title: text(metadata.title) || "Untitled enforcement action",
    date,
    month: parseMonth(date),
    doc_type: text(metadata.doc_type) || (agency === "SEC" ? "Litigation Release" : "AWC"),
    url: text(metadata.url),
    release_no: agency === "SEC" ? inferReleaseNo(metadata, fullText) : text(metadata.case_id),
    action_type: actionType,
    forum: inferForum(agency, metadata, fullText),
    outcome_status: inferOutcomeStatus(metadata, fullText, actionType),
    citations,
    entities,
    sanctions: dedupeByKey(
      [...arrayStrings(metadata.sanctions), text(metadata.sanctions_text), ...extractSanctions(fullText)],
      (item) => item.toLowerCase()
    ).filter(Boolean),
    summary: summarize(doc),
    data_quality: {
      has_date: Boolean(date),
      has_full_text: Boolean(fullText),
      has_citations: citations.length > 0
    }
  };
}

function buildCitationActivity(
  agency: Agency,
  actions: EnforcementBetaAction[],
  months: string[],
  monthIndex: Map<string, number>
): EnforcementBetaCitationActivity[] {
  const counts = new Map<string, EnforcementBetaCitationActivity>();
  for (const action of actions) {
    const monthSlot = monthIndex.get(action.month) ?? -1;
    for (const citation of action.citations) {
      const existing = counts.get(citation.citation) ?? {
        ...citation,
        agency,
        total: 0,
        by_month: new Array(months.length).fill(0) as number[]
      };
      existing.total += 1;
      if (monthSlot >= 0) {
        existing.by_month[monthSlot] += 1;
      }
      counts.set(citation.citation, existing);
    }
  }
  return [...counts.values()].sort((a, b) => b.total - a.total || a.citation.localeCompare(b.citation));
}

function buildAgencyPayload(
  agency: Agency,
  docs: CustomDocumentRecord[],
  months: string[],
  monthIndex: Map<string, number>
): EnforcementBetaAgencyPayload {
  const actions = docs
    .map((doc) => normalizeAction(doc, agency))
    .sort((a, b) => parseComparableDate(b.date) - parseComparableDate(a.date));
  const citationActivity = buildCitationActivity(agency, actions, months, monthIndex);
  const maxCellValue = citationActivity.length
    ? Math.max(...citationActivity.flatMap((citation) => citation.by_month), 1)
    : 1;
  const citedActions = actions.filter((action) => action.data_quality.has_citations).length;

  return {
    agency,
    total_actions: actions.length,
    dated_actions: actions.filter((action) => action.data_quality.has_date).length,
    full_text_actions: actions.filter((action) => action.data_quality.has_full_text).length,
    cited_actions: citedActions,
    citation_coverage_pct: actions.length ? Math.round((citedActions / actions.length) * 100) : 0,
    missing_date_count: actions.filter((action) => !action.data_quality.has_date).length,
    missing_full_text_count: actions.filter((action) => !action.data_quality.has_full_text).length,
    missing_citation_count: actions.length - citedActions,
    doc_type_counts: countBy(actions.map((action) => action.doc_type)),
    action_type_counts: countBy(actions.map((action) => action.action_type)),
    outcome_counts: countBy(actions.map((action) => action.outcome_status)),
    citation_activity: citationActivity,
    top_citations: citationActivity.slice(0, 12),
    actions,
    recent_actions: actions.slice(0, 40),
    max_cell_value: maxCellValue
  };
}

function uniqueSorted(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))].sort((a, b) => a.localeCompare(b));
}

export async function GET() {
  const requestId = createRequestId();

  try {
    const corpus = await loadCorpusDocuments();
    const { months, monthIndex } = buildMonths();
    const sec = buildAgencyPayload("SEC", corpus.filter(isSecEnforcementDoc), months, monthIndex);
    const finra = buildAgencyPayload("FINRA", corpus.filter(isFinraAwcDoc), months, monthIndex);
    const combinedActions = [...sec.actions, ...finra.actions].sort(
      (a, b) => parseComparableDate(b.date) - parseComparableDate(a.date)
    );
    const citedActions = sec.cited_actions + finra.cited_actions;
    const totalActions = sec.total_actions + finra.total_actions;
    const citations = dedupeByKey(
      [...sec.citation_activity, ...finra.citation_activity].map(({ citation, label, agency }) => ({ citation, label, agency })),
      (citation) => `${citation.agency}:${citation.citation}`
    ).sort((a, b) => a.agency.localeCompare(b.agency) || a.citation.localeCompare(b.citation));

    const payload: EnforcementBetaPayload = {
      generated_at: new Date().toISOString(),
      months,
      totals: {
        combined_actions: totalActions,
        sec_actions: sec.total_actions,
        finra_actions: finra.total_actions,
        cited_actions: citedActions,
        citation_coverage_pct: totalActions ? Math.round((citedActions / totalActions) * 100) : 0,
        latest_action_date: combinedActions[0]?.date ?? ""
      },
      agencies: { sec, finra },
      combined_actions: combinedActions,
      combined_recent_actions: combinedActions.slice(0, 60),
      filters: {
        agencies: ["SEC", "FINRA"],
        doc_types: uniqueSorted(combinedActions.map((action) => action.doc_type)),
        action_types: uniqueSorted(combinedActions.map((action) => action.action_type)),
        outcomes: uniqueSorted(combinedActions.map((action) => action.outcome_status)),
        citations
      }
    };

    return ok(payload, requestId);
  } catch (error) {
    return fail(
      `Failed to load enforcement beta data: ${error instanceof Error ? error.message : "Unknown error"}`,
      "ENFORCEMENT_BETA_FAILED",
      500,
      requestId
    );
  }
}
