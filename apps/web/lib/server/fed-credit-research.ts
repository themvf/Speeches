export const FED_CREDIT_RESEARCH_URL = "https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv";
export const FED_CREDIT_RESEARCH_SOURCE_URL = "https://www.federalreserve.gov/econres/notes/feds-notes/updating-the-recession-risk-and-the-excess-bond-premium-20161006.html";
export const FED_CREDIT_RESEARCH_CACHE_SECONDS = 6 * 60 * 60;

export interface FedCreditResearchPoint {
  date: string;
  corporateSpread: number;
  excessBondPremium: number;
  defaultRiskComponent: number;
  recessionProbability: number;
}

function isoDate(value: string): string | null {
  const match = value.trim().match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
  if (!match) return null;
  const [, month, day, year] = match;
  return `${year}-${month.padStart(2, "0")}-${day.padStart(2, "0")}`;
}

export function parseFedCreditResearchCsv(csv: string): FedCreditResearchPoint[] {
  return csv.trim().split(/\r?\n/).slice(1).flatMap((line) => {
    const [rawDate, rawSpread, rawEbp, rawProbability] = line.split(",");
    const date = isoDate(rawDate ?? "");
    const corporateSpread = Number(rawSpread);
    const excessBondPremium = Number(rawEbp);
    const recessionProbability = Number(rawProbability);
    if (!date || !Number.isFinite(corporateSpread) || !Number.isFinite(excessBondPremium) || !Number.isFinite(recessionProbability)) return [];
    return [{
      date,
      corporateSpread,
      excessBondPremium,
      defaultRiskComponent: corporateSpread - excessBondPremium,
      recessionProbability,
    }];
  }).sort((left, right) => left.date.localeCompare(right.date));
}

export async function fetchFedCreditResearch(): Promise<FedCreditResearchPoint[]> {
  const response = await fetch(FED_CREDIT_RESEARCH_URL, {
    headers: {
      Accept: "text/csv,*/*",
      "User-Agent": "SEC-Speeches-Market/1.0 (+https://github.com/themvf/Speeches)",
    },
    next: { revalidate: FED_CREDIT_RESEARCH_CACHE_SECONDS },
    signal: AbortSignal.timeout(10_000),
  });
  if (!response.ok) throw new Error(`Federal Reserve credit research returned HTTP ${response.status}`);
  const points = parseFedCreditResearchCsv(await response.text());
  if (!points.length) throw new Error("Federal Reserve credit research returned no observations.");
  return points;
}
