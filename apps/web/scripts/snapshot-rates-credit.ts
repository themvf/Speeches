import { fetchRatesCreditData, RATES_CREDIT_DEFINITIONS } from "../lib/server/rates-credit.ts";
import { loadRatesCreditHistory, persistRatesCreditSnapshots } from "../lib/server/rates-credit-store.ts";

const history = await loadRatesCreditHistory(RATES_CREDIT_DEFINITIONS.map((definition) => definition.seriesId));
const data = await fetchRatesCreditData(undefined, undefined, history);
const rows = await persistRatesCreditSnapshots(data);

console.log(JSON.stringify({
  persistedRows: rows,
  treasurySeries: data.treasuryCurve.length,
  realYieldSeries: data.realYields.length,
  investmentGradeSeries: data.investmentGrade.length,
  highYieldSeries: data.highYield.length,
  creditDataStatus: data.creditDataStatus,
  warnings: data.warnings,
  generatedAt: data.generatedAt,
}, null, 2));
