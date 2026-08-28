/**
 * Market breadth, by equal-weight versus cap-weight.
 *
 * The page can say the S&P is down 1.2% but not whether that was 400 stocks
 * falling or three megacaps dragging an otherwise fine tape. Those are opposite
 * conditions and they render identically.
 *
 * Why this measure and not advance/decline: Yahoo, the only quote source this
 * page has, serves no breadth symbols at all - ^ADD, ^ADVN, ^TICK and ^TRIN all
 * 404. And counting up-versus-down across the ~35-name Movers watchlist would
 * be worse than nothing: those are hand-picked megacaps, so a "breadth" reading
 * taken over them measures the very thing breadth exists to distinguish itself
 * from.
 *
 * Equal-weight versus cap-weight is the standard proxy and costs two extra
 * quotes. When RSP beats SPY the average stock is outrunning the index, so the
 * move is broad; when SPY beats RSP, a handful of large names are carrying it.
 *
 * It is a PROXY, not advance/decline, and the UI says so. Nothing here forecasts
 * or advises - it describes what the tape did.
 */

export type BreadthTone = "broad" | "narrow" | "even";

export interface BreadthPair {
  id: string;
  label: string;
  /** Cap-weighted side, e.g. SPY. */
  capSymbol: string;
  capPct: number;
  /** Equal-weighted side, e.g. RSP. */
  equalSymbol: string;
  equalPct: number;
  /** equalPct - capPct, in percentage points. Positive means broad. */
  spreadPp: number;
  tone: BreadthTone;
  /** Plain reading of this pair, e.g. "Index fell; the average stock rose." */
  reading: string;
}

export interface MarketBreadth {
  pairs: BreadthPair[];
  /** One sentence over the set, or "" when there is nothing to say. */
  summary: string;
  /** Small caps vs large caps, when both indices are available. */
  smallVsLarge: { smallPct: number; largePct: number; spreadPp: number } | null;
}

/**
 * Below this the two sides are doing the same thing and calling it "broad" or
 * "narrow" would be reading noise. Roughly a normal day's tracking difference.
 */
export const EVEN_THRESHOLD_PP = 0.25;

function tone(spreadPp: number): BreadthTone {
  if (spreadPp > EVEN_THRESHOLD_PP) return "broad";
  if (spreadPp < -EVEN_THRESHOLD_PP) return "narrow";
  return "even";
}

/**
 * The most informative case is a sign disagreement - the index moved one way
 * while the average constituent moved the other. That is the reading a
 * cap-weighted number alone can never give you, so it gets said outright.
 */
function reading(capPct: number, equalPct: number, spreadTone: BreadthTone): string {
  const capDown = capPct < 0;
  const equalUp = equalPct > 0;

  if (capDown && equalUp) return "Index fell while the average stock rose - a few large names drove the decline.";
  if (!capDown && !equalUp && capPct > 0) return "Index rose while the average stock fell - a few large names carried it.";
  if (spreadTone === "broad") return "The average stock outpaced the index - participation was wide.";
  if (spreadTone === "narrow") return "The index outpaced the average stock - large names led.";
  return "Index and average stock moved together - participation was even.";
}

export function buildBreadthPair(input: {
  id: string;
  label: string;
  capSymbol: string;
  capPct: number | null | undefined;
  equalSymbol: string;
  equalPct: number | null | undefined;
}): BreadthPair | null {
  const { capPct, equalPct } = input;
  if (typeof capPct !== "number" || typeof equalPct !== "number") return null;
  if (!Number.isFinite(capPct) || !Number.isFinite(equalPct)) return null;

  const spreadPp = equalPct - capPct;
  const spreadTone = tone(spreadPp);
  return {
    id: input.id,
    label: input.label,
    capSymbol: input.capSymbol,
    capPct,
    equalSymbol: input.equalSymbol,
    equalPct,
    spreadPp,
    tone: spreadTone,
    reading: reading(capPct, equalPct, spreadTone),
  };
}

export function summarizeBreadth(pairs: BreadthPair[]): string {
  if (!pairs.length) return "";
  const broad = pairs.filter((pair) => pair.tone === "broad");
  const narrow = pairs.filter((pair) => pair.tone === "narrow");

  if (narrow.length && !broad.length) {
    return narrow.length === pairs.length
      ? "Narrow tape: large names are doing the work across the board."
      : "Leaning narrow: large names are carrying part of the tape.";
  }
  if (broad.length && !narrow.length) {
    return broad.length === pairs.length
      ? "Broad tape: the average stock is outpacing the index."
      : "Leaning broad: participation is wider than the index suggests.";
  }
  if (broad.length && narrow.length) return "Mixed: participation differs between the two indices.";
  return "Participation is even - index and average stock are moving together.";
}
