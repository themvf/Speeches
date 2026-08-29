# Rate Transmission: spreads → mortgage rates and corporate bonds (and back)

Status: **all phases shipped 2026-08-29.** Phase 1 (levels, curve tails,
attribution) shipped 2026-08-28; phases 2-3 (pass-through, lead/lag) and the
consolidation into one workspace shipped the next day - see
`docs/macro-rates-credit-consolidation-spec.md` for what changed and where the
result differed from the estimate. Two corrections this document did not
anticipate: pass-through must lag the Treasury change by one week for the
mortgage survey (§5.2 called for it and the first build still missed it), and
corporate lead/lag is impossible from a published spread because the rebuilt
level contains its own base.

## 1. The question

Every borrowing rate in the economy is a Treasury yield plus a spread. When the
30-year mortgage rate moves 40bp, some of that is the 10-year Treasury moving
and some of it is the mortgage spread moving, and those two have completely
different causes and completely different implications. The Macro tab today
shows the pieces — `MORTGAGE30US`, `BAA10Y`, `T10Y2Y`, `DFII10` — as separate
cards that never speak to each other. Nothing on the page answers:

- Of the last quarter's move in mortgage rates, how much was Treasuries and how
  much was the spread?
- Is the front end (policy expectations) or the long tail (term premium) doing
  the work?
- Does a full 25bp move in the 10-year actually reach borrowers, or does the
  spread absorb it?
- **And back the other way:** do credit spreads move before Treasuries, or
  after? MBS convexity hedging and flight-to-quality both run that direction.

This spec covers a **Rate Transmission** panel that answers those, using data
FRED already serves, without adding a database table or an LLM call.

## 2. Data

### 2.1 Series

| Series | What | Source | In repo today |
|---|---|---|---|
| `DGS10` | 10Y Treasury constant maturity, daily | H.15, Federal Reserve | no |
| `DGS2` | 2Y, daily | H.15 | no |
| `DGS3MO` | 3M, daily | H.15 | no |
| `DGS30` | 30Y, daily | H.15 | no |
| `DFII10` | 10Y TIPS real yield, daily | H.15 | yes (`real_yield_10y`) |
| `DFF` | Effective fed funds, daily | H.15 | yes (`effective_fed_funds`) |
| `MORTGAGE30US` | 30Y fixed mortgage, **weekly (Thu)** | Freddie Mac PMMS | yes (`mortgage_rate_30y`) |
| `DBAA` | Moody's Baa corporate yield, daily | Moody's via H.15 | no |
| `DAAA` | Moody's Aaa corporate yield, daily | Moody's via H.15 | no |
| `BAA10Y` | Baa **spread** over 10Y, daily | FRED-computed | yes (`credit_spread_baa`) |
| `AAA10Y` | Aaa spread over 10Y, daily | FRED-computed | no |
| `T10Y2Y` | 10Y−2Y, daily | FRED-computed | yes (`yield_curve_10y2y`) |
| `T10Y3M` | 10Y−3M, daily | FRED-computed | no |

Optional later: `MORTGAGE15US`; `OBMMIC30YF` (Optimal Blue, **daily** mortgage
rate — would fix the PMMS timing problem in §5.2, licence unverified).

### 2.2 Licensing — verified 2026-08-28

CLAUDE.md's rule stands and nothing in the code enforces it: `/api/market/*` is
public with no auth, so a *pre-approval required* series cannot ship here. Known
from prior sessions: `BAA10Y` and `NFCICREDIT` are **citation required**
(satisfied by the tab's existing FRED attribution), `DFII10` is **public
domain**, and ICE BofA OAS (`BAMLH0A0HYM2`, `BAMLC0A0CM`) is **pre-approval
required and unusable** — which is why this design routes around index OAS
entirely and uses Moody's Baa/Aaa instead.

The series pages were checked directly:

- `DGS10`, `DGS2`, `DGS3MO`, and `DGS30` are **Public Domain: Citation
  Requested** and are approved for the public route.
- `T10Y3M` is **Copyrighted: Citation Required** and is approved with a source
  link, though Phase 1 computes its requested curve segments from Treasury
  levels rather than fetching this series.
- `DBAA`, `DAAA`, and `AAA10Y` display **Copyrighted: Citation Required**, but
  their notes separately state that Moody's information may not be reproduced,
  stored, or redistributed without prior written consent. They are therefore
  **blocked** for the public route. Phase 1 renders the corporate rows as
  unavailable pending a licensed corporate-yield source.
- `OBMMIC30YF` remains unknown and blocked until checked.

### 2.3 Why these series do NOT go into `FRED_MACRO_DEFINITIONS`

Tempting, since that list already fetches and caches FRED series with history.
Three reasons not to:

1. **Payload.** Definitions are served whole to the browser in `/api/market/macro`.
   Five years of daily history is ~1,300 points ≈ 50KB of JSON per series; six
   new series would add ~300KB to a payload every Macro tab visit pays for.
   A dedicated route returns computed results plus a downsampled history — a few KB.
2. **Cards.** `IndicatorGrid` renders every indicator in a group. Six more
   definitions means six more cards in Financial Conditions, which already
   carries nine plus the curve.
3. **Calendar coupling.** Every definition needs a `releaseId`, and
   `fred-calendar.test.ts` then demands a pinned release name and either a
   pinned ET time or a place on the declared-gap list. A dedicated route needs
   no `releaseId` at all. (For the record: H.15 is release 18 and Interest Rate
   Spreads is release 304, both already in `DAILY_REFRESH_RELEASE_IDS`, so
   adding them would have been calendar-safe — it just isn't necessary.)

Consequence: `MarketMacroIndicatorId`, `SIGNALS`, and the calendar are all
untouched by this feature.

## 3. The three analyses

### 3.1 Attribution — an exact identity, no model (build this first)

For a window (1M / 3M / 6M / 12M / YTD):

```
Δ(30Y mortgage)  =  Δ(10Y Treasury)  +  Δ(mortgage spread)
Δ(Baa yield)     =  Δ(10Y Treasury)  +  Δ(BAA10Y)
Δ(Aaa yield)     =  Δ(10Y Treasury)  +  Δ(AAA10Y)
```

Reported in basis points with each leg's share of the absolute move. This is an
accounting identity, not a fit — there is no coefficient to be wrong about, and
it answers the headline question directly ("mortgage rates are up 34bp this
quarter: 51bp from Treasuries, −17bp from a narrowing spread"). Highest value
per unit of risk in the whole spec.

Same decomposition on the curve itself, splitting the tails:

```
short tail  =  2Y − 3M          policy path priced into the front end
belly       =  10Y − 2Y         (T10Y2Y — the existing headline card)
long tail   =  30Y − 10Y        term premium / duration demand
policy gap  =  2Y − effective fed funds
```

### 3.2 Pass-through beta — rolling regression

Does a move in the 10-year actually reach the borrower?

- **Mortgage:** OLS of weekly Δ`MORTGAGE30US` on weekly Δ`DGS10`, 52-week
  rolling window. Report β, R², n.
- **Corporate:** OLS of daily Δ`DBAA` on daily Δ`DGS10`, 60-business-day
  rolling window. Same for `DAAA`.

Reading convention to put on screen: β near 1 with high R² means the move passes
through roughly intact; β well below 1, or a low R², means the spread is
absorbing it. Suppress the whole stat when n is below 30, the same instinct as
`percentileContext`'s n<12 guard — a beta off twelve points is noise with a
decimal point.

Report the standard error alongside β. A β of 0.6 ± 0.35 is not a finding.

### 3.3 Lead/lag — the "vice versa"

Cross-correlate first differences at lags −10…+10 business days and report the
lag with the strongest correlation:

- Δ`DBAA` vs Δ`DGS10` (daily)
- Δ`DGS2` vs Δ`DBAA` (daily)
- Δ`MORTGAGE30US` vs Δ`DGS10` (weekly, lags −4…+4)

Guardrails: name a lead only when |r| ≥ 0.2 **and** it beats lag-0 by a margin;
otherwise print "no clear lead". Label it correlation, never causation.

**The trap that will silently ruin this if ignored:** do not cross-correlate a
*spread* against its own subtrahend. `BAA10Y` is defined as `DBAA − DGS10`, so
regressing Δ`BAA10Y` on Δ`DGS10` carries a built-in −1 loading and will always
look like "Treasuries and spreads move opposite", which is arithmetic, not
economics. Lead/lag runs on **yield levels** (`DBAA`, `DAAA`, `MORTGAGE30US`,
`DGS10`); spreads appear only in §3.1's identity, where the mechanical
relationship is the point.

What the panel is actually looking for here, worth naming in the copy: the
long-end/mortgage feedback (rising mortgage rates extend MBS duration, hedgers
sell Treasuries, the long tail steepens further) and flight-to-quality (credit
widens, Treasuries rally, spread widens further). The panel can show
co-movement and its timing. It cannot identify the mechanism, and the copy must
not pretend otherwise.

## 4. Architecture

### 4.1 New route — `GET /api/market/rate-transmission`

`runtime = "nodejs"`, `revalidate = 3600` (daily series update ~4:15pm ET;
PMMS Thursdays at 12:00 ET — an hour of cache is plenty and cheaper than the
Macro route's 15 minutes).

Fetches each series' observations once, computes everything server-side, returns:

```ts
interface RateTransmissionData {
  asOf: string;
  levels: {           // current decomposition
    mortgage30y: RateDecomposition | null;   // rate, base (10Y), spread, spread percentile
    baa:         RateDecomposition | null;
    aaa:         RateDecomposition | null;
  };
  curve: { shortTail, belly, longTail, policyGap };  // pp, each with percentile
  attribution: AttributionWindow[];        // 1M/3M/6M/12M × {rate, base, spread} in bp
  passThrough: PassThrough[];              // {target, beta, stderr, r2, n, window}
  leadLag: LeadLagPair[];                  // {a, b, bestLagDays, correlation, verdict}
  history: { seriesId: string; label: string; points: MarketMacroPoint[] }[];  // downsampled spreads for plotting
  warnings: string[];                      // series that failed to load
  generatedAt: string;
}
```

`warnings` follows the convention `/api/metrics` and `/api/market/attention`
already use: degrade visibly, never silently.

### 4.2 Shared FRED plumbing

`fredUrl()` and `fetchFredJson()` in `lib/server/fred-macro.ts` are currently
module-private. Export a small `fetchFredSeriesPoints(seriesId, { units, limit })`
from that file and have the new route use it, rather than growing a second copy
of FRED auth/error handling. `parseFredObservations` is already exported and
already drops FRED's `"."` missing-value rows.

### 4.3 Pure analysis module — `lib/rate-transmission.ts`

Same shape as `lib/macro-context.ts`: pure functions, no fetching, unit-tested,
imported by both the route and the component.

```
alignAsOf(base, target)        // §5.1
decompose(rate, base)          // level identity + percentile
attributeWindow(series, days)  // §3.1
rollingOls(x, y, window)       // β, stderr, R², n
crossCorrelate(x, y, maxLag)   // §3.3
```

### 4.4 Failure behaviour

`Promise.allSettled` across series. A single FRED failure drops that series'
rows and adds a `warnings` entry; the panel renders what survived. Route-level
failure returns the standard `fail()` envelope and the panel renders nothing
while the rest of the Macro tab is unaffected.

Deploy-skew rule (CLAUDE.md, both directions): the component treats every top
level field as possibly absent, so a browser on an older bundle receiving a
newer payload — or a newer bundle receiving an older one — loses a section, not
the tab.

## 5. Alignment — where this gets quietly wrong

### 5.1 Weekly vs daily

`MORTGAGE30US` is weekly (Thursday). `DGS10` is daily, business days only, with
holes on holidays. **Zipping the two arrays by index is wrong** and will produce
a plausible-looking, systematically misaligned regression. Every pairing goes
through an as-of join: for each PMMS date, take the last `DGS10` observation on
or before that date. This is the same class of bug as the `published_date` TEXT
comparison in CLAUDE.md — it does not throw, it just returns a wrong number that
looks right.

### 5.2 PMMS measures a different week than the Treasury print

Freddie Mac's survey reflects rates lenders quoted earlier in the week, so the
published Thursday number mechanically lags same-day Treasuries by several days.
Two consequences that must be stated on screen, not buried here:

- Same-week pass-through β is **biased low**; the 1-week-lagged β is the more
  honest number. Report both.
- §3.3 will show the 10-year "leading" mortgage rates by roughly a week. Part of
  that is survey timing, not transmission.

A daily mortgage series (`OBMMIC30YF`) would remove this entirely — worth
revisiting if its licence permits.

### 5.3 One spread, one number

The panel's Baa spread must be FRED's `BAA10Y` — the exact number the existing
Financial Conditions card shows — not a second one recomputed as `DBAA − DGS10`,
which will differ by a day's staleness. Same discipline that removed the
duplicate 2s10s by passing FRED's `T10Y2Y` into `YieldCurve` instead of letting
the plot derive its own from Treasury XML. `DBAA`/`DAAA` are fetched for the
*levels* in §3.3 only.

## 6. UI

A new collapsible section on the Macro tab, a sibling of the `GROUPS` blocks
rather than a member of them (it is not indicator-driven), placed directly after
Financial Conditions. It does not belong inside either Financial Conditions or
Housing because it spans both.

```
▸ Rate Transmission   How Treasury moves reach mortgages and corporate borrowers

  ┌ What borrowers pay ─────────────────────────────────────┐
  │  30Y mortgage   6.42%  =  10Y 4.28%  +  spread 2.14pp   │
  │                                        higher than 71%  │
  │                                        of readings …    │
  │  Baa corporate  5.92%  =  10Y 4.28%  +  spread 1.64pp   │
  │  Aaa corporate  5.31%  =  10Y 4.28%  +  spread 1.03pp   │
  └─────────────────────────────────────────────────────────┘

  ┌ Curve tails ────────────────────────────────────────────┐
  │  Short (2Y−3M)   Belly (10Y−2Y)   Long (30Y−10Y)        │
  └─────────────────────────────────────────────────────────┘

  ┌ What moved it  [1M][3M][6M][12M] ───────────────────────┐
  │  30Y mortgage   +34bp  ██████████░░░  Treasury +51bp    │
  │                                       Spread   −17bp    │
  └─────────────────────────────────────────────────────────┘

  ┌ Pass-through (52w rolling) ─────────────────────────────┐
  │  Mortgage vs 10Y   β 0.82 ± 0.11   R² 0.74   n 52       │
  │  Baa vs 10Y        β 0.94 ± 0.06   R² 0.88   n 60       │
  └─────────────────────────────────────────────────────────┘

  ┌ Timing ─────────────────────────────────────────────────┐
  │  Baa yield vs 10Y     strongest at lag 0    r 0.71      │
  │  Mortgage vs 10Y      strongest at +1 week  r 0.83      │
  │    ↳ partly survey timing — see note                    │
  └─────────────────────────────────────────────────────────┘
```

Hand-rolled SVG for the attribution bars and the spread history, matching every
other chart in this app. Percentile strings come from the existing
`percentileContext`, which already names its window — mandatory here, since the
mortgage spread's daily history only runs back ~17 months at `limit: 365` and
"a multi-year low" off that would be exactly the error that function exists to
prevent. Fetch these series with a longer `limit` (5y) precisely so the
percentile means something; the route downsamples before returning.

## 7. Copy rules

Everything on this panel describes what happened. `macro-context.test.ts`
already asserts no condition string matches
`/\b(will|should|buy|sell|expect|forecast|predict|recommend)\b/i` — extend the
same assertion over every string this module emits.

Beyond that list, prefer "accounted for", "co-moved with", "passed through" over
"drove" or "caused". §3.1 is an identity and may speak plainly. §3.2 and §3.3
are estimates and must carry their n, their error, and the correlation caveat.
Keep the existing "research context only" posture.

## 8. Tests

- `lib/rate-transmission.test.ts` (`node --experimental-strip-types --test`,
  new `test:rate-transmission` script alongside the existing per-module ones):
  - as-of alignment picks the last observation on or before the target date,
    skips holiday holes, and **fails a fixture built to pass under naive
    index-zipping** — the misalignment bug has to be caught by a test that
    would go green without the fix;
  - the attribution identity closes: base + spread legs sum to the total move,
    to the basis point, on generated fixtures;
  - OLS β/stderr/R² against a hand-computed fixture, and the n<30 suppression;
  - cross-correlation recovers a known injected lag, and returns "no clear
    lead" on white noise;
  - no emitted string matches the forbidden-language pattern.
- `lib/server/fred-macro.test.ts`: assert the transmission series list is
  non-empty and every entry carries a licence note in its config comment.
- `tsc --noEmit`, `eslint --max-warnings 0`, `next build`.
- Live check after deploy: FRED is unreachable from the dev sandbox, so the
  first real end-to-end run is on Vercel. Confirm `warnings` is empty and the
  Baa spread on the panel equals the Baa spread on the Financial Conditions
  card to the hundredth.

## 9. Phasing

| Phase | Scope | Ships |
|---|---|---|
| 0 | Verify licences (§2.2) from a machine that can reach FRED | a go/no-go per series |
| 1 | Route + `lib/rate-transmission.ts` + §3.1 attribution + levels/curve-tail panel | the headline answer, zero estimation risk |
| 2 | §3.2 pass-through betas | "does a Treasury move reach borrowers" |
| 3 | §3.3 lead/lag | the "vice versa" |
| 4 | Optional: daily mortgage series if licensing allows; 15Y mortgage; a spread-history chart with recession shading | |

Phase 1 is worth shipping alone. Phases 2 and 3 are the ones that can be subtly
wrong, and each carries its own guardrails above.

## 10. Decisions for the user

1. **Placement** — own collapsible section after Financial Conditions (assumed
   above), or fold into Financial Conditions next to the yield curve?
2. **History depth** — 5 years of daily history makes the percentiles
   meaningful and costs a handful of extra FRED requests per hour of cache.
   Longer (10y, covering 2013 taper and 2020) is possible at no UI cost since
   the route downsamples. Preference?
3. **High yield** — deliberately absent. The natural series (ICE BofA HY OAS)
   is licence-blocked for a public endpoint, and Moody's stops at Baa. So this
   panel covers investment grade only unless a licensed HY source is added.
4. **JIRA** — CLAUDE.md requires a SEC ticket per initiative; `~/.jira_token`
   does not exist in this environment, so none was created. This would be
   epic-shaped (one child per phase).
