# Rates & Credit Intelligence Strategy

## Objective

Add a dedicated Rates & Credit workspace to the Market → Macro page. The workspace should interpret relationships across rates and credit rather than display a larger collection of independent market-data cards.

It should answer five questions:

1. What changed?
2. What kind of rates move is occurring?
3. Is credit confirming or contradicting the move?
4. Where is stress accumulating?
5. What should be monitored next?

## Product Architecture

### Rates & Credit Pulse

The top of the workspace should summarize:

- Rates regime: tightening, easing, mixed, or range-bound.
- Curve regime: bull steepener, bull flattener, bear steepener, or bear flattener.
- Credit regime: risk-on, neutral, deteriorating, or stressed.
- Mortgage regime: easing, affordability pressure, or collateral stress.
- Overall financial condition: benign, late-cycle, tightening, or dislocation.
- The two to four largest observable drivers of the current classification.

Every conclusion must expose its supporting observations. A signal such as `Credit deteriorating` should explain the spread move, quality-tier dispersion, breadth, and lookback window that produced it.

### Treasury And Policy-Rate Complex

Track the full rate structure:

- Effective fed funds and SOFR.
- Treasury curve: 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, and 30Y.
- Curve now versus one week, one month, and one year ago.
- 2s10s, 3m10y, and 5s30s slopes.
- Five-, ten-, and thirty-year real yields.
- Five- and ten-year breakeven inflation.
- A term-premium proxy.
- Treasury volatility when a reliable source is available.

Classify the curve move automatically. The useful conclusion is not simply `10Y +12 bp`; it is whether the market is bear steepening, bear flattening, bull steepening, or bull flattening and whether real rates or inflation compensation led the move.

### Corporate Credit Spread Matrix

Track:

- Broad investment-grade option-adjusted spread.
- AAA, AA, A, and BBB option-adjusted spreads.
- Broad high-yield option-adjusted spread.
- BB, B, and CCC-and-lower option-adjusted spreads.
- HY minus IG and CCC minus BB quality differentials.
- The percentage of tracked credit segments widening.

For every series show the current level, one-day, one-week, one-month, and three-month change, historical percentile, z-score, and trend. Begin retaining daily internal snapshots because upstream public series may provide only a rolling history.

### Rating-Agency Intelligence

Normalize public rating actions from S&P, Moody's, Fitch, and other relevant NRSROs. Measure:

- Upgrade and downgrade counts.
- Downgrade-to-upgrade ratio.
- Positive and negative outlook changes.
- Watchlist and CreditWatch changes.
- Fallen angels and rising stars.
- Rating drift by sector.
- Debt-weighted rating momentum.
- Largest actions by debt outstanding.
- Cross-agency rating disagreement.

Use a normalized notch scale so equivalent ratings from different agencies can be compared.

### Mortgage And Collateral Channel

Keep three distinct concepts separate:

- Consumer borrowing: primary 30Y mortgage rate and mortgage-rate-minus-10Y spread.
- Agency MBS valuation: current-coupon or TBA yield, MBS OAS, and the mortgage basis.
- Collateral performance: delinquencies, foreclosures, prepayments, applications, and refinancing incentive.

The consumer mortgage rate is an affordability measure; it is not a substitute for MBS OAS or the mortgage basis.

### CDS Layer

Treat CDS as a licensed-data module:

- CDX IG and CDX HY.
- Cash-CDS basis.
- Single-name CDS for important issuers.
- Sector median CDS.
- Largest daily wideners.
- Implied hazard rates or default probabilities.

If reliable CDS data is not connected, label the module unavailable. Do not relabel cash spreads, bond ETFs, or other proxies as CDS.

## Intelligence Layer

Build explainable derived signals:

- **Rates pressure:** policy rate, nominal yields, real yields, curve movement, and rate volatility.
- **Credit stress:** IG/HY spread level, speed of change, historical percentile, quality decompression, and breadth.
- **Rating momentum:** downgrade intensity, outlook changes, fallen angels, and debt weighting.
- **Mortgage pressure:** mortgage rate, mortgage-to-Treasury spread, MBS basis, and collateral performance.
- **Composite financial-conditions regime:** the combined state of the preceding signals.
- **Divergence alerts:** cases such as equity strength accompanied by widening HY or CDS spreads.

Each score must include an explanation view with the observations, transformations, thresholds, and contribution weights used to derive it.

## Data-Source Hierarchy

### Public First

- U.S. Treasury: official yield curve and auction data.
- FRED: policy rates, nominal and real Treasury yields, breakevens, mortgage rates, and financial-condition indices.
- FINRA TRACE: corporate, agency, ABS, CMO, MBS, and TBA trade activity, subject to FINRA's data terms.
- SEC/NRSRO disclosures: structured rating histories and rating-agency oversight records.
- SEC filings: issuer debt, interest expense, liquidity, and maturity disclosures.

### Licensed When Available

- CDX and single-name CDS.
- Evaluated bond prices and issuer-level OAS.
- Agency MBS current-coupon and OAS analytics.
- Comprehensive real-time rating feeds.
- Treasury and swap volatility.
- ICE BofA credit spread series distributed through FRED when the deployment's use and redistribution rights permit it. FRED access alone does not grant public redistribution rights.

Every displayed observation must carry source, observation date, retrieval time, frequency, and freshness status.

## Delivery Phases

### Phase 1 — Public-Data Rates And Spreads

- Complete nominal Treasury curve.
- Five-, ten-, and thirty-year real yields.
- Broad IG and HY OAS when an authorized internal-use or redistribution entitlement is configured.
- IG and HY rating-quality buckets under the same entitlement gate.
- Curve-move classification.
- Spread percentiles and z-scores.
- Rates, curve, credit, and composite Pulse signals.
- Dedicated Rates & Credit section on the Macro page.
- Durable daily Neon snapshots, merged with the current source response before historical statistics are calculated.

### Phase 2 — Market Depth, Ratings, And Mortgages

- FINRA TRACE activity and liquidity breadth.
- Most-active and unusually moving corporate bonds.
- MBS, CMO, ABS, and TBA activity.
- Rating-action ingestion and normalized rating momentum.
- Mortgage basis and collateral-performance panels.
- Durable daily snapshot storage.

### Phase 3 — Issuer And Licensed Intelligence

- Issuer maturity walls, leverage, coverage, rating path, and bond dispersion.
- Sector exposure maps and refinancing-risk screens.
- CDX and single-name CDS.
- Cash-CDS basis.
- Evaluated bond pricing and MBS OAS.
- Cross-asset divergence and issuer-level alerts.

## Acceptance Principles

- Lead with interpretation, then reveal the underlying observations.
- Separate levels, changes, valuation, breadth, and momentum.
- Express spread changes in basis points and yields in percent.
- Do not compare data with incompatible observation dates without a freshness warning.
- Missing or stale data must degrade visibly and must not silently become zero.
- A derived regime must be reproducible from the response payload.
- Public, proprietary, and proxy data must be clearly distinguished.
- Avoid presenting a model score as a fact; show its drivers and confidence.
