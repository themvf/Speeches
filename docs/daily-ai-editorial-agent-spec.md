# Daily AI Editorial Agent — Product and Implementation Specification

**Status:** Proposed
**Owner:** Josh
**Created:** 2026-08-30
**Target:** Existing `sec_speeches` intelligence platform
**Working name:** Daily AI Column Editor

## 1. Executive summary

Build a nightly editorial workflow that turns the day's captured AI-related news into a source-backed editorial package for human authorship and publication on Medium or another long-form platform.

The system must not operate as an autonomous article publisher or generic news-roundup generator. Its primary job is to identify the strongest non-obvious story angle, organize the supporting evidence, expose uncertainty, and prepare an outline that a human author can substantially shape. An optional rough draft may be generated, but it must remain clearly labeled as AI-generated working material and must never be published automatically.

The workflow will reuse the application's existing news ingestion, AI topic taxonomy, article analysis, deduplication patterns, model-provider configuration, and Neon persistence. A new editorial layer will sit beside—not replace—the existing daily recap feature.

The recommended production schedule is **9:00 p.m. America/New_York** each day. Scheduling must use the IANA timezone name rather than a fixed `EST` offset so daylight-saving changes are handled correctly.

The user-facing home for this feature will be **Briefings → Scheduled Briefings** at `/briefings/scheduled`. This page will own briefing enable/disable controls, provider selection, manual runs, run status, stored daily outputs, and side-by-side provider comparison. The existing `/briefings` custom briefing builder remains available as **Briefings → Custom Builder**.

## 2. Problem statement

The platform already captures and analyzes a large volume of financial, regulatory, technology, and AI-related news. The existing daily recap answers, “What happened in this topic?” It does not answer the editorial questions required for an original article:

- What is the most consequential development?
- What connects several apparently separate events?
- What is the defensible, non-obvious thesis?
- Why is this author particularly qualified to make that argument?
- Which claims are facts, interpretations, predictions, or unresolved questions?
- Is the result strong enough to publish, or should the system recommend skipping the day?

A direct “summarize today’s AI news as a Medium article” prompt would tend to produce derivative roundup content and weak human authorship. The new workflow must instead produce an evidence-backed editorial decision and a human-ready writing package.

## 3. Product goals

### 3.1 Primary goals

1. Produce one high-quality editorial package from the previous 24 hours of AI-related coverage.
2. Prefer a single coherent thesis over a list of unrelated stories.
3. Preserve direct links between factual claims and captured sources.
4. Clearly distinguish sourced facts, model inference, author judgment, and prediction.
5. Create a repeatable human review workflow with no automatic publication.
6. Learn from the author’s edits and outcomes without initially requiring model fine-tuning.
7. Permit a valid `no_publish` outcome when no angle clears the quality threshold.

### 3.2 Secondary goals

1. Reduce the time required to go from daily news review to a publishable article.
2. Create a reusable library of successful angles, structures, edits, and editorial feedback.
3. Support future destinations such as a newsletter, company blog, or LinkedIn without coupling the core workflow to Medium.
4. Provide enough observability to understand why a run selected—or rejected—an angle.
5. Support multiple independently enabled scheduled briefings through one reusable control surface.
6. Run OpenAI and DeepSeek against the exact same frozen news set and compare their outputs without source or prompt drift.

## 4. Non-goals

The initial release will not:

- Publish directly to Medium or any other public platform.
- Attempt to evade AI-content detection or disclosure requirements.
- Optimize solely for clicks, read time, or engagement.
- Generate generic link roundups.
- Invent first-person experiences, interviews, quotations, credentials, or opinions for the author.
- Fine-tune a model before sufficient edited examples exist.
- Replace the existing `daily_recaps` product.
- Guarantee that one article will be publishable every day.

## 5. Editorial positioning

### 5.1 Provisional audience

The default audience is professionals who follow the intersection of:

- Artificial intelligence
- Financial markets and financial services
- Regulation, enforcement, and policy
- Enterprise strategy, governance, and operational risk

This positioning should be confirmed before implementation. The system’s value will come from applying this specific lens to AI news rather than competing with general technology-news summaries.

### 5.2 Required article characteristics

Every recommended angle must have:

- One sentence stating the central thesis.
- A clear “why now.”
- A defined reader and reader benefit.
- At least two independent supporting sources, unless the story is based on one authoritative primary source.
- At least one original analytical contribution beyond source summary.
- A counterargument, limitation, or uncertainty.
- A concrete “what to watch next.”
- A reason the proposed article is not merely a recap or link roundup.

### 5.3 Medium policy boundary

The workflow must be designed around Medium’s published quality and AI-content rules rather than attempts to “game” an algorithm.

- Medium treats AI-generated writing as writing where most content was produced by an AI program with little substantive editing.
- AI-generated writing is not eligible for the Medium Partner Program paywall.
- Generated or materially AI-assisted text requires disclosure under Medium’s current policy.
- Outline assistance, research assistance, fact-checking, spelling, and grammar assistance are treated differently from generated article prose.
- Generic summaries, link roundups, and derivative content are weak candidates for General Distribution.
- Strong candidates emphasize human experience or judgment, original perspective, reader value, accurate sourcing, craftsmanship, and representative titles.

Policy references:

- <https://help.medium.com/hc/en-us/articles/22576852947223-Artificial-Intelligence-AI-content-policy>
- <https://help.medium.com/hc/en-us/articles/360006362473-Medium-s-Distribution-Guidelines-How-curators-review-stories-for-Boost-General-and-Network-Distribution>

These links and the policy snapshot date must be stored in the editorial configuration so the guidance can be reviewed periodically.

## 6. User workflow

### 6.1 Nightly automated flow

1. At 9:00 p.m. America/New_York, load all enabled scheduled-briefing definitions.
2. Start one idempotent run per enabled briefing for the local editorial date.
3. Query eligible AI-related items captured since the previous successful cutoff.
4. Exclude blocked, invalid, duplicate, unsupported, and insufficiently sourced items.
5. Freeze one normalized source snapshot for the run.
6. Cluster repeated and syndicated coverage into underlying developments.
7. For every enabled provider, submit the same source snapshot, editorial configuration, output schema, and prompt version.
8. Score developments and generate up to three candidate angles per provider.
9. Run editorial-quality and factual-support checks independently for every provider output.
10. Select the strongest angle per output or return `no_publish`.
11. Save each provider output beneath the shared run, including model metadata, usage, latency, validation results, and errors.
12. Notify the author after all selected providers reach a terminal state.

### 6.2 Human review flow

1. Open **Briefings → Scheduled Briefings** and select the latest run.
2. If comparison mode is enabled, review the provider outputs side by side or use blind-review mode.
3. Rate each output before revealing its provider and model when using blind review.
4. Select the preferred provider output for the run; do not merge outputs automatically.
5. Review its three candidate angles and model recommendation.
6. Approve an angle, choose another candidate, request regeneration, or mark the run `no_publish`.
7. Add the author’s unique contribution:
   - Personal observation or experience
   - Domain judgment
   - Original analogy or framework
   - Disagreement with the proposed thesis
   - A practical implication for the target reader
8. Generate or revise the outline.
9. Optionally request a clearly labeled rough draft from the selected provider.
10. Edit the article outside or inside the application.
11. Run a final source, claim, disclosure, and style review.
12. Copy the final article into Medium manually.
13. Record the published URL and, when available, performance metrics and lessons.

### 6.3 Scheduled-briefing controls

Each scheduled briefing is represented by a configuration card containing:

- Briefing name and description
- Master `Enabled` toggle
- Schedule and timezone
- Topic/source scope
- OpenAI toggle
- DeepSeek toggle
- Model selected for each provider
- Comparison/blind-review toggle
- Rough-draft toggle
- Notification toggle
- `Run now` action
- Last-run status and timestamp
- Next scheduled-run timestamp

Turning a briefing off prevents future scheduled runs but does not delete its settings, history, outputs, feedback, or published URLs. Manual `Run now` remains available when the briefing is disabled, but it must require an explicit user action. At least one provider must be enabled before a run can start.

The initial definition will be **Daily AI Editorial**. The underlying settings model should support future definitions such as Daily Regulatory Recap, Weekly Enforcement Brief, or Morning Market Brief without requiring a new settings table for each product.

## 7. Source eligibility and collection

### 7.1 Initial source set

The MVP will use stored RSS news records and their associated analysis:

- `rss_articles`
- `rss_article_analysis`
- `rss_topic_rules`, specifically `AI_TECH`
- Feed metadata from `rss_feeds`

The implementation must reuse the existing ingestion-policy checks and active AI topic rule. It must not maintain a second, drifting AI keyword list inside the editorial code.

### 7.2 Follow-on source set

After MVP validation, add enriched documents from the Neon `documents` mirror, including relevant NewsAPI, Bloomberg-public, Substack, official statements, speeches, and other captured source kinds. This phase must normalize RSS and document records into one editorial-source contract.

### 7.3 Time window

- Default window: since the previous successful editorial cutoff through the current run start.
- First-run fallback: previous 24 hours.
- Store `window_start` and `window_end` on every run.
- Use `America/New_York` to derive the editorial date.
- Use UTC timestamps for storage and database comparison.
- All provider outputs in a run must use the identical saved source snapshot.
- A rerun for the same editorial date and provider must reuse the same saved source snapshot unless the user explicitly requests a source refresh for the entire run.
- Refreshing sources invalidates comparability with earlier provider outputs. The UI must require confirmation, create a new snapshot revision, and clearly mark outputs produced from different revisions as not directly comparable.

### 7.4 Required source fields

Each normalized source must contain:

```ts
type EditorialSource = {
  sourceId: string;
  sourceType: "rss_article" | "document";
  sourceKind: string;
  feedKey?: string;
  publisher: string;
  title: string;
  description: string;
  url: string;
  author?: string;
  publishedAt: string;
  fetchedAt: string;
  analysisStatus?: string;
  thesis?: string;
  whyItMatters: string[];
  riskSignals: string[];
  entities: string[];
  keywords: string[];
  topics: string[];
  toneLabel?: "positive" | "neutral" | "negative";
  isPrimarySource: boolean;
};
```

### 7.5 Source-quality rules

- Prefer primary sources when available.
- Preserve multiple independent reports when they add confirmation or distinct facts.
- Collapse exact duplicates and obvious syndication.
- Never treat repeated syndication as independent confirmation.
- Exclude articles that only mention AI incidentally.
- Require a usable title, canonical URL, and publication timestamp.
- Flag paywalled or excerpt-only sources rather than implying the full article was reviewed.
- Do not generate claims from information absent from the stored source fields or explicitly fetched source content.

## 8. Editorial pipeline

### 8.1 Stage A — deterministic preparation

Responsibilities:

- Load the bounded source window.
- Apply existing ingestion and topic-policy filters.
- Normalize source fields.
- Canonicalize URLs.
- Deduplicate exact matches and syndication.
- Calculate publisher diversity and primary-source presence.
- Freeze and persist the source snapshot.

This stage should be deterministic and independently testable.

### 8.2 Stage B — development clustering

Group sources that describe the same underlying event or development. Each cluster must contain:

- A neutral cluster label
- Member source IDs
- Earliest and latest timestamps
- Primary source, if any
- Independent publisher count
- Confirmed facts
- Disagreements or unresolved details
- Entities and themes

The model may suggest cluster membership, but deterministic similarity and URL/title evidence should be supplied to reduce arbitrary grouping.

### 8.3 Stage C — development scoring

Score each cluster from 0–5 on:

| Dimension | Description |
| --- | --- |
| Consequence | Potential effect on the target audience |
| Novelty | Whether the development materially changes the prior state |
| Evidence | Quality, independence, and authority of supporting sources |
| Timeliness | Why the story matters now |
| Lens fit | Relevance to AI, markets, regulation, governance, or risk |
| Thesis potential | Ability to support a clear, non-obvious argument |
| Human contribution | Presence of a meaningful question requiring the author’s judgment |

Store both numeric scores and short explanations. Ranking must not rely on sentiment or article volume alone.

### 8.4 Stage D — candidate-angle generation

Generate at most three meaningfully different candidate angles. Each candidate must include:

```ts
type EditorialCandidate = {
  candidateId: string;
  workingTitle: string;
  subtitle: string;
  thesis: string;
  audience: string;
  readerPromise: string;
  whyNow: string;
  supportingClusterIds: string[];
  originalContribution: string;
  authorQuestions: string[];
  counterargument: string;
  uncertainties: string[];
  whatToWatch: string[];
  recapRisk: "low" | "medium" | "high";
  supportScore: number;
  originalityScore: number;
  recommendationReason: string;
};
```

Candidates must differ in thesis, not merely title wording.

### 8.5 Stage E — selection and no-publish gate

Select a recommended candidate only when all of these conditions are met:

- Evidence threshold is satisfied.
- The thesis is supported by the source snapshot.
- The angle adds analysis beyond summarization.
- No critical claim depends on one weak secondary source.
- The title represents the proposed article without sensationalism.
- The article can identify a meaningful question for human judgment.

Otherwise save the run as `no_publish` with explicit reasons. `no_publish` is a successful editorial outcome, not a pipeline failure.

### 8.6 Stage F — claim ledger

For the selected candidate, generate a claim ledger:

```ts
type EditorialClaim = {
  claimId: string;
  claimText: string;
  claimType: "fact" | "inference" | "opinion" | "prediction";
  supportingSourceIds: string[];
  supportStatus: "supported" | "partially_supported" | "unsupported";
  confidence: "high" | "medium" | "low";
  caveat?: string;
};
```

No fact marked `unsupported` may appear in the outline or rough draft. Inferences and predictions must be labeled as such in the package.

### 8.7 Stage G — editorial package

The default package must contain:

1. Editorial recommendation
2. Three candidate angles
3. Selected thesis
4. Title and subtitle options
5. Two opening-hook options
6. Structured outline
7. Claim ledger
8. Source list with URLs
9. Counterargument and uncertainties
10. “What to watch next”
11. Questions for the human author
12. Medium-policy checklist
13. Final factual-review checklist

### 8.8 Optional rough-draft mode

Rough-draft generation must be disabled by default in MVP configuration.

When enabled:

- The output must be labeled `AI-generated working draft`.
- It must contain source markers that can be traced to the claim ledger.
- It must not invent first-person language or experience.
- It must not assert opinions as belonging to the author.
- It must include a disclosure reminder.
- It must never be sent to a publishing API.

## 9. Prompting and editorial configuration

### 9.1 Separate editorial configuration

Add versioned editorial configuration rather than embedding the complete writing brief in an API route. Configuration should include:

- Audience definition
- Editorial territory
- Author biography and allowed expertise claims
- Voice rules
- Structural preferences
- Prohibited phrases and habits
- Source and citation rules
- Originality requirements
- Medium-policy snapshot date and links
- Minimum evidence thresholds
- Rough-draft enablement
- Prompt version

The author biography must explicitly distinguish facts the system may state from experiences or opinions it must ask the author to supply.

### 9.2 Provider configuration

Use separate environment variables so editorial experimentation does not silently change recap behavior:

```text
EDITORIAL_OPENAI_MODEL=gpt-5.6-luna
EDITORIAL_OPENAI_REASONING_EFFORT=medium
EDITORIAL_DEEPSEEK_MODEL=deepseek-v4-pro
EDITORIAL_DEEPSEEK_REASONING_EFFORT=high
EDITORIAL_ROUGH_DRAFT_ENABLED=false
EDITORIAL_TIMEZONE=America/New_York
EDITORIAL_MIN_QUALITY_SCORE
EDITORIAL_MAX_SOURCES
```

Provider enablement belongs in `scheduled_briefings`, not environment variables, so the UI toggles take effect without a deployment. Environment variables supply allowed model defaults and credentials. Reuse existing provider adapters where practical, but keep prompts, output limits, validation, logging, and fallback behavior editorial-specific.

### 9.3 Structured output

Model stages must return schema-validated JSON. Markdown rendering should be derived from the saved structured package. Do not make loosely formatted prose the system of record.

### 9.4 Fallback behavior

If model generation fails:

- Save that provider output as `failed`, including provider, model, prompt version, stage, and sanitized error.
- Mark the parent run `partially_ready` when another enabled provider succeeded; mark it `failed` only when no provider produced a reviewable output.
- Do not create a generic fabricated editorial package.
- Permit an idempotent retry of only the failed provider.
- Preserve the frozen source snapshot.

This differs intentionally from the current recap’s source-summary fallback: a low-quality fallback article would undermine the editorial quality gate.

## 10. Data model

The storage model must separate the reusable briefing definition, the shared daily source snapshot, and each provider’s output. This prevents provider comparisons from accidentally using different articles or silently overwriting one another.

### 10.1 `scheduled_briefings`

Stores the toggle and schedule configuration shown on `/briefings/scheduled`:

```sql
CREATE TABLE scheduled_briefings (
  id                     BIGSERIAL PRIMARY KEY,
  briefing_key           TEXT NOT NULL UNIQUE,
  name                   TEXT NOT NULL,
  description            TEXT NOT NULL DEFAULT '',
  briefing_type          TEXT NOT NULL,
  enabled                BOOLEAN NOT NULL DEFAULT false,
  schedule_local_time    TIME NOT NULL DEFAULT '21:00',
  timezone               TEXT NOT NULL DEFAULT 'America/New_York',
  topic_keys             TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  source_kinds           TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  openai_enabled         BOOLEAN NOT NULL DEFAULT true,
  openai_model           TEXT NOT NULL DEFAULT 'gpt-5.6-luna',
  deepseek_enabled       BOOLEAN NOT NULL DEFAULT true,
  deepseek_model         TEXT NOT NULL DEFAULT 'deepseek-v4-pro',
  blind_comparison       BOOLEAN NOT NULL DEFAULT true,
  rough_draft_enabled    BOOLEAN NOT NULL DEFAULT false,
  notifications_enabled BOOLEAN NOT NULL DEFAULT true,
  config                 JSONB NOT NULL DEFAULT '{}'::jsonb,
  config_version         TEXT NOT NULL,
  created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (openai_enabled OR deepseek_enabled)
);
```

The initial seeded row is `daily_ai_editorial`. Additional briefing definitions may be added later without schema changes.

### 10.2 `scheduled_briefing_runs`

Stores one canonical source snapshot and overall status for a briefing/date pair:

```sql
CREATE TABLE scheduled_briefing_runs (
  id                         BIGSERIAL PRIMARY KEY,
  scheduled_briefing_id      BIGINT NOT NULL REFERENCES scheduled_briefings(id),
  briefing_date              DATE NOT NULL,
  snapshot_revision          INTEGER NOT NULL DEFAULT 1,
  window_start               TIMESTAMPTZ NOT NULL,
  window_end                 TIMESTAMPTZ NOT NULL,
  status                     TEXT NOT NULL,
  source_snapshot            JSONB NOT NULL DEFAULT '[]'::jsonb,
  source_snapshot_hash       TEXT NOT NULL,
  source_count               INTEGER NOT NULL DEFAULT 0,
  selected_output_id         BIGINT,
  selected_candidate_id      TEXT,
  no_publish_reasons         JSONB NOT NULL DEFAULT '[]'::jsonb,
  started_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at               TIMESTAMPTZ,
  updated_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (scheduled_briefing_id, briefing_date, snapshot_revision)
);
```

Run statuses:

```text
collecting
generating
partially_ready
ready_for_review
no_publish
failed
approved
human_editing
published
archived
```

`partially_ready` means at least one requested provider succeeded while another is still running or failed. The successful output remains reviewable.

### 10.3 `scheduled_briefing_outputs`

Stores one independently reviewable result per provider/model:

```sql
CREATE TABLE scheduled_briefing_outputs (
  id                       BIGSERIAL PRIMARY KEY,
  briefing_run_id          BIGINT NOT NULL REFERENCES scheduled_briefing_runs(id) ON DELETE CASCADE,
  provider                 TEXT NOT NULL,
  model                    TEXT NOT NULL,
  status                   TEXT NOT NULL,
  prompt_version           TEXT NOT NULL,
  config_version           TEXT NOT NULL,
  input_hash               TEXT NOT NULL,
  clusters                 JSONB NOT NULL DEFAULT '[]'::jsonb,
  candidates               JSONB NOT NULL DEFAULT '[]'::jsonb,
  recommended_candidate_id TEXT,
  claim_ledger             JSONB NOT NULL DEFAULT '[]'::jsonb,
  editorial_package        JSONB NOT NULL DEFAULT '{}'::jsonb,
  rough_draft              TEXT NOT NULL DEFAULT '',
  validation_results       JSONB NOT NULL DEFAULT '{}'::jsonb,
  usage                    JSONB NOT NULL DEFAULT '{}'::jsonb,
  latency_ms               INTEGER,
  error                    TEXT NOT NULL DEFAULT '',
  started_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at             TIMESTAMPTZ,
  updated_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (briefing_run_id, provider, model, input_hash)
);
```

After both tables exist, add the `selected_output_id` foreign key from `scheduled_briefing_runs` to `scheduled_briefing_outputs`.

### 10.4 `editorial_feedback`

Store human decisions, blind ratings, and learning signals separately from immutable provider output:

```sql
CREATE TABLE editorial_feedback (
  id                    BIGSERIAL PRIMARY KEY,
  briefing_run_id       BIGINT NOT NULL REFERENCES scheduled_briefing_runs(id) ON DELETE CASCADE,
  briefing_output_id    BIGINT REFERENCES scheduled_briefing_outputs(id) ON DELETE SET NULL,
  provider_revealed     BOOLEAN NOT NULL DEFAULT false,
  blind_quality_rating  INTEGER,
  evidence_rating       INTEGER,
  originality_rating    INTEGER,
  usefulness_rating     INTEGER,
  selected_candidate_id TEXT,
  decision              TEXT NOT NULL,
  author_notes          TEXT NOT NULL DEFAULT '',
  generated_text        TEXT NOT NULL DEFAULT '',
  final_text            TEXT NOT NULL DEFAULT '',
  edit_summary          JSONB NOT NULL DEFAULT '{}'::jsonb,
  published_url         TEXT NOT NULL DEFAULT '',
  published_at          TIMESTAMPTZ,
  performance_metrics   JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### 10.5 Comparison integrity and idempotency

- One canonical source snapshot revision per briefing/date pair.
- Every compared provider receives the same serialized source snapshot, prompt version, editorial configuration version, and output schema.
- The UI must not label two outputs “comparable” unless their `input_hash` values represent the same non-provider inputs.
- Rerunning one failed provider creates or resumes only that provider output; it does not recollect sources.
- A source refresh creates a new run revision and never mutates the earlier frozen snapshot.
- Provider identity may be hidden in the UI until a blind rating or explicit reveal action is recorded.
- Each model stage records an input hash so retries can avoid repeating completed work.

## 11. Application interfaces

### 11.1 Server functions

Add server-side functions for:

- `getEditorialSources(windowStart, windowEnd)`
- `listScheduledBriefings()`
- `getScheduledBriefing(briefingKey)`
- `updateScheduledBriefing(briefingKey, settings)`
- `createOrResumeBriefingRun(briefingKey, briefingDate)`
- `saveBriefingSourceSnapshot(runId, sources)`
- `createOrResumeProviderOutput(runId, provider, model)`
- `saveProviderEditorialAnalysis(outputId, clusters, candidates, claims)`
- `getBriefingRun(briefingKey, briefingDate)`
- `listBriefingRuns(filters)`
- `saveEditorialFeedback(runId, outputId, feedback)`
- `selectBriefingOutput(runId, outputId)`
- `markBriefingRunPublished(runId, url, metrics)`

### 11.2 API routes

Suggested endpoints:

```text
GET   /api/briefings/scheduled
PATCH /api/briefings/scheduled/:briefingKey
POST  /api/briefings/scheduled/:briefingKey/run
GET   /api/briefings/scheduled/:briefingKey/runs
GET   /api/briefings/scheduled/:briefingKey/runs/:date
POST  /api/briefings/scheduled/:briefingKey/runs/:date/select-output
POST  /api/briefings/scheduled/:briefingKey/runs/:date/retry-provider
POST  /api/briefings/scheduled/:briefingKey/runs/:date/refresh-sources
POST  /api/briefings/scheduled/:briefingKey/runs/:date/rough-draft
POST  /api/briefings/scheduled/:briefingKey/runs/:date/feedback
POST  /api/briefings/scheduled/:briefingKey/runs/:date/published
```

The scheduled endpoint must require a cron secret or equivalent server-to-server authentication. Human endpoints must use the application’s existing authentication and rate-limiting patterns.

### 11.3 Briefings information architecture

Keep **Briefings** as one primary navigation item and add a local sub-navigation beneath the page heading:

```text
Briefings
├── Custom Builder       /briefings
└── Scheduled Briefings  /briefings/scheduled
```

Do not add another top-level navigation item for Editorial. The existing custom builder remains unchanged in the first delivery phase.

### 11.4 Scheduled Briefings page

The page has two connected regions:

**Configuration and status**

- One card per scheduled briefing
- Master enable/disable toggle
- Schedule and timezone
- OpenAI and DeepSeek toggles and models
- Blind-comparison toggle
- Rough-draft and notification toggles
- Run-now action
- Last run, next run, and health state

**Output archive and review**

- Date-grouped run history
- Provider completion and validation status
- Same-source confirmation and snapshot hash
- Side-by-side provider output comparison
- Blind `Output A` / `Output B` mode with provider reveal
- Per-output ratings for evidence, originality, usefulness, and overall quality
- Preferred-output selection
- Candidate-angle comparison within each provider
- Source clusters and links
- Claim ledger with unsupported-claim warnings
- Author-question input fields
- Outline editor
- Optional rough-draft viewer/editor
- Approval, regeneration, skip, and publish-record actions
- Copy as Medium-ready Markdown and download `.md`
- Diff view between selected generated text and final human text
- Policy/disclosure checklist

On small screens, provider outputs render as tabs rather than narrow side-by-side columns.

## 12. Scheduling and operational design

### 12.1 Production scheduler

Use a repository-owned scheduled workflow or hosted cron as the production trigger. The job must call application code that reads and writes the shared Neon database; it must not depend on a desktop session being open.

For GitHub Actions, cron expressions are UTC. To run at 9:00 p.m. New York time across daylight-saving changes, schedule both relevant UTC hours and use a timezone-aware gate plus database idempotency:

```yaml
on:
  schedule:
    - cron: "0 1 * * *"
    - cron: "0 2 * * *"
```

The workflow must proceed only when the current local hour in `America/New_York` is 21. Because scheduled jobs may start late, the implementation should use a bounded grace window and rely on the unique editorial date to prevent duplicates.

### 12.2 Codex teammate or scheduled task

A recurring Codex task is optional and complementary. It may:

- Notify the author when the package is ready.
- Summarize the recommendation.
- Ask for the missing human contribution.
- Help revise the outline or draft.
- Record feedback after publication.

It must not be the only production scheduler or the only component capable of reading the saved editorial package. Official OpenAI documentation lists scheduled tasks and dedicated context-aware project teammates as supported workflow patterns: <https://learn.chatgpt.com/use-cases?category=automation&category=data&category=engineering&category=front-end&category=macos&sort=latest&task_type=analysis&task_type=testing&team=design&team=design-engineering&team=operations&team=research>.

### 12.3 Notifications

Notify only for:

- `ready_for_review`
- `no_publish`
- `failed`

The notification should include the editorial date, selected thesis or no-publish reason, source count, warnings, and a link to the admin review screen.

## 13. Quality safeguards

### 13.1 Automated validations

Before marking a package `ready_for_review`, verify:

- Every factual claim has at least one supporting source.
- High-impact factual claims have a primary source or two independent secondary sources when reasonably available.
- All source IDs resolve to the frozen source snapshot.
- All source URLs are non-empty and unique after canonicalization.
- The selected candidate has a thesis, reader promise, original contribution, counterargument, and what-to-watch section.
- The title is descriptive and not sensationalistic.
- The package contains no fabricated quotes or first-person claims.
- The output does not present a generic numbered roundup as the recommended structure.
- The model output passes its JSON schema.

### 13.2 Human checklist

Before publication, require the author to confirm:

- I agree with and can defend the central thesis.
- I supplied meaningful original judgment or experience.
- I reviewed every material factual claim and source.
- I removed unsupported or overstated claims.
- I reviewed quotations against their original sources.
- I chose an appropriate AI disclosure based on the text retained.
- The title and subtitle accurately represent the article.
- The article provides more value than the source summaries alone.

## 14. Learning and optimization

### 14.1 Initial approach

Do not fine-tune during the pilot. Use:

- A versioned editorial brief
- Structured human feedback
- Generated-to-final diffs
- Retrieval of a small number of relevant successful examples
- Prompt and threshold versioning

### 14.2 Training-readiness gate

Reconsider fine-tuning only after at least 20–30 substantially edited and reviewed articles exist and repeated edit patterns can be demonstrated. A training proposal must identify:

- Which stable behavior cannot be achieved reliably through configuration or retrieved examples
- Sufficient clean input/output pairs
- An evaluation set excluded from training
- Success metrics beyond engagement
- A rollback plan

### 14.3 Outcome metrics

Track operational and editorial metrics separately.

Operational:

- Successful nightly-run rate
- Source collection completeness
- Model-stage failure rate
- Median runtime and model cost
- Duplicate-run count
- Provider-specific success, latency, token usage, and estimated cost
- Same-input comparison-integrity failures

Editorial:

- Candidate acceptance rate
- `no_publish` rate
- Time from package creation to human decision
- Amount and type of human revision
- Unsupported-claim count caught before publication
- Publication rate
- Author quality rating
- Blind win rate by provider and model
- Preferred-provider rate after identity reveal
- Reader metrics entered after publication, such as views, reads, read ratio, followers, highlights, and responses

Engagement metrics must inform editorial learning but must not become the sole optimization target.

## 15. Security, privacy, and compliance

- Do not place database credentials or provider keys in prompts, logs, artifacts, or source snapshots.
- Sanitize provider errors before persistence.
- Treat captured article text as source material, not reusable prose.
- Avoid long quotations and derivative reproduction.
- Preserve publisher attribution and canonical URLs.
- Do not send drafts to third parties without an explicit user action.
- Keep public publishing outside the automated trust boundary.

## 16. Testing strategy

### 16.1 Unit tests

- Editorial-date and DST calculations
- Source-window boundaries
- Topic-policy reuse
- URL canonicalization and deduplication
- Syndication grouping fixtures
- Source normalization
- Scoring normalization
- JSON schema validation
- Claim-to-source referential integrity
- No-publish gate
- Idempotent reruns
- Unsupported-claim rejection
- Briefing enabled/disabled scheduling behavior
- At-least-one-provider settings constraint
- Same-snapshot input-hash calculation
- Provider-identity masking before reveal

### 16.2 Integration tests

- Generate a package from fixture RSS articles and analyses.
- Rerun the same briefing/date without duplicating the source snapshot.
- Generate OpenAI and DeepSeek outputs from the identical snapshot and non-provider configuration.
- Retry one failed provider without rerunning the successful provider.
- Preserve a successful provider output when the other provider fails.
- Turn a scheduled briefing off without deleting its history.
- Resume after failure at each model stage.
- Preserve the source snapshot during retry.
- Return `no_publish` for weak or derivative source sets.
- Reject unauthenticated scheduled requests.
- Save and retrieve human feedback.

### 16.3 Pilot evaluation set

Create a fixed set of at least ten historical daily source windows:

- Strong single-event day
- Several connected developments
- High-volume syndicated story
- Weak-news day
- Conflicting reports
- Primary-source-only story
- Paywalled/excerpt-only coverage
- AI story with only incidental financial relevance
- Regulatory AI story
- Day with more than the normal source cap

Human reviewers should score evidence, originality, relevance, usefulness, and publishability without seeing which provider/model generated each package. Provider identities are revealed only after ratings are saved.

## 17. Acceptance criteria

The MVP is complete when:

1. `/briefings` exposes local navigation for Custom Builder and Scheduled Briefings.
2. `/briefings/scheduled` shows the Daily AI Editorial configuration, master toggle, provider toggles, schedule, run health, and stored history.
3. Turning the briefing off prevents future scheduled runs without deleting settings or history.
4. A timezone-aware nightly job creates at most one source-snapshot revision per enabled briefing and New York calendar date.
5. The run uses stored AI-related sources and existing topic-policy logic.
6. The original source snapshot is persisted and inspectable.
7. Exact duplicates and obvious syndication are collapsed.
8. When both providers are enabled, OpenAI and DeepSeek receive the identical snapshot, prompt version, configuration version, and output schema.
9. Each provider output is stored separately with its model, validation, usage, latency, and error metadata.
10. A failed provider can be retried without replacing a successful provider output or recollecting sources.
11. The review UI supports labeled comparison and optional blind `Output A` / `Output B` comparison.
12. A human can rate both outputs and select one preferred output for continued editing.
13. Up to three genuinely distinct candidates are generated per successful provider.
14. The system can successfully return `no_publish`.
15. The selected candidate includes a thesis, reader promise, original contribution, counterargument, uncertainties, and what-to-watch section.
16. Every factual claim in the package maps to at least one stored source.
17. Unsupported factual claims block that provider output from `ready_for_review`.
18. The UI exposes candidates, sources, claims, warnings, author questions, and export actions.
19. No endpoint or job can publish publicly.
20. A human can approve, skip, save feedback, copy/download Markdown, and record a published URL.
21. Automated tests cover DST, toggles, idempotency, comparison integrity, source integrity, partial provider failure, recovery, and the no-publish path.
22. At least seven consecutive nightly pilot runs complete without duplicate records or silent failures.

## 18. Delivery plan

### Phase 0 — editorial decisions

Confirm:

- Target reader
- Core editorial lens
- Author biography and allowed expertise claims
- Whether rough-draft mode is included in MVP or deferred
- Preferred length and publishing frequency
- Default AI-disclosure posture
- Notification destination
- Whether OpenAI, DeepSeek, or both are enabled by default; recommendation: both during the pilot
- Whether provider comparison is blind by default; recommendation: yes

Deliverable: version 1 of the editorial configuration and human review checklist.

### Phase 1 — source and persistence foundation

- Add `scheduled_briefings`, `scheduled_briefing_runs`, `scheduled_briefing_outputs`, and feedback tables plus server data-access functions.
- Seed the disabled `daily_ai_editorial` definition with both providers enabled for manual pilot runs.
- Implement normalized RSS source collection.
- Reuse AI topic and ingestion-policy logic.
- Add snapshot hashing and idempotency.
- Add fixture-based source, provider-comparison, toggle, and DST tests.

Deliverable: deterministic source snapshots for any requested editorial date.

### Phase 2 — editorial analysis engine

- Implement a provider adapter and independent output lifecycle for OpenAI and DeepSeek.
- Implement clustering, scoring, candidate generation, claim ledger, schemas, and validation against the same frozen inputs.
- Implement explicit no-publish behavior.
- Add per-provider retries, partial-success behavior, usage metadata, and observability.

Deliverable: structured editorial packages generated from fixtures and live stored data.

### Phase 3 — review UI

- Add the Briefings local sub-navigation and `/briefings/scheduled` page.
- Add settings cards, toggles, run history, blind provider comparison, preferred-output selection, sources, claims, author inputs, approvals, and feedback.
- Add Markdown rendering derived from structured data.

Deliverable: end-to-end human review without direct database access.

### Phase 4 — scheduling and notification

- Add the 9:00 p.m. America/New_York scheduled workflow.
- Add authentication, concurrency control, timeout handling, and notifications.
- Run a seven-day non-publishing pilot.

Deliverable: reliable nightly packages and visible failure/no-publish reporting.

### Phase 5 — optional rough draft and learning loop

- Add gated rough-draft generation.
- Add generated-to-final diff capture.
- Add successful-example retrieval.
- Add manual performance-metric entry.

Deliverable: measurable improvement from human edits without model fine-tuning.

### Phase 6 — expanded corpus

- Normalize enriched Neon documents into the editorial source contract.
- Add primary-source weighting and cross-source lineage.
- Re-evaluate source caps and query indexes using production volume.

Deliverable: unified editorial coverage across RSS and the broader captured corpus.

## 19. Immediate next steps

1. Review and approve the seven Phase 0 editorial decisions.
2. Decide whether the MVP ends at the editorial package or includes an optional rough draft. The recommended default is package-only.
3. Create `editorial-config.v1` with the audience, author biography, editorial lens, voice constraints, thresholds, and policy snapshot.
4. Select ten historical dates for the pilot evaluation set.
5. Implement Phase 1 as the first code change: generalized scheduled-briefing schema, normalized source query, frozen source snapshot, provider-output records, idempotency, and tests.
6. Add the `/briefings/scheduled` shell and settings toggles before model generation so the control contract is testable.
7. Generate both OpenAI and DeepSeek packages manually from the same ten historical snapshots.
8. Rate them blind and choose the production default from evidence rather than provider preference.
9. Tune quality thresholds using blinded human scoring, not engagement alone.
10. Complete the review UI and enable the seven-day nightly pilot.
11. Only after the pilot passes the acceptance criteria, enable the production schedule and optional Codex notification task.

## 20. Phase 0 decision record template

Complete this block before implementation:

```yaml
target_reader: ""
editorial_lens: ""
author_biography: ""
allowed_expertise_claims: []
prohibited_author_claims: []
preferred_article_length_words: 1200
target_publication_frequency: "3-5 per week, quality-gated"
rough_draft_in_mvp: false
default_ai_disclosure_posture: "Disclose whenever generated prose is retained"
notification_destination: "Codex task"
editorial_timezone: "America/New_York"
nightly_run_time: "21:00"
openai_enabled: true
openai_model: "gpt-5.6-luna"
deepseek_enabled: true
deepseek_model: "deepseek-v4-pro"
blind_provider_comparison: true
```
