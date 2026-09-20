export type IntelligenceSource = "x" | "telegram" | "onchain" | "exchange" | "official_web" | "github";
export type SourceStatus = "available" | "not_materialized" | "not_configured";

export type DossierSource = { source: IntelligenceSource; status: SourceStatus; observations: number };
export type DossierEvidence = {
  observationId: string;
  source: IntelligenceSource;
  sourceRecordId: string;
  relationship: "asserts" | "independently_corroborates" | "repeats" | "quotes" | "disputes" | "corrects" | "retracts";
  content: string | null;
  publishedAt: string | null;
  observedAt: string;
  rawHash: string;
};
export type DossierAssessment = {
  id: string;
  eventId: string;
  status: "confirmed" | "partially_confirmed" | "contradicted" | "superseded";
  assessedAt: string;
  rationale: string;
  version: string;
};
export type DossierClaim = {
  id: string;
  type: string;
  predicate: string;
  object: unknown;
  effectiveAt: string | null;
  verifiability: "objective" | "conditional" | "subjective";
  createdAt: string;
  extractionVersion: string;
  resolution: "confirmed" | "partially_confirmed" | "contradicted" | "superseded" | "unresolved" | "not_verifiable";
  evidence: DossierEvidence[];
  assessments: DossierAssessment[];
};
export type DossierEvent = {
  id: string;
  type: string;
  occurredAt: string;
  observedAt: string;
  verificationPolicy: string;
  verificationVersion: string;
  attributes: unknown;
  sourceObservationIds: string[];
};
export type DossierMeasurement = {
  id: string;
  measuredAt: string;
  priceUsd: number | null;
  liquidityUsd: number | null;
  volumeUsd: number | null;
  source: string;
  methodologyVersion: string;
};
export type DossierOutcome = {
  id: string;
  claimId: string;
  measurementId: string;
  anchorType: "first_social_observation" | "first_unlinked_corroboration" | "event_occurred" | "verified_at";
  anchorAt: string;
  horizonSeconds: number;
  methodologyVersion: string;
};
export type DossierTimelineItem = {
  kind: "claim" | "event" | "measurement";
  id: string;
  at: string;
  label: string;
};
export type ClaimDossier = {
  status: "available" | "not_configured" | "not_materialized";
  coin: string;
  asset: { entityId: string; identityKey: string; network: string; contractAddress: string | null; nativeSymbol: string | null } | null;
  sources: DossierSource[];
  claims: DossierClaim[];
  events: DossierEvent[];
  measurements: DossierMeasurement[];
  outcomes: DossierOutcome[];
  timeline: DossierTimelineItem[];
  generatedAt: string;
  note: string;
};
