import { neon } from "@neondatabase/serverless";
import { coinConfig } from "../crypto-coins.ts";
import { assetIdentityKey, claimResolution } from "../intelligence-fusion.ts";
import type {
  ClaimDossier, DossierAssessment, DossierClaim, DossierEvent, DossierEvidence,
  DossierMeasurement, DossierOutcome, DossierSource, DossierTimelineItem, IntelligenceSource,
} from "../intelligence-fusion-types.ts";

function iso(value: unknown): string {
  return value instanceof Date ? value.toISOString() : new Date(String(value)).toISOString();
}

function emptyDossier(coin: string, status: ClaimDossier["status"], note: string): ClaimDossier {
  return {
    status, coin, asset: null,
    sources: (["x", "telegram", "onchain", "exchange", "official_web", "github"] as IntelligenceSource[]).map((source) => ({
      source, status: source === "telegram" ? "not_configured" as const : "not_materialized" as const, observations: 0,
    })),
    claims: [], events: [], measurements: [], outcomes: [], timeline: [], generatedAt: new Date().toISOString(), note,
  };
}

export async function readClaimDossier(coin: string): Promise<ClaimDossier> {
  if (!process.env.DATABASE_URL) return emptyDossier(coin, "not_configured", "The intelligence database is not connected in this environment.");
  const sql = neon(process.env.DATABASE_URL);
  const relations = await sql`SELECT to_regclass('public.intelligence_observations') AS observations,
    to_regclass('public.intelligence_claims') AS claims,to_regclass('public.intelligence_events') AS events`;
  if (!relations[0]?.observations || !relations[0]?.claims || !relations[0]?.events) {
    return emptyDossier(coin, "not_materialized", "The fusion schema has not been materialized. No collection was attempted.");
  }

  const cfg = coinConfig(coin), identityKey = assetIdentityKey(cfg.network, cfg.address, cfg.symbol);
  const assets = await sql.query(
    `SELECT e.id,a.identity_key,a.network,a.contract_address,a.native_symbol
       FROM intelligence_asset_identities a JOIN intelligence_entities e ON e.id=a.entity_id
      WHERE a.identity_key=$1`, [identityKey],
  );
  if (!assets.length) return emptyDossier(coin, "not_materialized", "No derived observations exist for this canonical asset. No collection was attempted.");
  const asset = assets[0] as Record<string, unknown>, entityId = String(asset.id);

  const [sourceRows, claimRows, eventRows, measurementRows, outcomeRows] = await Promise.all([
    sql.query(`SELECT o.source,count(DISTINCT o.id)::int AS observations
      FROM intelligence_observations o JOIN intelligence_observation_entities oe ON oe.observation_id=o.id
      WHERE oe.entity_id=$1 GROUP BY o.source`, [entityId]),
    sql.query(`SELECT c.id,c.claim_type,c.predicate,c.object,c.effective_at,c.verifiability,c.created_at,c.extraction_version,
      COALESCE(jsonb_agg(DISTINCT jsonb_build_object('observationId',o.id,'source',o.source,'sourceRecordId',o.source_record_id,
        'relationship',oc.relationship,'content',o.content,'publishedAt',o.published_at,'observedAt',o.observed_at,'rawHash',o.raw_hash))
        FILTER(WHERE o.id IS NOT NULL),'[]') AS evidence,
      COALESCE(jsonb_agg(DISTINCT jsonb_build_object('id',a.id,'eventId',a.event_id,'status',a.status,
        'assessedAt',a.assessed_at,'rationale',a.rationale,'version',a.version)) FILTER(WHERE a.id IS NOT NULL),'[]') AS assessments
      FROM intelligence_claims c
      LEFT JOIN intelligence_observation_claims oc ON oc.claim_id=c.id LEFT JOIN intelligence_observations o ON o.id=oc.observation_id
      LEFT JOIN intelligence_claim_event_assessments a ON a.claim_id=c.id
      WHERE c.subject_entity_id=$1 GROUP BY c.id ORDER BY c.created_at,c.id`, [entityId]),
    sql.query(`SELECT e.id,e.event_type,e.occurred_at,e.observed_at,e.verification_policy,e.verification_version,e.attributes,
      COALESCE(array_agg(DISTINCT eo.observation_id) FILTER(WHERE eo.observation_id IS NOT NULL),'{}') AS source_observation_ids
      FROM intelligence_events e JOIN intelligence_event_entities ee ON ee.event_id=e.id
      LEFT JOIN intelligence_event_observations eo ON eo.event_id=e.id
      WHERE ee.entity_id=$1 GROUP BY e.id ORDER BY e.occurred_at,e.id`, [entityId]),
    sql.query(`SELECT id,measured_at,price_usd,liquidity_usd,volume_usd,source,methodology_version
      FROM intelligence_market_measurements WHERE entity_id=$1 ORDER BY measured_at,id`, [entityId]),
    sql.query(`SELECT o.id,o.claim_id,o.measurement_id,o.anchor_type,o.anchor_at,o.horizon_seconds,o.methodology_version
      FROM intelligence_claim_outcomes o JOIN intelligence_claims c ON c.id=o.claim_id
      WHERE c.subject_entity_id=$1 ORDER BY o.anchor_at,o.horizon_seconds,o.id`, [entityId]),
  ]);

  const counts = new Map(sourceRows.map((row) => [String(row.source), Number(row.observations)]));
  const sources: DossierSource[] = (["x", "telegram", "onchain", "exchange", "official_web", "github"] as IntelligenceSource[]).map((source) => ({
    source,
    status: source === "telegram" ? "not_configured" : counts.has(source) ? "available" : "not_materialized",
    observations: counts.get(source) ?? 0,
  }));
  const claims: DossierClaim[] = claimRows.map((row) => {
    const rawEvidence = (row.evidence ?? []) as Record<string, unknown>[];
    const evidence: DossierEvidence[] = rawEvidence.map((item) => ({
      observationId: String(item.observationId), source: item.source as IntelligenceSource,
      sourceRecordId: String(item.sourceRecordId), relationship: item.relationship as DossierEvidence["relationship"],
      content: item.content == null ? null : String(item.content), publishedAt: item.publishedAt == null ? null : iso(item.publishedAt),
      observedAt: iso(item.observedAt), rawHash: String(item.rawHash),
    })).sort((a, b) => (a.publishedAt ?? a.observedAt).localeCompare(b.publishedAt ?? b.observedAt));
    const rawAssessments = (row.assessments ?? []) as Record<string, unknown>[];
    const assessments: DossierAssessment[] = rawAssessments.map((item) => ({
      id: String(item.id), eventId: String(item.eventId), status: item.status as DossierAssessment["status"],
      assessedAt: iso(item.assessedAt), rationale: String(item.rationale), version: String(item.version),
    }));
    const verifiability = row.verifiability as DossierClaim["verifiability"];
    return {
      id: String(row.id), type: String(row.claim_type), predicate: String(row.predicate), object: row.object,
      effectiveAt: row.effective_at == null ? null : iso(row.effective_at), verifiability,
      createdAt: iso(row.created_at), extractionVersion: String(row.extraction_version),
      resolution: claimResolution(verifiability, assessments), evidence, assessments,
    };
  });
  const events: DossierEvent[] = eventRows.map((row) => ({
    id: String(row.id), type: String(row.event_type), occurredAt: iso(row.occurred_at), observedAt: iso(row.observed_at),
    verificationPolicy: String(row.verification_policy), verificationVersion: String(row.verification_version),
    attributes: row.attributes, sourceObservationIds: (row.source_observation_ids as unknown[]).map(String),
  }));
  const measurements: DossierMeasurement[] = measurementRows.map((row) => ({
    id: String(row.id), measuredAt: iso(row.measured_at), priceUsd: row.price_usd == null ? null : Number(row.price_usd),
    liquidityUsd: row.liquidity_usd == null ? null : Number(row.liquidity_usd),
    volumeUsd: row.volume_usd == null ? null : Number(row.volume_usd), source: String(row.source),
    methodologyVersion: String(row.methodology_version),
  }));
  const outcomes: DossierOutcome[] = outcomeRows.map((row) => ({
    id: String(row.id), claimId: String(row.claim_id), measurementId: String(row.measurement_id),
    anchorType: row.anchor_type as DossierOutcome["anchorType"], anchorAt: iso(row.anchor_at),
    horizonSeconds: Number(row.horizon_seconds), methodologyVersion: String(row.methodology_version),
  }));
  const timeline: DossierTimelineItem[] = [
    ...claims.map((claim) => ({ kind: "claim" as const, id: claim.id, at: claim.effectiveAt ?? claim.createdAt, label: claim.predicate })),
    ...events.map((event) => ({ kind: "event" as const, id: event.id, at: event.occurredAt, label: event.type })),
    ...measurements.map((measurement) => ({ kind: "measurement" as const, id: measurement.id, at: measurement.measuredAt, label: "market_measurement" })),
  ].sort((a, b) => a.at.localeCompare(b.at) || a.kind.localeCompare(b.kind) || a.id.localeCompare(b.id));
  return {
    status: "available", coin,
    asset: { entityId, identityKey: String(asset.identity_key), network: String(asset.network),
      contractAddress: asset.contract_address == null ? null : String(asset.contract_address),
      nativeSymbol: asset.native_symbol == null ? null : String(asset.native_symbol) },
    sources, claims, events, measurements, outcomes, timeline, generatedAt: new Date().toISOString(),
    note: "Read-only dossier derived from saved archives. Telegram is a placeholder and no provider call was made.",
  };
}
