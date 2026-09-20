import type { ClaimDossier, DossierAssessment } from "./intelligence-fusion-types.ts";

export function assetIdentityKey(network: string, address: string | null, symbol: string) {
  if (!address) return `asset:${network}:native:${symbol.toUpperCase()}`;
  return `asset:${network}:${address.startsWith("0x") ? address.toLowerCase() : address}`;
}

export function claimResolution(
  verifiability: "objective" | "conditional" | "subjective",
  assessments: Pick<DossierAssessment, "status" | "assessedAt">[],
): ClaimDossier["claims"][number]["resolution"] {
  if (verifiability === "subjective") return "not_verifiable";
  if (!assessments.length) return "unresolved";
  return [...assessments].sort((a, b) => b.assessedAt.localeCompare(a.assessedAt))[0]!.status;
}

