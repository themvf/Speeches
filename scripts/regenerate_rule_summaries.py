"""Regenerate rule_summaries.json with embedded comment presentation data."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rule_summaries import build_rule_summaries_payload

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

custom_path = DATA_DIR / "custom_documents.json"
enrich_path = DATA_DIR / "document_enrichment_state.json"
output_path = DATA_DIR / "rule_summaries.json"

print(f"Loading custom_documents.json ({custom_path.stat().st_size / 1_000_000:.1f} MB)...")
with open(custom_path, encoding="utf-8") as f:
    custom_payload = json.load(f)

print(f"Loading document_enrichment_state.json ({enrich_path.stat().st_size / 1_000_000:.1f} MB)...")
with open(enrich_path, encoding="utf-8") as f:
    enrichment_state = json.load(f)

print("Building rule summaries with embedded comment presentations...")
result = build_rule_summaries_payload(custom_payload, enrichment_state)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

output_size = output_path.stat().st_size
groups = len(result.get("groups", []))
total_comments = sum(len(g.get("comments", [])) for g in result.get("groups", []))
print(f"Done: {groups} groups, {total_comments} embedded comments, output size: {output_size / 1_000:.0f} KB")
