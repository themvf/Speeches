#!/usr/bin/env python3
"""One-time backfill for the entity alias map (see CLAUDE.md "Entity
normalization / alias map"): re-normalizes existing intelligence_mentions
rows so alias variants written before the alias map existed ("securities and
exchange commission", "the commission", ...) collapse into their canonical
normalized_value ("sec"), matching what prepareMentionBatch now writes for
new mentions.

Scoped to mention_type = 'entity' only - the exact scope of the write-path
change in neon.ts. Keyword/topic/individual rows are never touched.

For each alias pair, within one transaction:
  1. Merge: where an alias row and a canonical row exist for the same
     source, fold the alias row's confidence into the canonical row
     (GREATEST), then delete the alias row - a plain UPDATE would violate
     the (source_type, source_id, mention_type, normalized_value) unique
     constraint.
  2. Rewrite: remaining alias rows get the canonical normalized_value and
     the canonical display label as value.

Idempotent: a re-run finds no alias rows left and does nothing.

Usage:
    python backfill_entity_mentions.py [--dry-run] [--summary-path PATH]

Required env vars:
    DATABASE_URL   (Neon connection string - this script's entire purpose is
                    writing to Neon, so a missing value is a hard error)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import entity_aliases
import neon_feeds


def _utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


MERGE_CONFIDENCE_SQL = """
    UPDATE intelligence_mentions AS c
    SET confidence = GREATEST(c.confidence, a.confidence), value = %(label)s
    FROM intelligence_mentions AS a
    WHERE c.mention_type = 'entity' AND a.mention_type = 'entity'
      AND c.source_type = a.source_type AND c.source_id = a.source_id
      AND a.normalized_value = %(alias)s AND c.normalized_value = %(canonical)s
"""

DELETE_COLLIDING_ALIAS_SQL = """
    DELETE FROM intelligence_mentions AS a
    USING intelligence_mentions AS c
    WHERE a.mention_type = 'entity' AND c.mention_type = 'entity'
      AND c.source_type = a.source_type AND c.source_id = a.source_id
      AND a.normalized_value = %(alias)s AND c.normalized_value = %(canonical)s
"""

REWRITE_ALIAS_SQL = """
    UPDATE intelligence_mentions
    SET normalized_value = %(canonical)s, value = %(label)s
    WHERE mention_type = 'entity' AND normalized_value = %(alias)s
"""

# Aliased because neon_feeds._get_conn() uses RealDictCursor - rows come
# back as dicts keyed by column name, not indexable tuples.
COUNT_ALIAS_SQL = """
    SELECT COUNT(*) AS alias_rows FROM intelligence_mentions
    WHERE mention_type = 'entity' AND normalized_value = %(alias)s
"""


def _run(dry_run: bool) -> Dict[str, Any]:
    pairs = entity_aliases.entity_alias_pairs()
    summary: Dict[str, Any] = {
        "ok": True,
        "dry_run": dry_run,
        "alias_pairs_total": len(pairs),
        "pairs_with_rows": [],
        "rows_merged_and_deleted": 0,
        "rows_rewritten": 0,
        "ran_at": _utc_now_iso(),
    }

    # psycopg2's `with conn:` commits the transaction on clean exit (and
    # rolls back on exception), so all pairs land atomically; the explicit
    # rollback below keeps a dry run write-free even if a future edit adds
    # statements before the dry_run check.
    with neon_feeds._get_conn() as conn:
        with conn.cursor() as cur:
            for alias_norm, canonical_label, canonical_norm in sorted(pairs):
                params = {"alias": alias_norm, "canonical": canonical_norm, "label": canonical_label}
                cur.execute(COUNT_ALIAS_SQL, params)
                alias_rows = int(cur.fetchone()["alias_rows"])
                if alias_rows == 0:
                    continue
                detail: Dict[str, Any] = {
                    "alias_normalized": alias_norm,
                    "canonical_normalized": canonical_norm,
                    "alias_rows": alias_rows,
                }
                if not dry_run:
                    cur.execute(MERGE_CONFIDENCE_SQL, params)
                    cur.execute(DELETE_COLLIDING_ALIAS_SQL, params)
                    deleted = cur.rowcount
                    cur.execute(REWRITE_ALIAS_SQL, params)
                    rewritten = cur.rowcount
                    detail["deleted_after_merge"] = deleted
                    detail["rewritten"] = rewritten
                    summary["rows_merged_and_deleted"] += deleted
                    summary["rows_rewritten"] += rewritten
                summary["pairs_with_rows"].append(detail)
        if dry_run:
            conn.rollback()
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report affected row counts only; no writes")
    parser.add_argument("--summary-path", default="", help="Also write the JSON summary to this path")
    args = parser.parse_args(argv)

    try:
        summary = _run(dry_run=args.dry_run)
    except Exception as exc:
        summary = {"ok": False, "error": str(exc), "dry_run": args.dry_run, "ran_at": _utc_now_iso()}

    output = json.dumps(summary, indent=2)
    print(output)
    if args.summary_path:
        try:
            with open(args.summary_path, "w", encoding="utf-8") as handle:
                handle.write(output)
        except Exception as exc:
            print(f"[backfill_entity_mentions] could not write summary file: {exc}", file=sys.stderr)
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
