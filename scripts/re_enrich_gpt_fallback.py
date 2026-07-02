#!/usr/bin/env python3
"""Re-enrich stale GPT fallback records with DeepSeek.

Targets:
- Corpus enrichment entries where model is gpt-5.1 and status is fallback_enriched.
- RSS article analyses where model is gpt-5.1 and fallback is true, when DATABASE_URL is available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_financial_news_pipeline as core  # noqa: E402

try:
    import psycopg2
    import psycopg2.extras
except Exception:  # pragma: no cover
    psycopg2 = None


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def is_gpt51_fallback_entry(entry: Dict[str, Any]) -> bool:
    return text(entry.get("model")).lower().startswith("gpt-5.1") and text(entry.get("status")).lower() == "fallback_enriched"


def load_corpus_source_kinds(custom_payload: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in custom_payload.get("documents", []):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        doc_id = text(metadata.get("document_id"))
        source_kind = text(metadata.get("source_kind"))
        if doc_id and source_kind:
            out[doc_id] = source_kind
    return out


def re_enrich_corpus(args: argparse.Namespace) -> Dict[str, Any]:
    secrets_payload = core._load_streamlit_secrets()
    storage, gcs_status = core._get_gcs_storage(secrets_payload)
    if args.require_remote_persistence and storage is None:
        raise RuntimeError(gcs_status)

    custom_payload = core._load_custom_documents(storage)
    enrichment_state = core._load_enrichment_state(storage)
    entries = enrichment_state.get("entries", {}) if isinstance(enrichment_state.get("entries"), dict) else {}
    source_by_doc_id = load_corpus_source_kinds(custom_payload)

    stale_by_source: Dict[str, List[str]] = defaultdict(list)
    skipped_missing_source: List[str] = []
    for doc_id, entry in entries.items():
        if not isinstance(entry, dict) or not is_gpt51_fallback_entry(entry):
            continue
        source_kind = source_by_doc_id.get(text(doc_id))
        if not source_kind:
            skipped_missing_source.append(text(doc_id))
            continue
        stale_by_source[source_kind].append(text(doc_id))

    max_docs = max(0, int(args.max_docs or 0))
    if max_docs > 0:
        remaining = max_docs
        limited: Dict[str, List[str]] = {}
        for source_kind, ids in sorted(stale_by_source.items()):
            if remaining <= 0:
                break
            selected = ids[:remaining]
            if selected:
                limited[source_kind] = selected
                remaining -= len(selected)
        stale_by_source = defaultdict(list, limited)

    source_summaries: List[Dict[str, Any]] = []
    processed = 0
    enriched = 0
    fallback = 0
    for source_kind, doc_ids in sorted(stale_by_source.items()):
        if not doc_ids:
            continue
        ns = argparse.Namespace(
            command="enrich",
            source_kind=source_kind,
            mode="all",
            doc_id=doc_ids,
            doc_ids_from_summary="",
            limit=None,
            order="stored",
            model=args.model,
            provider="deepseek",
            heuristic_only=False,
            dry_run=bool(args.dry_run),
            require_remote_persistence=bool(args.require_remote_persistence),
            summary_path="",
        )
        summary = core._run_news_enrichment(ns)
        processed += int(summary.get("processed_count", 0) or 0)
        enriched += int(summary.get("enriched_count", 0) or 0)
        fallback += int(summary.get("fallback_enriched_count", 0) or 0)
        source_summaries.append(
            {
                "source_kind": source_kind,
                "selected_doc_ids": doc_ids,
                "processed_count": summary.get("processed_count", 0),
                "enriched_count": summary.get("enriched_count", 0),
                "fallback_enriched_count": summary.get("fallback_enriched_count", 0),
                "used_models": summary.get("used_models", []),
            }
        )

    return {
        "found_count": sum(len(ids) for ids in stale_by_source.values()) + len(skipped_missing_source),
        "selected_count": sum(len(ids) for ids in stale_by_source.values()),
        "processed_count": processed,
        "enriched_count": enriched,
        "fallback_enriched_count": fallback,
        "skipped_missing_source_count": len(skipped_missing_source),
        "skipped_missing_source_doc_ids": skipped_missing_source[:50],
        "by_source_kind": source_summaries,
    }


def deepseek_feed_analysis(row: Dict[str, Any], topics: List[str], model: str, api_key: str) -> Dict[str, Any]:
    prompt = "\n".join(
        [
            f"Title: {text(row.get('title'))}",
            f"Source: {text(row.get('feed_key'))}",
            f"Author: {text(row.get('author'))}",
            f"Published: {text(row.get('published_at')) or text(row.get('fetched_at'))}",
            f"URL: {text(row.get('url'))}",
            f"Feed tone: {text(row.get('tone_label'))}",
            f"Matched topics: {', '.join(topics)}",
            "Item type: article",
            "",
            "RSS summary / excerpt:",
            text(row.get("description"))[:6000],
        ]
    )
    instructions = (
        "You are a regulatory intelligence analyst for financial services, securities, banking, fintech, "
        "and enforcement coverage. Analyze only the supplied RSS/feed metadata and excerpt. Do not invent facts. "
        "Return dense, specific JSON for a working analyst, not a generic news summary. thesis must state the "
        "concrete development in one sentence and include named agencies, companies, people, markets, products, "
        "or proceedings when supplied. why_it_matters must contain 3-5 substantive bullets connecting the supplied "
        "facts to enforcement posture, supervision, compliance controls, market structure, investor harm, litigation, "
        "or policy impact. risk_signals must contain concrete red flags from the text; if the feed excerpt is sparse, "
        "identify exactly which facts are missing instead of writing boilerplate. follow_up_questions must be specific "
        "to the item and ask for missing securities, parties, procedural posture, timing, losses, rules, or affected "
        "markets where relevant. Avoid vague phrases such as 'potential regulatory risk', 'may warrant review', or "
        "'market impact' unless tied to a named fact from the input. "
        "Return only valid JSON with keys: thesis, why_it_matters, risk_signals, follow_up_questions, "
        "keywords, individuals, entities."
    )
    response = requests.post(
        os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    content = text(payload.get("choices", [{}])[0].get("message", {}).get("content"))
    parsed = json.loads(content)
    return {
        "thesis": text(parsed.get("thesis"))[:360],
        "why_it_matters": string_list(parsed.get("why_it_matters"), 5),
        "risk_signals": string_list(parsed.get("risk_signals"), 5),
        "follow_up_questions": string_list(parsed.get("follow_up_questions"), 5),
        "keywords": string_list(parsed.get("keywords"), 12, 80),
        "individuals": string_list(parsed.get("individuals"), 10, 80),
        "entities": string_list(parsed.get("entities"), 14, 100),
        "model": model,
        "generated_at": utc_now_iso(),
        "fallback": False,
    }


def string_list(value: Any, max_items: int, max_chars: int = 180) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        cleaned = text(item)[:max_chars]
        if cleaned:
            out.append(cleaned)
        if len(out) >= max_items:
            break
    return out


def rss_rows(conn: Any, limit: int) -> List[Dict[str, Any]]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT a.id, a.guid, a.feed_key, a.title, a.url, a.description, a.author,
                   a.published_at, a.fetched_at, a.tone_label, ra.source_hash, ra.topics
            FROM rss_articles a
            JOIN rss_article_analysis ra ON ra.article_id = a.id
            WHERE ra.fallback = true
              AND lower(ra.model) LIKE 'gpt-5.1%%'
            ORDER BY COALESCE(a.published_at, a.fetched_at) DESC
            LIMIT %s
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]


def update_rss_row(conn: Any, row: Dict[str, Any], analysis: Dict[str, Any], topics: List[str]) -> None:
    analysis_text = json.dumps(analysis, ensure_ascii=False)
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE rss_article_analysis
            SET status = 'enriched',
                model = %s,
                generated_at = %s,
                thesis = %s,
                why_it_matters = %s,
                risk_signals = %s,
                follow_up_questions = %s,
                keywords = %s,
                individuals = %s,
                entities = %s,
                topics = %s,
                analysis_text = %s,
                fallback = false,
                error = ''
            WHERE article_id = %s
            """,
            (
                analysis["model"],
                analysis["generated_at"],
                analysis["thesis"],
                analysis["why_it_matters"],
                analysis["risk_signals"],
                analysis["follow_up_questions"],
                analysis["keywords"],
                analysis["individuals"],
                analysis["entities"],
                topics,
                analysis_text,
                row["id"],
            ),
        )


def re_analyze_rss(args: argparse.Namespace) -> Dict[str, Any]:
    database_url = text(os.getenv("DATABASE_URL"))
    api_key = text(os.getenv("DEEPSEEK_API") or os.getenv("DEEPSEEK_API_KEY"))
    if not database_url:
        return {"skipped": True, "reason": "DATABASE_URL is not configured.", "found_count": 0}
    if not api_key:
        return {"skipped": True, "reason": "DEEPSEEK_API is not configured.", "found_count": 0}
    if psycopg2 is None:
        return {"skipped": True, "reason": "psycopg2 is not installed.", "found_count": 0}

    limit = max(1, int(args.rss_limit or 1))
    processed = 0
    failed: List[Dict[str, Any]] = []
    preview: List[Dict[str, Any]] = []
    with psycopg2.connect(database_url) as conn:
        rows = rss_rows(conn, limit)
        for row in rows:
            topics = [text(item) for item in (row.get("topics") or []) if text(item)]
            try:
                if args.dry_run:
                    processed += 1
                    continue
                analysis = deepseek_feed_analysis(row, topics, args.model, api_key)
                update_rss_row(conn, row, analysis, topics)
                conn.commit()
                processed += 1
                if len(preview) < 20:
                    preview.append({"article_id": row["id"], "feed_key": row["feed_key"], "title": row["title"]})
            except Exception as exc:
                conn.rollback()
                failed.append({"article_id": row.get("id"), "title": row.get("title"), "error": str(exc)[:500]})

    return {
        "skipped": False,
        "found_count": len(rows),
        "processed_count": processed,
        "failed_count": len(failed),
        "failed": failed[:50],
        "processed_preview": preview,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-enrich stale gpt-5.1 fallback records with DeepSeek")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--max-docs", type=int, default=0, help="Maximum corpus docs to process; 0 means all found.")
    parser.add_argument("--rss-limit", type=int, default=500)
    parser.add_argument("--skip-corpus", action="store_true")
    parser.add_argument("--skip-rss", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-remote-persistence", action="store_true")
    parser.add_argument("--summary-path", default="gpt_fallback_reenrich_summary.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary: Dict[str, Any] = {
        "ok": True,
        "ran_at": utc_now_iso(),
        "model": args.model,
        "dry_run": bool(args.dry_run),
    }
    try:
        summary["corpus"] = {"skipped": True, "reason": "skip-corpus requested"} if args.skip_corpus else re_enrich_corpus(args)
    except Exception as exc:
        summary["ok"] = False
        summary["corpus"] = {"skipped": False, "error": str(exc)}
    try:
        summary["rss"] = {"skipped": True, "reason": "skip-rss requested"} if args.skip_rss else re_analyze_rss(args)
    except Exception as exc:
        summary["ok"] = False
        summary["rss"] = {"skipped": False, "error": str(exc)}

    Path(args.summary_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
