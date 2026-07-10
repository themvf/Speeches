#!/usr/bin/env python3
"""
Standalone sentiment/tone scoring pipeline.

Workflow: extraction -> enrichment -> sentiment (this script)

Reads full text from custom_documents.json, scores the author's editorial
tone using an LLM, and writes results back into document_enrichment_state.json
under a top-level 'sentiment' key per entry — leaving enrichment untouched.

Storage, model-client, and provider-fallback plumbing are delegated to
run_financial_news_pipeline (core) so this script inherits the same hardened
GCS read-failure guard and generation-match write protection instead of
maintaining its own copy.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import run_financial_news_pipeline as core


SENTIMENT_LABELS = {"positive", "negative", "neutral"}

# Sentiment statuses that count as "already done" for --mode only_missing.
# Anything else (fallback_scored from a provider outage, failed, or missing)
# is retried so a transient outage doesn't permanently pin a doc to neutral.
_COMPLETED_SENTIMENT_STATUSES = {"scored", "reviewed"}

SENTIMENT_SYSTEM_PROMPT = """\
You are a tone-scoring agent for financial and regulatory news.

Score the AUTHOR'S editorial tone and framing — not the subject matter or event severity.

Rules:
- Institutional press releases (DOJ, SEC, CFTC, Fed, FINRA) are neutral by default
  unless the author uses charged, alarming, or celebratory language.
- Wire-service factual reporting (Reuters, AP, Bloomberg News) defaults to neutral.
- An arrest announcement written in plain institutional language = neutral.
- A piece calling a regulation "dangerously overreaching" = negative.
- A piece framing a ruling as a "landmark victory for investors" = positive.
- Score reflects word choice, framing, and rhetorical stance — not what happened.

Return ONLY valid JSON with no markdown or commentary:
{"score": <float -1.0 to 1.0>, "label": "positive" | "negative" | "neutral", "rationale": "<one concise sentence>"}
"""


def _utc_now_iso() -> str:
    return core._utc_now_iso()


def _normalize_sentiment(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    try:
        score = float(raw.get("score", 0.0) or 0.0)
    except Exception:
        score = 0.0
    score = max(-1.0, min(1.0, score))

    label = str(raw.get("label", "neutral") or "neutral").strip().lower()
    if label not in SENTIMENT_LABELS:
        label = "neutral"

    rationale = str(raw.get("rationale", "") or "").strip()[:300]
    return {"score": round(score, 4), "label": label, "rationale": rationale}


def _heuristic_sentiment(text: str) -> Dict[str, Any]:
    """
    Conservative fallback: defaults to neutral. Only shifts on explicitly
    editorial language — avoids subject-matter words like 'fraud' or 'arrest'
    that would wrongly skew institutional reporting.
    """
    lower = str(text or "").lower()

    positive_editorial = [
        "landmark victory", "breakthrough", "historic", "celebrated",
        "welcomed by", "praised", "applauded", "hailed",
    ]
    negative_editorial = [
        "reckless", "dangerously", "alarming", "overreaching", "disastrous",
        "slammed", "blasted", "outrage", "fiasco", "catastrophic",
    ]

    pos_hits = sum(1 for phrase in positive_editorial if phrase in lower)
    neg_hits = sum(1 for phrase in negative_editorial if phrase in lower)

    if pos_hits == 0 and neg_hits == 0:
        return {"score": 0.0, "label": "neutral", "rationale": "No editorial language detected; defaulting to neutral."}

    raw_score = (pos_hits - neg_hits) / max(pos_hits + neg_hits, 1)
    score = round(max(-1.0, min(1.0, raw_score)), 4)
    label = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
    return {"score": score, "label": label, "rationale": "Heuristic fallback based on editorial word patterns."}


def _score_with_model(client: Any, provider: str, model_name: str, title: str, text: str) -> Dict[str, Any]:
    if len(text) > 60000:
        text = text[:40000] + "\n\n[...TRUNCATED...]\n\n" + text[-10000:]

    prompt = f"Title: {title}\n\nArticle:\n{text}"

    for attempt in range(1, 3):
        instruction = SENTIMENT_SYSTEM_PROMPT
        if attempt > 1:
            instruction += " Respond with raw JSON only. No markdown, no code fences."
        raw_text = core._create_enrichment_completion(
            client=client,
            provider=provider,
            model_name=model_name,
            instruction=instruction,
            prompt=prompt,
        )
        parsed = core._extract_first_json_object(raw_text)
        if parsed:
            return _normalize_sentiment(parsed)

    raise RuntimeError("Model did not return parseable JSON after 2 attempts.")


def _build_candidates(
    custom_payload: Dict[str, Any],
    source_kind: str,
    doc_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    selected_ids = {str(d).strip() for d in (doc_ids or []) if str(d).strip()}
    candidates = []
    for item in custom_payload.get("documents", []):
        if not isinstance(item, dict):
            continue
        m = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
        c = item.get("content", {}) if isinstance(item.get("content", {}), dict) else {}
        doc_id = str(m.get("document_id", "") or "").strip()
        if not doc_id:
            continue
        if selected_ids and doc_id not in selected_ids:
            continue
        if source_kind and str(m.get("source_kind", "") or "").strip() != source_kind:
            continue
        full_text = str(c.get("full_text", "") or "").strip()
        if not full_text:
            continue
        candidates.append({
            "doc_id": doc_id,
            "title": str(m.get("title", "") or "").strip(),
            "full_text": full_text,
        })
    return candidates


def _needs_scoring(entry: Any) -> bool:
    """True if a doc still needs (re)scoring under --mode only_missing."""
    if not isinstance(entry, dict):
        return True
    sentiment = entry.get("sentiment")
    if not isinstance(sentiment, dict):
        return True
    status = str(sentiment.get("status", "") or "").strip().lower()
    return status not in _COMPLETED_SENTIMENT_STATUSES


def _openai_fallback_models() -> List[str]:
    return ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1", "gpt-4o"]


def _run_score(args: argparse.Namespace) -> Dict[str, Any]:
    secrets_payload = core._load_streamlit_secrets()
    storage, gcs_status = core._get_gcs_storage(secrets_payload)
    if args.require_remote_persistence and storage is None:
        raise RuntimeError(gcs_status)

    custom_payload = core._load_custom_documents(storage)
    enrichment_state = core._load_enrichment_state(storage)
    entries = enrichment_state.setdefault("entries", {})

    doc_ids: List[str] = [str(d).strip() for d in (args.doc_id or []) if str(d).strip()]

    candidates = _build_candidates(
        custom_payload=custom_payload,
        source_kind=args.source_kind,
        doc_ids=doc_ids or None,
    )

    if not doc_ids and args.mode == "only_missing":
        candidates = [c for c in candidates if _needs_scoring(entries.get(c["doc_id"]))]

    limit = len(candidates) if args.limit is None else max(0, int(args.limit))
    targets = candidates[:limit] if limit > 0 else []

    provider = str(args.provider or "deepseek").strip().lower()
    if provider not in {"openai", "deepseek"}:
        provider = "deepseek"
    client = None if args.heuristic_only else core._get_model_client(secrets_payload, provider)
    if provider == "deepseek":
        accessible_models = core._candidate_deepseek_models()
        preferred_model = args.model or accessible_models[0]
    else:
        accessible_models = _openai_fallback_models()
        preferred_model = args.model or accessible_models[0]

    scored_count = 0
    fallback_count = 0
    failed: List[Dict[str, Any]] = []
    used_models: List[str] = []

    for candidate in targets:
        doc_id = candidate["doc_id"]
        model_used = ""
        status = "scored"
        error_msg = ""

        try:
            if client is None:
                raise RuntimeError(f"{provider.title()} client unavailable.")
            ordered = [preferred_model] + [m for m in accessible_models if m != preferred_model]
            sentiment = None
            last_error = None
            for model_name in ordered:
                try:
                    sentiment = _score_with_model(client, provider, model_name, candidate["title"], candidate["full_text"])
                    model_used = model_name
                    break
                except Exception as e:
                    last_error = e
                    if not core._is_model_access_error(e):
                        raise
            if sentiment is None:
                raise last_error or RuntimeError("No model available.")
            scored_count += 1
            if model_used and model_used not in used_models:
                used_models.append(model_used)
        except Exception as e:
            sentiment = _heuristic_sentiment(candidate["full_text"])
            status = "fallback_scored"
            error_msg = str(e)
            fallback_count += 1

        sentiment_entry = {
            **sentiment,
            "model": model_used or "heuristic",
            "provider": provider,
            "status": status,
            "error": error_msg,
            "updated_at": _utc_now_iso(),
        }

        if doc_id not in entries or not isinstance(entries[doc_id], dict):
            entries[doc_id] = {"doc_id": doc_id}
        entries[doc_id]["sentiment"] = sentiment_entry

        if error_msg:
            failed.append({"doc_id": doc_id, "title": candidate["title"], "error": error_msg})

    enrichment_state["entries"] = entries
    if not args.dry_run and targets:
        enrichment_state["updated_at"] = _utc_now_iso()
        core._save_enrichment_state(
            storage,
            enrichment_state,
            require_remote=args.require_remote_persistence,
        )

    summary = {
        "mode": "score",
        "ran_at": _utc_now_iso(),
        "provider": provider,
        "source_kind": args.source_kind,
        "mode_selection": args.mode,
        "candidate_count": len(candidates),
        "selected_count": len(targets),
        "scored_count": scored_count,
        "fallback_scored_count": fallback_count,
        "used_models": used_models,
        "failed_count": len(failed),
        "failed": failed[:25],
        "dry_run": bool(args.dry_run),
        "remote_persistence": bool(storage is not None),
    }
    _write_summary(args.summary_path, summary)
    return summary


def _write_summary(path: str, payload: Dict[str, Any]) -> None:
    core._write_summary(path, payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentiment/tone scoring pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="Score editorial tone of ingested documents")
    score.add_argument("--source-kind", default="newsapi_article",
                       help="Filter to this source_kind (default: newsapi_article). Pass '' to score all.")
    score.add_argument("--mode", choices=["only_missing", "all"], default="only_missing",
                       help="only_missing: score docs that are unscored or previously fell back/failed. all: rescore everything.")
    score.add_argument("--doc-id", action="append", default=[],
                       help="Score specific doc IDs. Repeatable.")
    score.add_argument("--provider", choices=["openai", "deepseek"], default=os.getenv("SENTIMENT_PROVIDER", "deepseek"),
                       help="Model provider (default: deepseek).")
    score.add_argument("--model", default="",
                       help="Preferred model id (defaults to the provider's first candidate).")
    score.add_argument("--heuristic-only", action="store_true",
                       help="Skip LLM; use keyword heuristic only.")
    score.add_argument("--limit", type=int, default=None,
                       help="Max documents to score per run.")
    score.add_argument("--dry-run", action="store_true",
                       help="Score but do not persist results.")
    score.add_argument("--require-remote-persistence", action="store_true")
    score.add_argument("--summary-path", default="",
                       help="Write JSON run summary to this path.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.command == "score":
            summary = _run_score(args)
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2
    except Exception as e:
        error_payload = {"ok": False, "error": str(e), "command": args.command, "ran_at": _utc_now_iso()}
        _write_summary(getattr(args, "summary_path", ""), error_payload)
        print(json.dumps(error_payload, indent=2, ensure_ascii=False))
        return 1

    summary["ok"] = True
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
