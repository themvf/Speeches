#!/usr/bin/env python3
"""
Standalone sentiment/tone scoring pipeline.

Workflow: extraction -> enrichment -> sentiment (this script)

Reads full text from custom_documents.json, scores the author's editorial
tone using an LLM, and writes results back into document_enrichment_state.json
under a top-level 'sentiment' key per entry — leaving enrichment untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import tomllib
except Exception:
    tomllib = None

from gcs_storage import GCSStorage

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
STREAMLIT_SECRETS_PATH = ROOT / ".streamlit" / "secrets.toml"

CUSTOM_DOCS_BLOB_NAME = "custom_documents.json"
CUSTOM_DOCS_LOCAL_PATH = DATA_DIR / "custom_documents.json"
ENRICHMENT_STATE_BLOB_NAME = "document_enrichment_state.json"
ENRICHMENT_STATE_LOCAL_PATH = DATA_DIR / "document_enrichment_state.json"

SENTIMENT_LABELS = {"positive", "negative", "neutral"}

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


def _stderr(message: str) -> None:
    print(str(message), file=sys.stderr)


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _coerce_int(value: Any, default: int = 0, min_value: int = 0) -> int:
    try:
        num = int(value)
    except Exception:
        num = int(default)
    return max(int(min_value), num)


def _sanitize_api_key(value: Any) -> str:
    key = str(value or "").strip()
    if not key:
        return ""
    if len(key) >= 2 and key[0] == key[-1] and key[0] in {"'", '"'}:
        key = key[1:-1].strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key


def _load_streamlit_secrets() -> Dict[str, Any]:
    if tomllib is None or not STREAMLIT_SECRETS_PATH.exists():
        return {}
    try:
        with open(STREAMLIT_SECRETS_PATH, "rb") as f:
            payload = tomllib.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _nested_get(payload: Dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _get_openai_api_key(secrets_payload: Dict[str, Any]) -> str:
    return _sanitize_api_key(
        os.getenv("OPENAI_API_KEY", "")
        or _nested_get(secrets_payload, "openai", "api_key")
        or ""
    )


def _get_gcs_storage(secrets_payload: Dict[str, Any]) -> Tuple[Optional[GCSStorage], str]:
    bucket_name = str(
        os.getenv("GCS_BUCKET_NAME", "")
        or _nested_get(secrets_payload, "gcs", "bucket_name")
        or ""
    ).strip()

    credentials_info = None
    credentials_path = str(
        os.getenv("GCS_CREDENTIALS_PATH", "")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        or ""
    ).strip()

    if credentials_path:
        try:
            with open(credentials_path, "r", encoding="utf-8") as f:
                credentials_info = json.load(f)
        except Exception as e:
            return None, f"Failed to read GCS credentials file: {e}"
    else:
        secret_gcs = _nested_get(secrets_payload, "gcs")
        if isinstance(secret_gcs, dict) and bucket_name:
            credentials_info = {k: v for k, v in secret_gcs.items() if k != "bucket_name"}

    if not bucket_name or not isinstance(credentials_info, dict) or not credentials_info:
        return None, "GCS bucket/credentials not configured"

    try:
        return GCSStorage(bucket_name, credentials_info), ""
    except Exception as e:
        return None, f"Failed to initialize GCS storage: {e}"


def _load_json_blob(
    storage: Optional[GCSStorage],
    blob_name: str,
    local_path: Path,
    default: Any = None,
) -> Any:
    if storage is not None:
        try:
            blob = storage.bucket.blob(blob_name)
            if blob.exists():
                payload = json.loads(blob.download_as_text(encoding="utf-8"))
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(local_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                return payload
        except Exception as e:
            _stderr(f"Remote load failed for {blob_name}: {e}")
    if local_path.exists():
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default if default is not None else {}


def _save_json_blob(
    storage: Optional[GCSStorage],
    blob_name: str,
    local_path: Path,
    payload: Any,
    require_remote: bool = False,
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if storage is None:
        if require_remote:
            raise RuntimeError(f"Remote persistence required for {blob_name}, but GCS is not configured.")
        return
    storage.bucket.blob(blob_name).upload_from_string(
        json.dumps(payload, indent=2, ensure_ascii=False),
        content_type="application/json",
    )


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


def _extract_first_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start: end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _score_with_llm(client: Any, model: str, title: str, text: str) -> Dict[str, Any]:
    if len(text) > 60000:
        text = text[:40000] + "\n\n[...TRUNCATED...]\n\n" + text[-10000:]

    prompt = f"Title: {title}\n\nArticle:\n{text}"

    for attempt in range(1, 3):
        instruction = SENTIMENT_SYSTEM_PROMPT
        if attempt > 1:
            instruction += " Respond with raw JSON only. No markdown, no code fences."
        response = client.responses.create(model=model, instructions=instruction, input=prompt)
        raw_text = getattr(response, "output_text", None) or ""
        if not raw_text and hasattr(response, "model_dump"):
            for item in (response.model_dump() or {}).get("output", []):
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") in ("output_text", "text") and c.get("text"):
                            raw_text = c["text"]
                            break
        parsed = _extract_first_json(raw_text)
        if parsed:
            return _normalize_sentiment(parsed)

    raise RuntimeError("LLM did not return parseable JSON after 2 attempts.")


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


def _candidate_models() -> List[str]:
    return ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1", "gpt-4o"]


def _get_openai_client(secrets_payload: Dict[str, Any]) -> Optional[Any]:
    api_key = _get_openai_api_key(secrets_payload)
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        _stderr(f"Failed to initialize OpenAI client: {e}")
        return None


def _is_model_access_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "model_not_found" in msg or "does not have access to model" in msg


def _write_summary(path: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _run_score(args: argparse.Namespace) -> Dict[str, Any]:
    secrets_payload = _load_streamlit_secrets()
    storage, gcs_status = _get_gcs_storage(secrets_payload)
    if args.require_remote_persistence and storage is None:
        raise RuntimeError(gcs_status)

    custom_payload = _load_json_blob(storage, CUSTOM_DOCS_BLOB_NAME, CUSTOM_DOCS_LOCAL_PATH, {"documents": []})
    enrichment_state = _load_json_blob(storage, ENRICHMENT_STATE_BLOB_NAME, ENRICHMENT_STATE_LOCAL_PATH, {"entries": {}})
    if not isinstance(enrichment_state, dict):
        enrichment_state = {"entries": {}}
    entries = enrichment_state.setdefault("entries", {})

    doc_ids: List[str] = [str(d).strip() for d in (args.doc_id or []) if str(d).strip()]

    candidates = _build_candidates(
        custom_payload=custom_payload,
        source_kind=args.source_kind,
        doc_ids=doc_ids or None,
    )

    if not doc_ids and args.mode == "only_missing":
        candidates = [
            c for c in candidates
            if not isinstance(entries.get(c["doc_id"], {}).get("sentiment"), dict)
        ]

    limit = len(candidates) if args.limit is None else max(0, int(args.limit))
    targets = candidates[:limit] if limit > 0 else []

    client = None if args.heuristic_only else _get_openai_client(secrets_payload)
    preferred_model = args.model or _candidate_models()[0]
    accessible_models = _candidate_models()

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
                raise RuntimeError("OpenAI client unavailable.")
            ordered = [preferred_model] + [m for m in accessible_models if m != preferred_model]
            sentiment = None
            last_error = None
            for model_name in ordered:
                try:
                    sentiment = _score_with_llm(client, model_name, candidate["title"], candidate["full_text"])
                    model_used = model_name
                    break
                except Exception as e:
                    last_error = e
                    if not _is_model_access_error(e):
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
        _save_json_blob(
            storage,
            ENRICHMENT_STATE_BLOB_NAME,
            ENRICHMENT_STATE_LOCAL_PATH,
            enrichment_state,
            require_remote=args.require_remote_persistence,
        )

    summary = {
        "mode": "score",
        "ran_at": _utc_now_iso(),
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sentiment/tone scoring pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="Score editorial tone of ingested documents")
    score.add_argument("--source-kind", default="newsapi_article",
                       help="Filter to this source_kind (default: newsapi_article). Pass '' to score all.")
    score.add_argument("--mode", choices=["only_missing", "all"], default="only_missing",
                       help="only_missing: skip already-scored docs (default). all: rescore everything.")
    score.add_argument("--doc-id", action="append", default=[],
                       help="Score specific doc IDs. Repeatable.")
    score.add_argument("--model", default="",
                       help="Preferred OpenAI model (default: gpt-4.1-mini).")
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
