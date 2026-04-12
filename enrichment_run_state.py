from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_nonnegative_int(value: Any, default: int = 0) -> int:
    try:
        num = int(value)
    except Exception:
        num = int(default)
    return max(0, num)


def _normalize_doc_ids(doc_ids: Any) -> List[str]:
    if not isinstance(doc_ids, (list, tuple, set)):
        return []

    out: List[str] = []
    seen = set()
    for item in doc_ids:
        doc_id = str(item or "").strip()
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def normalize_enrichment_run_state(state: Any) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {}

    remaining_doc_ids = _normalize_doc_ids(state.get("remaining_doc_ids", state.get("doc_ids", [])))
    processed = _coerce_nonnegative_int(state.get("processed", 0), default=0)
    fallback_count = _coerce_nonnegative_int(state.get("fallback_count", 0), default=0)
    skipped_count = _coerce_nonnegative_int(state.get("skipped_count", 0), default=0)
    total_selected = _coerce_nonnegative_int(
        state.get("total_selected", processed + len(remaining_doc_ids)),
        default=processed + len(remaining_doc_ids),
    )
    total_selected = max(total_selected, processed + len(remaining_doc_ids))

    status = str(state.get("status", "running") or "running").strip().lower()
    if status not in {"running", "completed", "aborted"}:
        status = "running"
    if not remaining_doc_ids and status == "running":
        status = "completed"

    out = {
        "run_id": str(state.get("run_id", "") or "").strip(),
        "run_type": str(state.get("run_type", "batch") or "batch").strip(),
        "scope_key": str(state.get("scope_key", "__all__") or "__all__").strip() or "__all__",
        "scope_label": str(state.get("scope_label", "All Organizations") or "All Organizations").strip()
        or "All Organizations",
        "model_name": str(state.get("model_name", "") or "").strip(),
        "mode": str(state.get("mode", "") or "").strip(),
        "status": status,
        "total_selected": total_selected,
        "processed": min(processed, total_selected),
        "fallback_count": fallback_count,
        "skipped_count": skipped_count,
        "remaining_doc_ids": remaining_doc_ids,
        "started_at": str(state.get("started_at", "") or "").strip(),
        "updated_at": str(state.get("updated_at", "") or "").strip(),
        "finished_at": str(state.get("finished_at", "") or "").strip(),
        "last_doc_id": str(state.get("last_doc_id", "") or "").strip(),
        "last_doc_title": str(state.get("last_doc_title", "") or "").strip(),
        "last_error": str(state.get("last_error", "") or "").strip(),
        "error": str(state.get("error", "") or "").strip(),
    }

    if not out["run_id"]:
        return {}
    return out


def create_enrichment_run_state(
    *,
    run_type: str,
    doc_ids: List[str],
    scope_key: str,
    scope_label: str,
    model_name: str,
    mode: str,
) -> Dict[str, Any]:
    now = _utc_now_iso()
    normalized_doc_ids = _normalize_doc_ids(doc_ids)
    return normalize_enrichment_run_state(
        {
            "run_id": f"enrich_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "run_type": str(run_type or "batch").strip() or "batch",
            "scope_key": str(scope_key or "__all__").strip() or "__all__",
            "scope_label": str(scope_label or "All Organizations").strip() or "All Organizations",
            "model_name": str(model_name or "").strip(),
            "mode": str(mode or "").strip(),
            "status": "running" if normalized_doc_ids else "completed",
            "total_selected": len(normalized_doc_ids),
            "processed": 0,
            "fallback_count": 0,
            "skipped_count": 0,
            "remaining_doc_ids": normalized_doc_ids,
            "started_at": now,
            "updated_at": now,
            "finished_at": now if not normalized_doc_ids else "",
            "last_doc_id": "",
            "last_doc_title": "",
            "last_error": "",
            "error": "",
        }
    )


def next_enrichment_run_doc_ids(state: Dict[str, Any], max_docs: int = 1) -> List[str]:
    normalized = normalize_enrichment_run_state(state)
    if not normalized or normalized.get("status") != "running":
        return []

    step_size = max(1, _coerce_nonnegative_int(max_docs, default=1))
    return list(normalized.get("remaining_doc_ids", []))[:step_size]


def advance_enrichment_run_state(
    state: Dict[str, Any],
    *,
    processed_doc_ids: List[str],
    fallback_count: int = 0,
    skipped_count: int = 0,
    last_doc_id: str = "",
    last_doc_title: str = "",
    last_error: str = "",
) -> Dict[str, Any]:
    normalized = normalize_enrichment_run_state(state)
    if not normalized:
        return {}

    done_doc_ids = _normalize_doc_ids(processed_doc_ids)
    remaining = list(normalized.get("remaining_doc_ids", []))
    for doc_id in done_doc_ids:
        if remaining and remaining[0] == doc_id:
            remaining.pop(0)
        elif doc_id in remaining:
            remaining.remove(doc_id)

    normalized["remaining_doc_ids"] = remaining
    normalized["processed"] = min(
        normalized["total_selected"],
        normalized["processed"] + len(done_doc_ids),
    )
    normalized["fallback_count"] += _coerce_nonnegative_int(fallback_count, default=0)
    normalized["skipped_count"] += _coerce_nonnegative_int(skipped_count, default=0)
    normalized["last_doc_id"] = str(last_doc_id or normalized.get("last_doc_id", "")).strip()
    normalized["last_doc_title"] = str(last_doc_title or normalized.get("last_doc_title", "")).strip()
    normalized["last_error"] = str(last_error or "").strip()
    normalized["updated_at"] = _utc_now_iso()

    if not remaining:
        normalized["status"] = "completed"
        normalized["finished_at"] = normalized["updated_at"]

    return normalize_enrichment_run_state(normalized)


def complete_enrichment_run_state(state: Dict[str, Any], error: str = "") -> Dict[str, Any]:
    normalized = normalize_enrichment_run_state(state)
    if not normalized:
        return {}

    now = _utc_now_iso()
    normalized["remaining_doc_ids"] = []
    normalized["status"] = "completed"
    normalized["error"] = str(error or "").strip()
    normalized["updated_at"] = now
    normalized["finished_at"] = now
    return normalize_enrichment_run_state(normalized)


def abort_enrichment_run_state(state: Dict[str, Any], error: str) -> Dict[str, Any]:
    normalized = normalize_enrichment_run_state(state)
    if not normalized:
        return {}

    now = _utc_now_iso()
    normalized["status"] = "aborted"
    normalized["error"] = str(error or "").strip()
    normalized["last_error"] = str(error or "").strip()
    normalized["updated_at"] = now
    normalized["finished_at"] = now
    return normalize_enrichment_run_state(normalized)
