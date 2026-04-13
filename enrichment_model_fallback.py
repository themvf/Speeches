from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Sequence


DEFAULT_CHAT_MODELS = [
    "gpt-5.1",
    "gpt-5-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
]

_MODEL_ACCESS_ERROR_MARKERS = (
    "model_not_found",
    "does not have access to model",
    "access to model",
)

_TRANSIENT_ERROR_MARKERS = (
    "rate limit",
    "429",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "connection reset",
    "connection aborted",
    "connection error",
    "server error",
    "bad gateway",
    "gateway timeout",
    "service unavailable",
    "overloaded",
    "try again",
)


def candidate_chat_models() -> List[str]:
    return list(DEFAULT_CHAT_MODELS)


def list_project_models(client: Any) -> List[str]:
    listed = client.models.list()
    return sorted({getattr(m, "id", "") for m in getattr(listed, "data", []) if getattr(m, "id", "")})


def get_accessible_chat_models(client: Any) -> List[str]:
    candidates = candidate_chat_models()
    try:
        ids = set(list_project_models(client))
        available = [model for model in candidates if model in ids]
        if available:
            return available
    except Exception:
        pass
    return candidates


def is_model_access_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _MODEL_ACCESS_ERROR_MARKERS)


def is_transient_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_ERROR_MARKERS)


def build_model_attempt_order(preferred_model: str, accessible_models: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    for item in [preferred_model, *list(accessible_models or []), *DEFAULT_CHAT_MODELS]:
        model = str(item or "").strip()
        if model and model not in ordered:
            ordered.append(model)
    return ordered


def run_with_model_fallback(
    *,
    client: Any,
    doc: Dict[str, Any],
    preferred_model: str,
    accessible_models: Sequence[str],
    run_agent: Callable[[Any, Dict[str, Any], str], Dict[str, Any]],
    max_attempts_per_model: int = 2,
    retry_delay_seconds: float = 0.75,
    sleep_fn: Callable[[float], None] | None = None,
) -> Dict[str, Any]:
    ordered_models = build_model_attempt_order(preferred_model, accessible_models)
    if not ordered_models:
        raise RuntimeError("No model available for enrichment.")

    if sleep_fn is None:
        sleep_fn = time.sleep

    attempts_per_model = max(1, int(max_attempts_per_model or 1))
    used_models: List[str] = []
    last_error: Exception | None = None

    for model_name in ordered_models:
        if model_name not in used_models:
            used_models.append(model_name)

        for attempt in range(1, attempts_per_model + 1):
            try:
                enrichment = run_agent(client, doc, model_name)
                return {
                    "enrichment": enrichment,
                    "model_used": model_name,
                    "used_models": list(used_models),
                }
            except Exception as exc:
                last_error = exc
                if attempt < attempts_per_model and is_transient_error(exc):
                    delay = max(0.0, float(retry_delay_seconds or 0.0)) * attempt
                    if delay > 0:
                        sleep_fn(delay)
                    continue
                break

    raise last_error or RuntimeError("No model available for enrichment.")
