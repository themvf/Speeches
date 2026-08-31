#!/usr/bin/env python3
"""Run a source-bounded OpenAI/DeepSeek editorial-package pilot.

This is a manual evaluation utility, not the production scheduled workflow.
It reads the public production feed, freezes one curated 24-hour AI source
snapshot, sends the identical editorial prompt to both providers, and writes
comparison artifacts beneath tmp_artifacts/.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import html
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import requests


FEED_URL = "https://speeches-zeta.vercel.app/api/intel/feed"
OPENAI_MODEL = os.environ.get("EDITORIAL_PILOT_OPENAI_MODEL", "gpt-5.6-luna").strip()
DEEPSEEK_MODEL = "deepseek-v4-pro"
TOPIC_LABEL = "Artificial Intelligence"
MAX_SOURCES = 16
REQUEST_TIMEOUT_SECONDS = 180


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "editorial_recommendation",
        "candidates",
        "selected_package",
        "quality_warnings",
    ],
    "properties": {
        "editorial_recommendation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["decision", "selected_candidate_id", "rationale"],
            "properties": {
                "decision": {"type": "string", "enum": ["publish", "no_publish"]},
                "selected_candidate_id": {"type": ["string", "null"]},
                "rationale": {"type": "string"},
            },
        },
        "candidates": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "candidate_id",
                    "working_title",
                    "subtitle",
                    "thesis",
                    "reader_promise",
                    "why_now",
                    "original_contribution",
                    "counterargument",
                    "uncertainties",
                    "what_to_watch",
                    "supporting_source_ids",
                    "recap_risk",
                    "support_score",
                    "originality_score",
                ],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "working_title": {"type": "string"},
                    "subtitle": {"type": "string"},
                    "thesis": {"type": "string"},
                    "reader_promise": {"type": "string"},
                    "why_now": {"type": "string"},
                    "original_contribution": {"type": "string"},
                    "counterargument": {"type": "string"},
                    "uncertainties": {"type": "array", "items": {"type": "string"}},
                    "what_to_watch": {"type": "array", "items": {"type": "string"}},
                    "supporting_source_ids": {"type": "array", "items": {"type": "string"}},
                    "recap_risk": {"type": "string", "enum": ["low", "medium", "high"]},
                    "support_score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "originality_score": {"type": "integer", "minimum": 1, "maximum": 5},
                },
            },
        },
        "selected_package": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": [
                "candidate_id",
                "opening_hooks",
                "outline",
                "claim_ledger",
                "author_questions",
                "source_notes",
            ],
            "properties": {
                "candidate_id": {"type": "string"},
                "opening_hooks": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "string"},
                },
                "outline": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["heading", "purpose", "source_ids"],
                        "properties": {
                            "heading": {"type": "string"},
                            "purpose": {"type": "string"},
                            "source_ids": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "claim_ledger": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "claim",
                            "claim_type",
                            "supporting_source_ids",
                            "support_status",
                            "confidence",
                            "caveat",
                        ],
                        "properties": {
                            "claim": {"type": "string"},
                            "claim_type": {
                                "type": "string",
                                "enum": ["fact", "inference", "opinion", "prediction"],
                            },
                            "supporting_source_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "support_status": {
                                "type": "string",
                                "enum": ["supported", "partially_supported", "unsupported"],
                            },
                            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                            "caveat": {"type": "string"},
                        },
                    },
                },
                "author_questions": {"type": "array", "items": {"type": "string"}},
                "source_notes": {"type": "array", "items": {"type": "string"}},
            },
        },
        "quality_warnings": {"type": "array", "items": {"type": "string"}},
    },
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def strip_html(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_url(value: str) -> str:
    parts = urlsplit(value)
    kept_query = [
        (key, val)
        for key, val in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"mod", "syn-25a6b1a6"}
    ]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), urlencode(kept_query), ""))


def fetch_ai_sources(window_end: datetime) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    window_start = window_end - timedelta(hours=24)
    response = requests.get(
        FEED_URL,
        params={"limit": 400, "since": window_start.isoformat()},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    articles = payload.get("data", {}).get("articles", [])

    explicit_ai = re.compile(
        r"(?:\bAI\b|A\.I\.|\bartificial intelligence\b|\bOpenAI\b|\bAnthropic\b|"
        r"\bChatGPT\b|\bClaude\b|\bmachine learning\b|\blarge language model\b|\bLLM\b|"
        r"\bAI agent\b|\bAI model\b|\bAI infrastructure\b|\bAI job\b|\bAI era\b|"
        r"\bAI boom\b|\brobot(?:s|ics)?\b)",
        re.IGNORECASE,
    )
    low_value_title = re.compile(
        r"(?:AI-picked|big analyst AI moves|seek AI alternatives)",
        re.IGNORECASE,
    )
    relevant_non_ai_titles = {
        "The companies desperate to hire graduates",
        "Meta Tests Robots That Can Swap Cables And Reset Servers At Its Data Centers",
    }

    selected: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for article in articles:
        topics = article.get("topics") or []
        title = strip_html(article.get("title"))
        description = strip_html(article.get("description"))
        if TOPIC_LABEL not in topics:
            continue
        if not explicit_ai.search(title) and title not in relevant_non_ai_titles:
            continue
        if low_value_title.search(title):
            continue
        url = canonical_url(str(article.get("url") or ""))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        selected.append(
            {
                "source_id": f"rss:{article.get('id')}",
                "title": title,
                "description": description[:1600],
                "publisher": strip_html(article.get("feed_label") or article.get("feed_key")),
                "published_at": article.get("published_at") or article.get("fetched_at"),
                "url": url,
                "topics": topics,
                "access_note": "Captured headline and feed description only; full article was not fetched for this pilot.",
            }
        )
        if len(selected) >= MAX_SOURCES:
            break

    snapshot_meta = {
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "feed_generated_at": payload.get("data", {}).get("generatedAt"),
        "source_count": len(selected),
        "selection_note": (
            "Filtered to feed items tagged Artificial Intelligence with an explicit AI term in the captured "
            "headline/description; removed canonical-URL duplicates and obvious promotional market content."
        ),
    }
    return selected, snapshot_meta


def prompt_messages(sources: list[dict[str, Any]], snapshot_meta: dict[str, Any]) -> list[dict[str, str]]:
    schema_text = json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, separators=(",", ":"))
    source_text = json.dumps(sources, ensure_ascii=False, indent=2)
    developer = (
        "You are the Daily AI Column Editor for a financial and regulatory intelligence publication. "
        "Create an editorial decision package, not a finished article and not a link roundup. Work only from "
        "the supplied captured headlines and descriptions. Never imply that you read full articles. Never invent "
        "facts, quotations, first-person experience, or the author's opinion. Distinguish fact from inference and "
        "prediction. Prefer a coherent, non-obvious thesis that matters to professionals following AI, financial "
        "markets, regulation, governance, and enterprise risk. A no_publish decision is valid when the evidence is "
        "too weak or derivative. Return JSON only and conform to the supplied schema."
    )
    user = (
        "Generate tonight's source-bounded editorial package. Propose up to three genuinely different angles and "
        "select the strongest. Every factual claim must cite one or more source_id values. Mark claims only "
        "partially supported when the captured description is insufficient. Questions requiring personal judgment "
        "must be directed to the human author rather than answered on the author's behalf.\n\n"
        f"WINDOW:\n{json.dumps(snapshot_meta, ensure_ascii=False, indent=2)}\n\n"
        f"OUTPUT JSON SCHEMA:\n{schema_text}\n\n"
        f"FROZEN SOURCES:\n{source_text}"
    )
    return [{"role": "developer", "content": developer}, {"role": "user", "content": user}]


def parse_openai_response(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    texts: list[str] = []
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                texts.append(content["text"])
    return "\n".join(texts)


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a provider JSON response, tolerating only markdown fences."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Provider response was not a JSON object")
    return parsed


def run_openai(messages: list[dict[str, str]]) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    started = time.perf_counter()
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": OPENAI_MODEL,
            "input": messages,
            "reasoning": {"effort": "medium"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "editorial_package",
                    "strict": True,
                    "schema": OUTPUT_SCHEMA,
                }
            },
            "max_output_tokens": 10000,
            "store": False,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if not response.ok:
        error_payload = response.json().get("error", {}) if response.headers.get("content-type", "").startswith("application/json") else {}
        error_code = str(error_payload.get("code") or "request_failed")
        raise RuntimeError(f"OpenAI HTTP {response.status_code}: {error_code}")
    raw = response.json()
    content = parse_openai_response(raw)
    return {
        "provider": "openai",
        "model": raw.get("model", OPENAI_MODEL),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": raw.get("usage", {}),
        "package": json.loads(content),
    }


def run_deepseek(messages: list[dict[str, str]]) -> dict[str, Any]:
    api_key = (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API") or "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY/DEEPSEEK_API is not configured")
    started = time.perf_counter()
    compatible_messages = [
        {**message, "role": "system" if message.get("role") == "developer" else message.get("role")}
        for message in messages
    ]
    request_body = {
        "model": DEEPSEEK_MODEL,
        "messages": compatible_messages,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "response_format": {"type": "json_object"},
        "max_tokens": 10000,
    }
    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=request_body,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if not response.ok:
        error_payload = response.json().get("error", {}) if response.headers.get("content-type", "").startswith("application/json") else {}
        error_code = str(error_payload.get("code") or "request_failed")
        error_type = str(error_payload.get("type") or "unknown_error")
        raise RuntimeError(f"DeepSeek HTTP {response.status_code}: {error_code} ({error_type})")
    raw = response.json()
    content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        package = parse_json_object(content)
    except json.JSONDecodeError:
        repair_messages = compatible_messages + [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "Your preceding response was not valid JSON. Return the same editorial package repaired as "
                    "one valid JSON object conforming exactly to the supplied schema. Output JSON only, with no "
                    "markdown fences or commentary."
                ),
            },
        ]
        repair_response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": DEEPSEEK_MODEL,
                "messages": repair_messages,
                "thinking": {"type": "disabled"},
                "response_format": {"type": "json_object"},
                "max_tokens": 10000,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if not repair_response.ok:
            raise RuntimeError(f"DeepSeek JSON repair HTTP {repair_response.status_code}")
        repair_raw = repair_response.json()
        content = repair_raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        package = parse_json_object(content)
        raw["usage_repair"] = repair_raw.get("usage", {})
    return {
        "provider": "deepseek",
        "model": raw.get("model", DEEPSEEK_MODEL),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": raw.get("usage", {}),
        "package": package,
    }


def validate_source_references(result: dict[str, Any], valid_ids: set[str]) -> list[str]:
    warnings: list[str] = []
    package = result.get("package", {})
    referenced: list[str] = []
    for candidate in package.get("candidates", []):
        referenced.extend(candidate.get("supporting_source_ids", []))
    selected = package.get("selected_package") or {}
    for section in selected.get("outline", []):
        referenced.extend(section.get("source_ids", []))
    for claim in selected.get("claim_ledger", []):
        referenced.extend(claim.get("supporting_source_ids", []))
    unknown = sorted({value for value in referenced if value not in valid_ids})
    if unknown:
        warnings.append(f"Unknown source IDs referenced: {', '.join(unknown)}")
    for claim in selected.get("claim_ledger", []):
        if claim.get("claim_type") == "fact" and not claim.get("supporting_source_ids"):
            warnings.append(f"Fact has no source: {claim.get('claim', '')[:120]}")
    return warnings


def markdown_for_result(label: str, result: dict[str, Any]) -> str:
    package = result["package"]
    recommendation = package["editorial_recommendation"]
    lines = [
        f"## {label}",
        "",
        f"- Model: `{result['model']}`",
        f"- Latency: {result['latency_ms'] / 1000:.1f}s",
        f"- Decision: **{recommendation['decision']}**",
        f"- Rationale: {recommendation['rationale']}",
        "",
        "### Candidate angles",
        "",
    ]
    for candidate in package.get("candidates", []):
        selected_mark = " — **selected**" if candidate["candidate_id"] == recommendation.get("selected_candidate_id") else ""
        lines.extend(
            [
                f"#### {candidate['working_title']}{selected_mark}",
                "",
                f"*{candidate['subtitle']}*",
                "",
                f"**Thesis:** {candidate['thesis']}",
                "",
                f"**Reader promise:** {candidate['reader_promise']}",
                "",
                f"**Why now:** {candidate['why_now']}",
                "",
                f"**Original contribution:** {candidate['original_contribution']}",
                "",
                f"**Counterargument:** {candidate['counterargument']}",
                "",
                f"**Scores:** support {candidate['support_score']}/5; originality {candidate['originality_score']}/5; recap risk {candidate['recap_risk']}",
                "",
            ]
        )
    selected = package.get("selected_package")
    if selected:
        lines.extend(["### Opening hooks", ""])
        lines.extend(f"- {item}" for item in selected.get("opening_hooks", []))
        lines.extend(["", "### Outline", ""])
        for section in selected.get("outline", []):
            lines.append(f"- **{section['heading']}:** {section['purpose']}")
        lines.extend(["", "### Questions for the author", ""])
        lines.extend(f"- {item}" for item in selected.get("author_questions", []))
    if package.get("quality_warnings") or result.get("validation_warnings"):
        lines.extend(["", "### Warnings", ""])
        lines.extend(f"- {item}" for item in package.get("quality_warnings", []))
        lines.extend(f"- {item}" for item in result.get("validation_warnings", []))
    return "\n".join(lines)


def main() -> int:
    snapshot_path_value = os.environ.get("EDITORIAL_PILOT_SNAPSHOT", "").strip()
    if snapshot_path_value:
        frozen_payload = json.loads(Path(snapshot_path_value).read_text(encoding="utf-8"))
        sources = frozen_payload["sources"]
        snapshot_meta = frozen_payload["metadata"]
        window_end = datetime.fromisoformat(snapshot_meta["window_end"])
    else:
        window_end = utc_now()
        sources, snapshot_meta = fetch_ai_sources(window_end)
        frozen_payload = {"metadata": snapshot_meta, "sources": sources}
    if len(sources) < 3:
        raise RuntimeError(f"Only {len(sources)} eligible AI sources were found; refusing to generate")

    hash_payload = json.loads(json.dumps(frozen_payload))
    hash_payload.get("metadata", {}).pop("snapshot_hash", None)
    serialized = json.dumps(hash_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    snapshot_hash = snapshot_meta.get("snapshot_hash") or hashlib.sha256(serialized).hexdigest()
    snapshot_meta["snapshot_hash"] = snapshot_hash
    messages = prompt_messages(sources, snapshot_meta)

    date_label = window_end.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    output_suffix = os.environ.get("EDITORIAL_PILOT_OUTPUT_SUFFIX", "").strip()
    if output_suffix and not re.fullmatch(r"[A-Za-z0-9_-]+", output_suffix):
        raise RuntimeError("EDITORIAL_PILOT_OUTPUT_SUFFIX may contain only letters, numbers, underscores, and hyphens")
    output_name = f"editorial_pilot_{date_label}" + (f"_{output_suffix}" if output_suffix else "")
    output_dir = Path("tmp_artifacts") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_snapshot.json").write_text(
        json.dumps(frozen_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "prompt.json").write_text(
        json.dumps({"messages": messages, "schema": OUTPUT_SCHEMA}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_runners = {"openai": run_openai, "deepseek": run_deepseek}
    requested = {
        value.strip().lower()
        for value in os.environ.get("EDITORIAL_PILOT_PROVIDERS", "openai,deepseek").split(",")
        if value.strip()
    }
    unknown_providers = requested - set(all_runners)
    if unknown_providers:
        raise RuntimeError(f"Unknown providers: {', '.join(sorted(unknown_providers))}")
    runners = {name: all_runners[name] for name in requested}
    results: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for name in set(all_runners) - requested:
        existing_path = output_dir / f"{name}_output.json"
        if existing_path.exists():
            results[name] = json.loads(existing_path.read_text(encoding="utf-8"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(runners))) as executor:
        futures = {executor.submit(runner, messages): name for name, runner in runners.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                result["validation_warnings"] = validate_source_references(
                    result, {source["source_id"] for source in sources}
                )
                results[name] = result
                (output_dir / f"{name}_output.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as exc:  # provider errors must not discard the successful output
                errors[name] = f"{type(exc).__name__}: {exc}"

    comparison_lines = [
        "# Daily AI Editorial Provider Pilot",
        "",
        f"- Window: {snapshot_meta['window_start']} through {snapshot_meta['window_end']}",
        f"- Frozen sources: {snapshot_meta['source_count']}",
        f"- Snapshot hash: `{snapshot_hash}`",
        "- Source limitation: captured headlines and feed descriptions only; no claim of full-article review.",
        "",
    ]
    for name in ("openai", "deepseek"):
        if name in results:
            comparison_lines.append(markdown_for_result(name.title(), results[name]))
            comparison_lines.append("")
        else:
            comparison_lines.extend([f"## {name.title()}", "", f"Generation failed: {errors.get(name, 'unknown error')}", ""])
    (output_dir / "comparison.md").write_text("\n".join(comparison_lines), encoding="utf-8")
    (output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "snapshot_hash": snapshot_hash,
                "source_count": len(sources),
                "successful_providers": sorted(results),
                "errors": errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({
        "output_dir": str(output_dir.resolve()),
        "snapshot_hash": snapshot_hash,
        "source_count": len(sources),
        "successful_providers": sorted(results),
        "errors": errors,
    }, indent=2))
    return 0 if results else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Pilot failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
