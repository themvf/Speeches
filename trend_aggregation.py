#!/usr/bin/env python3
"""Daily trend aggregation pipeline.

Reads enrichment state + all corpus documents from GCS, computes tag-based
trends with embedding clustering, generates LLM descriptions, and writes
trends_daily.json back to GCS.

Usage:
    python trend_aggregation.py [--dry-run] [--limit N] [--min-mentions N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gcs_storage import GCSStorage
import run_financial_news_pipeline as _core

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

ENRICHMENT_STATE_BLOB = "document_enrichment_state.json"
CUSTOM_DOCS_BLOB = "custom_documents.json"
SEC_SPEECHES_BLOB = "all_speeches.json"
TRENDS_BLOB = "trends_daily.json"
TRENDS_LOCAL_PATH = DATA_DIR / "trends_daily.json"

CHAT_MODEL = "gpt-5.1"
CHAT_MODEL_FALLBACKS = ["gpt-4o", "gpt-4o-mini"]
DEFAULT_MIN_MENTIONS = 5

# Fixed taxonomy of top-level regulatory/financial trend categories.
# Keywords are matched via substring (multi-word) or whole-word (single-word) against tags.
# Longer phrase matches score higher, so "credit default swap" beats "swap" for specificity.
TAXONOMY: List[Dict] = [
    {
        "id": "crypto-digital-assets",
        "label": "Crypto & Digital Assets",
        "keywords": [
            "crypto", "cryptocurrency", "bitcoin", "ethereum", "blockchain",
            "defi", "decentralized finance", "nft", "non-fungible", "stablecoin",
            "cbdc", "central bank digital", "digital asset", "token offering",
            "virtual currency", "web3", "crypto exchange", "digital currency",
            "binance", "coinbase", "tether", "ripple",
        ],
    },
    {
        "id": "artificial-intelligence-technology",
        "label": "Artificial Intelligence & Technology",
        "keywords": [
            "artificial intelligence", "machine learning", "large language model",
            "generative ai", "fintech", "algorithmic trading", "robo-advisor",
            "ai governance", "ai regulation", "technology risk", "ai-powered",
            "predictive analytics", "automation", "ai model", "llm",
        ],
    },
    {
        "id": "private-credit-alternative-lending",
        "label": "Private Credit & Alternative Lending",
        "keywords": [
            "private credit", "direct lending", "leveraged loan",
            "collateralized loan", "clo", "middle market lending",
            "business development company", "bdc", "private debt",
            "alternative credit", "non-bank lending", "syndicated loan",
        ],
    },
    {
        "id": "securities-fraud-manipulation",
        "label": "Securities Fraud & Manipulation",
        "keywords": [
            "insider trading", "ponzi", "market manipulation",
            "front-running", "front running", "pump and dump",
            "securities fraud", "accounting fraud", "misrepresentation",
            "material non-public", "mnpi", "stock fraud",
            "investment fraud", "spoofing", "layering", "wash trading",
        ],
    },
    {
        "id": "investment-advisers-asset-management",
        "label": "Investment Advisers & Asset Management",
        "keywords": [
            "investment adviser", "hedge fund", "private equity",
            "registered investment", "mutual fund", "asset management",
            "fiduciary", "advisory fee", "conflicts of interest",
            "fund governance", "custody rule", "registered fund",
        ],
    },
    {
        "id": "market-structure-trading",
        "label": "Market Structure & Trading",
        "keywords": [
            "market structure", "dark pool", "order routing",
            "payment for order flow", "pfof", "market maker",
            "high frequency trading", "hft", "alternative trading system",
            "best execution", "national market system", "trading venue",
        ],
    },
    {
        "id": "esg-sustainable-finance",
        "label": "ESG & Sustainable Finance",
        "keywords": [
            "esg", "environmental social", "climate risk", "greenwashing",
            "sustainable investing", "climate disclosure", "net zero",
            "carbon emission", "climate change", "sustainability report",
            "tcfd", "green finance", "transition risk",
        ],
    },
    {
        "id": "corporate-disclosure-reporting",
        "label": "Corporate Disclosure & Reporting",
        "keywords": [
            "financial reporting", "gaap", "earnings guidance",
            "material disclosure", "restatement", "annual report",
            "quarterly report", "financial statement", "non-gaap",
            "earnings management", "forward-looking", "regulation s-k",
        ],
    },
    {
        "id": "derivatives-structured-products",
        "label": "Derivatives & Structured Products",
        "keywords": [
            "derivative", "interest rate swap", "credit default swap",
            "futures contract", "structured product", "securitization",
            "collateralized debt", "cdo", "swap dealer",
            "margin requirement", "commodity derivative",
        ],
    },
    {
        "id": "retail-investor-protection",
        "label": "Retail Investor Protection",
        "keywords": [
            "retail investor", "regulation best interest", "reg bi",
            "suitability", "investor protection", "investor education",
            "main street investor", "gamification", "meme stock",
            "robinhood", "fractional share",
        ],
    },
    {
        "id": "cybersecurity-operational-risk",
        "label": "Cybersecurity & Operational Risk",
        "keywords": [
            "cybersecurity", "cyber incident", "ransomware", "data breach",
            "operational resilience", "third party risk", "vendor risk",
            "cyber attack", "information security", "cloud risk",
            "cyber disclosure", "outsourcing risk",
        ],
    },
    {
        "id": "capital-formation-ipos",
        "label": "Capital Formation & IPOs",
        "keywords": [
            "initial public offering", "ipo", "spac",
            "special purpose acquisition", "regulation a", "regulation cf",
            "crowdfunding", "capital raising", "blank check",
            "direct listing", "de-spac", "secondary offering",
        ],
    },
    {
        "id": "fixed-income-rates",
        "label": "Fixed Income & Interest Rates",
        "keywords": [
            "fixed income", "interest rate", "credit spread", "sofr",
            "yield curve", "municipal bond", "corporate bond",
            "treasury bond", "bond market", "quantitative tightening",
            "rate hike", "duration risk",
        ],
    },
    {
        "id": "aml-financial-crime",
        "label": "AML & Financial Crime",
        "keywords": [
            "anti-money laundering", "aml", "know your customer", "kyc",
            "sanctions", "ofac", "financial crime", "fincen",
            "bank secrecy", "suspicious activity", "beneficial ownership",
            "customer due diligence", "terrorist financing",
        ],
    },
    {
        "id": "broker-dealer-regulation",
        "label": "Broker-Dealer Regulation",
        "keywords": [
            "broker-dealer", "broker dealer", "finra",
            "prime brokerage", "net capital", "margin lending",
            "securities lending", "registered representative",
            "broker supervision", "clearing", "settlement",
        ],
    },
    {
        "id": "corporate-governance",
        "label": "Corporate Governance",
        "keywords": [
            "corporate governance", "board of directors", "proxy voting",
            "shareholder activism", "executive compensation",
            "say on pay", "director independence", "audit committee",
            "board diversity", "dual class shares", "shareholder rights",
        ],
    },
    {
        "id": "global-cross-border-regulation",
        "label": "Global & Cross-Border Regulation",
        "keywords": [
            "cross-border", "cross border", "iosco", "foreign private issuer",
            "pcaob", "basel", "international regulatory", "extraterritorial",
            "mifid", "emir", "eu regulation", "global financial",
        ],
    },
    {
        "id": "banking-systemic-risk",
        "label": "Banking & Systemic Risk",
        "keywords": [
            "bank regulation", "capital requirement", "systemic risk",
            "stress test", "fdic", "fsoc", "too big to fail",
            "bank failure", "deposit insurance", "banking supervision",
            "liquidity coverage", "basel capital",
        ],
    },
    {
        "id": "sec-rulemaking-policy",
        "label": "SEC Rulemaking & Policy",
        "keywords": [
            "sec rulemaking", "proposed rule", "final rule",
            "comment period", "regulatory reform",
            "securities exchange act", "investment company act",
            "dodd-frank", "rulemaking agenda", "sec agenda",
        ],
    },
    {
        "id": "enforcement-actions",
        "label": "Enforcement Actions & Penalties",
        "keywords": [
            "enforcement action", "cease and desist", "disgorgement",
            "civil penalty", "settled charges", "administrative proceeding",
            "doj enforcement", "cftc enforcement", "regulatory penalty",
            "whistleblower", "deferred prosecution", "consent order",
        ],
    },
]
SPARKLINE_DAYS = 30
GROWTH_WINDOW_DAYS = 30
GROWTH_BASELINE_DAYS = 60  # the 30 days before the growth window

# Date parsing patterns (ordered by specificity)
_DATE_PATTERNS: List[Tuple[str, str]] = [
    # ISO: 2025-06-30, 2025-06-30T12:00:00Z
    (r"^\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
    # Month Day, Year: June 30, 2025
    (r"^[A-Za-z]+ \d{1,2},? \d{4}", None),
    # Month Year: June 2025
    (r"^[A-Za-z]+ \d{4}", None),
    # Year only: 2025
    (r"^\d{4}$", "%Y"),
]

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_date(raw: str) -> Optional[date]:
    """Parse a date string in any of the formats we see in the corpus."""
    if not raw:
        return None
    raw = raw.strip()

    # Try ISO prefix first
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # "Month Day, Year" or "Month Day Year"
    m = re.match(r"^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", raw)
    if m:
        month_str, day_str, year_str = m.group(1).lower(), m.group(2), m.group(3)
        month_num = _MONTH_NAMES.get(month_str)
        if month_num:
            try:
                return date(int(year_str), month_num, int(day_str))
            except ValueError:
                pass

    # "Month Year"
    m = re.match(r"^([A-Za-z]+)\s+(\d{4})$", raw)
    if m:
        month_str, year_str = m.group(1).lower(), m.group(2)
        month_num = _MONTH_NAMES.get(month_str)
        if month_num:
            try:
                return date(int(year_str), month_num, 1)
            except ValueError:
                pass

    # Year only
    m = re.match(r"^(\d{4})$", raw)
    if m:
        try:
            return date(int(m.group(1)), 1, 1)
        except ValueError:
            pass

    return None


def _load_gcs_json(storage: GCSStorage, blob_name: str) -> Any:
    try:
        blob = storage.bucket.blob(blob_name)
        if blob.exists():
            return json.loads(blob.download_as_text(encoding="utf-8"))
    except Exception as exc:
        _stderr(f"[warn] failed to load {blob_name}: {exc}")
    return None


def _save_gcs_json(storage: GCSStorage, blob_name: str, payload: Any) -> None:
    blob = storage.bucket.blob(blob_name)
    blob.upload_from_string(
        json.dumps(payload, indent=2, default=str),
        content_type="application/json",
    )


def _build_gcs_storage() -> Optional[GCSStorage]:
    """Build GCSStorage using the same credential-parsing logic as the other pipelines."""
    storage, error = _core._get_gcs_storage({})
    if storage is None:
        _stderr(f"[warn] GCS not configured: {error}")
    return storage



def _score_tag_for_category(tag_lower: str, keywords: List[str]) -> float:
    """Score a tag against a category's keyword list.

    Multi-word phrases score by their word count (longer = more specific).
    Single-word keywords require a whole-word match to avoid false positives.
    """
    score = 0.0
    for kw in keywords:
        if " " in kw:
            if kw in tag_lower:
                score += len(kw.split())
        else:
            if re.search(r"\b" + re.escape(kw) + r"\b", tag_lower):
                score += 1.0
    return score


def _map_to_taxonomy(
    tag_counts: Dict[str, int],
    taxonomy: List[Dict],
) -> Dict[str, List[str]]:
    """Assign each tag to its best-matching taxonomy category via keyword scoring.

    No embeddings required. Returns dict mapping taxonomy_id -> list of tags.
    Tags with zero score across all categories are dropped.
    """
    mapping: Dict[str, List[str]] = defaultdict(list)

    for tag in tag_counts:
        tag_lower = tag.lower()
        best_id: Optional[str] = None
        best_score = 0.0

        for tax in taxonomy:
            score = _score_tag_for_category(tag_lower, tax["keywords"])
            if score > best_score:
                best_score = score
                best_id = tax["id"]

        if best_id:
            mapping[best_id].append(tag)

    return mapping


def _tag_label(canonical_tag: str) -> str:
    """Convert a canonical tag string to a display label."""
    ACRONYMS = {"sec", "doj", "finra", "cftc", "fomc", "fdic", "occ", "cfpb",
                "aml", "kyc", "esg", "defi", "ai", "etf"}
    words = re.sub(r"[-_]+", " ", canonical_tag).split()
    result = []
    for w in words:
        if w.lower() in ACRONYMS:
            result.append(w.upper())
        else:
            result.append(w.capitalize())
    return " ".join(result)


def _trend_id(canonical_tag: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", canonical_tag.lower()).strip("-")


def _generate_description(
    client: Any,
    trend_label: str,
    summaries: List[str],
) -> str:
    """Generate a 2-sentence description for a trend using up to 3 doc summaries."""
    if client is None or not summaries:
        return f"Emerging trend around {trend_label} across regulatory and financial documents."

    snippets = "\n\n".join(f"- {s[:400]}" for s in summaries[:3])
    prompt = (
        f"You are a financial regulatory intelligence analyst. "
        f"The following are brief summaries from recent documents about the trend '{trend_label}':\n\n"
        f"{snippets}\n\n"
        f"Write exactly 2 concise sentences describing what is happening with this trend in the "
        f"regulatory and financial landscape. Be specific and authoritative. Do not use hedging language."
    )
    models_to_try = [CHAT_MODEL] + CHAT_MODEL_FALLBACKS
    for model in models_to_try:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            err_str = str(exc)
            # Only fall through to next model on access/not-found errors
            if "model_not_found" in err_str or "access" in err_str.lower() or "not have access" in err_str:
                _stderr(f"[warn] model {model} unavailable, trying next fallback")
                continue
            _stderr(f"[warn] description generation failed for '{trend_label}': {exc}")
            return f"Active regulatory focus on {trend_label} across multiple document sources."
    _stderr(f"[warn] all models failed for '{trend_label}'")
    return f"Active regulatory focus on {trend_label} across multiple document sources."


def _date_to_str(d: date) -> str:
    return d.isoformat()


def build_trends(
    enrichment_state: Dict[str, Any],
    custom_docs: List[Dict[str, Any]],
    sec_speeches: List[Dict[str, Any]],
    client: Any,
    min_mentions: int = DEFAULT_MIN_MENTIONS,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Core trend aggregation logic.

    Returns the full trends_daily.json payload.
    """
    today = date.today()
    growth_start = today - timedelta(days=GROWTH_WINDOW_DAYS)
    baseline_start = today - timedelta(days=GROWTH_WINDOW_DAYS + GROWTH_BASELINE_DAYS)
    sparkline_start = today - timedelta(days=SPARKLINE_DAYS - 1)

    # Build enrichment lookup: doc_id -> enrichment entry
    entries: Dict[str, Any] = enrichment_state.get("entries", {})

    # Collect all documents
    all_docs: List[Dict[str, Any]] = []
    all_docs.extend(custom_docs)
    all_docs.extend(sec_speeches)

    # Build doc_id -> doc metadata lookup
    doc_by_id: Dict[str, Dict[str, Any]] = {}
    for doc in all_docs:
        meta = doc.get("metadata", {})
        doc_id = meta.get("document_id", "")
        if doc_id:
            doc_by_id[doc_id] = doc

    # Per-document: resolve date + tags from enrichment
    # Structure: tag -> list of (date, doc_id, source_kind, summary)
    tag_occurrences: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    enriched_count = 0
    for doc_id, entry in entries.items():
        status = entry.get("status", "")
        if status != "enriched":
            continue

        enrichment = entry.get("enrichment", {})
        tags: List[str] = enrichment.get("tags", [])
        if not tags:
            continue

        # Resolve published date
        # Try enrichment date field first, then fall back to document metadata
        raw_date = entry.get("date", "")
        doc_date = _parse_date(raw_date)

        if doc_date is None:
            # Fall back to document metadata
            doc = doc_by_id.get(doc_id, {})
            meta = doc.get("metadata", {})
            for field in ("published_date", "date", "updated_date"):
                raw = meta.get(field, "")
                if raw:
                    doc_date = _parse_date(raw)
                    if doc_date:
                        break

        if doc_date is None:
            # Last resort: parse updated_at
            updated_at = entry.get("updated_at", "")
            doc_date = _parse_date(updated_at[:10]) if updated_at else None

        if doc_date is None:
            continue

        summary = enrichment.get("summary", "")
        source_kind = entry.get("doc_type", "") or ""
        doc = doc_by_id.get(doc_id, {})
        source_kind = doc.get("metadata", {}).get("source_kind", "") or source_kind

        enriched_count += 1
        for tag in tags:
            tag_clean = tag.strip().lower()
            if not tag_clean:
                continue
            tag_occurrences[tag_clean].append({
                "date": doc_date,
                "doc_id": doc_id,
                "source_kind": source_kind,
                "summary": summary,
            })

    _stderr(f"[info] Processed {enriched_count} enriched docs, found {len(tag_occurrences)} unique tags")

    # Count total mentions per tag
    tag_counts: Dict[str, int] = {tag: len(occurrences) for tag, occurrences in tag_occurrences.items()}

    # Filter to tags with >= min_mentions
    qualifying_tags = {tag: count for tag, count in tag_counts.items() if count >= min_mentions}
    _stderr(f"[info] {len(qualifying_tags)} tags with >= {min_mentions} mentions")

    if not qualifying_tags:
        return {
            "version": 1,
            "generated_at": _utc_now_iso(),
            "trend_count": 0,
            "trends": [],
        }

    # Map tags to taxonomy categories via keyword matching (no embeddings needed)
    taxonomy_map = _map_to_taxonomy(qualifying_tags, TAXONOMY)
    _stderr(f"[info] Tags mapped to {len(taxonomy_map)} taxonomy categories")

    taxonomy_by_id = {tax["id"]: tax for tax in TAXONOMY}

    trends = []
    for tax_id, cluster in taxonomy_map.items():
        tax = taxonomy_by_id.get(tax_id)
        if not tax or not cluster:
            continue

        # Canonical tag = most-mentioned tag in this category
        canonical = max(cluster, key=lambda t: qualifying_tags.get(t, 0))

        # Aggregate all occurrences across cluster
        all_occurrences: List[Dict[str, Any]] = []
        for tag in cluster:
            all_occurrences.extend(tag_occurrences.get(tag, []))

        total_mentions = len(all_occurrences)
        if total_mentions < min_mentions:
            continue

        # Date ranges
        all_dates = [occ["date"] for occ in all_occurrences]
        first_seen = min(all_dates)
        last_seen = max(all_dates)

        # Source kinds
        sources = sorted(set(occ["source_kind"] for occ in all_occurrences if occ["source_kind"]))

        # Growth calculation: recent window vs baseline window
        recent_count = sum(1 for occ in all_occurrences if occ["date"] >= growth_start)
        baseline_count = sum(
            1 for occ in all_occurrences
            if baseline_start <= occ["date"] < growth_start
        )
        if baseline_count == 0:
            growth_pct = 100.0 if recent_count > 0 else 0.0
        else:
            growth_pct = round((recent_count - baseline_count) / baseline_count * 100, 1)

        # Sparkline: daily counts for the last SPARKLINE_DAYS days
        sparkline_map: Dict[str, int] = defaultdict(int)
        for occ in all_occurrences:
            if occ["date"] >= sparkline_start:
                sparkline_map[_date_to_str(occ["date"])] += 1

        sparkline = []
        for i in range(SPARKLINE_DAYS):
            day = sparkline_start + timedelta(days=i)
            day_str = _date_to_str(day)
            sparkline.append({"date": day_str, "count": sparkline_map.get(day_str, 0)})

        # Top doc_ids: pick up to 5 most recent docs
        sorted_occs = sorted(all_occurrences, key=lambda o: o["date"], reverse=True)
        seen_ids: set = set()
        top_doc_ids: List[str] = []
        for occ in sorted_occs:
            doc_id = occ["doc_id"]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                top_doc_ids.append(doc_id)
            if len(top_doc_ids) >= 5:
                break

        # Collect summaries for description generation (most recent 3 docs with summaries)
        summaries_for_desc: List[str] = []
        for occ in sorted_occs:
            s = occ.get("summary", "")
            if s and len(summaries_for_desc) < 3:
                summaries_for_desc.append(s)
            if len(summaries_for_desc) >= 3:
                break

        label = tax["label"]
        trend_id = tax["id"]

        # Generate LLM description
        if dry_run:
            description = f"[dry-run] Emerging trend: {label}"
        else:
            description = _generate_description(client, label, summaries_for_desc)

        trends.append({
            "id": trend_id,
            "label": label,
            "canonical_tag": canonical,
            "cluster_tags": sorted(cluster),
            "description": description,
            "total_mentions": total_mentions,
            "recent_mentions": recent_count,
            "growth_pct": growth_pct,
            "first_seen": _date_to_str(first_seen),
            "last_seen": _date_to_str(last_seen),
            "sparkline": sparkline,
            "top_doc_ids": top_doc_ids,
            "sources": sources,
        })

    # Sort trends by total_mentions descending
    trends.sort(key=lambda t: -t["total_mentions"])

    return {
        "version": 1,
        "generated_at": _utc_now_iso(),
        "trend_count": len(trends),
        "trends": trends,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trend aggregation pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Skip OpenAI calls, write locally only")
    parser.add_argument("--min-mentions", type=int, default=DEFAULT_MIN_MENTIONS,
                        help=f"Minimum mentions to include a trend (default: {DEFAULT_MIN_MENTIONS})")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of trends in output (0 = no limit)")
    parser.add_argument("--local-output", action="store_true",
                        help="Write output locally even if GCS is available")
    args = parser.parse_args()

    # Initialize OpenAI
    client = None
    if not args.dry_run:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key and OpenAI is not None:
            client = OpenAI(api_key=api_key)
        else:
            _stderr("[warn] OPENAI_API_KEY not set or openai not installed; skipping LLM calls")

    # Load GCS storage
    storage = _build_gcs_storage()
    if storage is None and not args.dry_run:
        _stderr("[warn] GCS not configured; falling back to local files")

    # Load enrichment state
    _stderr("[info] Loading enrichment state...")
    enrichment_state: Dict[str, Any] = {}
    if storage:
        raw = _load_gcs_json(storage, ENRICHMENT_STATE_BLOB)
        if raw:
            enrichment_state = raw
    if not enrichment_state:
        local_path = DATA_DIR / ENRICHMENT_STATE_BLOB
        if local_path.exists():
            enrichment_state = json.loads(local_path.read_text(encoding="utf-8"))
    _stderr(f"[info] Enrichment entries: {len(enrichment_state.get('entries', {}))}")

    # Load custom documents
    _stderr("[info] Loading custom documents...")
    custom_docs: List[Dict[str, Any]] = []
    if storage:
        raw = _load_gcs_json(storage, CUSTOM_DOCS_BLOB)
        if raw:
            custom_docs = raw.get("documents", [])
    if not custom_docs:
        local_path = DATA_DIR / CUSTOM_DOCS_BLOB
        if local_path.exists():
            payload = json.loads(local_path.read_text(encoding="utf-8"))
            custom_docs = payload.get("documents", [])
    _stderr(f"[info] Custom documents: {len(custom_docs)}")

    # Load SEC speeches
    _stderr("[info] Loading SEC speeches...")
    sec_speeches: List[Dict[str, Any]] = []
    if storage:
        raw = _load_gcs_json(storage, SEC_SPEECHES_BLOB)
        if raw:
            sec_speeches = raw.get("speeches", [])
    if not sec_speeches:
        local_path = DATA_DIR / "all_speeches_final.json"
        if local_path.exists():
            payload = json.loads(local_path.read_text(encoding="utf-8"))
            sec_speeches = payload.get("speeches", [])
    _stderr(f"[info] SEC speeches: {len(sec_speeches)}")

    # Run aggregation
    _stderr("[info] Running trend aggregation...")
    payload = build_trends(
        enrichment_state=enrichment_state,
        custom_docs=custom_docs,
        sec_speeches=sec_speeches,
        client=client,
        min_mentions=args.min_mentions,
        dry_run=args.dry_run,
    )

    if args.limit and args.limit > 0:
        payload["trends"] = payload["trends"][: args.limit]
        payload["trend_count"] = len(payload["trends"])

    _stderr(f"[info] Generated {payload['trend_count']} trends")

    # Write output
    if storage and not args.dry_run and not args.local_output:
        _stderr(f"[info] Writing {TRENDS_BLOB} to GCS...")
        try:
            _save_gcs_json(storage, TRENDS_BLOB, payload)
            _stderr("[info] GCS write complete")
        except Exception as exc:
            _stderr(f"[error] GCS write failed: {exc}; writing locally")
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            TRENDS_LOCAL_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    else:
        _stderr(f"[info] Writing {TRENDS_LOCAL_PATH}...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        TRENDS_LOCAL_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    _stderr(f"[done] trend_count={payload['trend_count']}")


if __name__ == "__main__":
    main()
