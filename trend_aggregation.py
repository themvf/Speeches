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
import math
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

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MODEL_FALLBACKS = ["text-embedding-ada-002", "text-embedding-3-large"]
CHAT_MODEL = "gpt-5.1"
CHAT_MODEL_FALLBACKS = ["gpt-4o", "gpt-4o-mini"]
DEFAULT_MIN_MENTIONS = 5
TAXONOMY_MATCH_THRESHOLD = 0.28  # minimum cosine similarity to assign a tag to a category

# Fixed taxonomy of top-level regulatory/financial trend categories.
# Each description is keyword-rich so embeddings land in the right semantic neighborhood.
TAXONOMY: List[Dict[str, str]] = [
    {
        "id": "crypto-digital-assets",
        "label": "Crypto & Digital Assets",
        "description": (
            "cryptocurrency bitcoin ethereum blockchain DeFi decentralized finance NFT "
            "non-fungible token stablecoin CBDC central bank digital currency digital asset "
            "token offering crypto exchange virtual currency Web3 crypto lending digital securities"
        ),
    },
    {
        "id": "artificial-intelligence-technology",
        "label": "Artificial Intelligence & Technology",
        "description": (
            "artificial intelligence AI machine learning large language model LLM fintech "
            "algorithmic trading automated trading robo-advisor technology risk AI governance "
            "predictive analytics natural language processing generative AI AI regulation"
        ),
    },
    {
        "id": "private-credit-alternative-lending",
        "label": "Private Credit & Alternative Lending",
        "description": (
            "private credit direct lending leveraged loans CLO collateralized loan obligation "
            "middle market lending business development company BDC private debt credit fund "
            "alternative credit non-bank lending syndicated loans"
        ),
    },
    {
        "id": "securities-fraud-manipulation",
        "label": "Securities Fraud & Manipulation",
        "description": (
            "insider trading Ponzi scheme market manipulation front running pump and dump "
            "securities fraud accounting fraud misrepresentation material non-public information "
            "MNPI stock fraud investment fraud spoofing layering wash trading"
        ),
    },
    {
        "id": "investment-advisers-asset-management",
        "label": "Investment Advisers & Asset Management",
        "description": (
            "investment adviser hedge fund private equity registered investment adviser RIA "
            "mutual fund registered fund asset management fiduciary duty advisory fees "
            "conflicts of interest fund governance custody rule"
        ),
    },
    {
        "id": "market-structure-trading",
        "label": "Market Structure & Trading",
        "description": (
            "exchange regulation dark pool order routing payment for order flow PFOF market maker "
            "high frequency trading HFT equities market liquidity trading venue ATS alternative "
            "trading system best execution national market system decimalization"
        ),
    },
    {
        "id": "esg-sustainable-finance",
        "label": "ESG & Sustainable Finance",
        "description": (
            "ESG environmental social governance climate risk greenwashing sustainable investing "
            "climate disclosure net zero carbon emissions climate change sustainability reporting "
            "TCFD transition risk physical risk green finance"
        ),
    },
    {
        "id": "corporate-disclosure-reporting",
        "label": "Corporate Disclosure & Reporting",
        "description": (
            "financial reporting GAAP earnings guidance material disclosure restatement 10-K 10-Q "
            "annual report quarterly report financial statement audit transparency Regulation S-K "
            "non-GAAP measures earnings management forward-looking statements"
        ),
    },
    {
        "id": "derivatives-structured-products",
        "label": "Derivatives & Structured Products",
        "description": (
            "derivatives swaps futures options structured products securitization ABS MBS "
            "collateralized debt obligation CDO interest rate swap credit default swap CDS "
            "commodity derivatives swap dealer margin requirements"
        ),
    },
    {
        "id": "retail-investor-protection",
        "label": "Retail Investor Protection",
        "description": (
            "retail investors Regulation Best Interest Reg BI suitability standard payment for "
            "order flow investor protection investor education best execution retail brokerage "
            "Robinhood gamification meme stocks main street investor"
        ),
    },
    {
        "id": "cybersecurity-operational-risk",
        "label": "Cybersecurity & Operational Risk",
        "description": (
            "cybersecurity cyber incident ransomware data breach operational resilience third party "
            "risk vendor risk incident response cyber attack information security DORA "
            "cyber disclosure rules cloud risk outsourcing risk"
        ),
    },
    {
        "id": "capital-formation-ipos",
        "label": "Capital Formation & IPOs",
        "description": (
            "IPO initial public offering SPAC special purpose acquisition company Reg A "
            "Regulation CF crowdfunding capital raising blank check company going public "
            "secondary offering direct listing de-SPAC transaction"
        ),
    },
    {
        "id": "fixed-income-rates",
        "label": "Fixed Income & Interest Rates",
        "description": (
            "bonds treasuries interest rates fixed income credit spreads SOFR duration yield curve "
            "municipal bonds corporate bonds government securities bond market rate hikes "
            "Federal Reserve monetary policy quantitative tightening"
        ),
    },
    {
        "id": "aml-financial-crime",
        "label": "AML & Financial Crime",
        "description": (
            "anti-money laundering AML know your customer KYC sanctions OFAC financial crime "
            "FinCEN Bank Secrecy Act suspicious activity reporting SAR beneficial ownership "
            "customer due diligence terrorist financing"
        ),
    },
    {
        "id": "broker-dealer-regulation",
        "label": "Broker-Dealer Regulation",
        "description": (
            "broker dealer FINRA clearing settlement custody prime brokerage net capital rule "
            "margin lending securities lending broker regulation registered representative "
            "SIPC protection broker supervision"
        ),
    },
    {
        "id": "corporate-governance",
        "label": "Corporate Governance",
        "description": (
            "board of directors corporate governance proxy voting shareholder activism "
            "executive compensation say on pay director independence audit committee "
            "board diversity dual class shares shareholder rights"
        ),
    },
    {
        "id": "global-cross-border-regulation",
        "label": "Global & Cross-Border Regulation",
        "description": (
            "cross border regulation IOSCO foreign private issuer PCAOB Basel III international "
            "regulatory cooperation extraterritorial jurisdiction foreign exchange EU regulation "
            "MiFID EMIR global financial standards"
        ),
    },
    {
        "id": "banking-systemic-risk",
        "label": "Banking & Systemic Risk",
        "description": (
            "bank regulation capital requirements systemic risk stress testing FDIC FSOC "
            "too big to fail bank failure deposit insurance Federal Reserve banking supervision "
            "Basel capital liquidity coverage ratio"
        ),
    },
    {
        "id": "sec-rulemaking-policy",
        "label": "SEC Rulemaking & Policy",
        "description": (
            "SEC rulemaking proposed rule final rule comment period regulatory reform "
            "Securities Exchange Act Investment Company Act Dodd-Frank rulemaking agenda "
            "regulatory agenda SEC agenda administrative law"
        ),
    },
    {
        "id": "enforcement-actions",
        "label": "Enforcement Actions & Penalties",
        "description": (
            "SEC enforcement action cease and desist disgorgement civil penalty settled charges "
            "administrative proceeding DOJ enforcement CFTC enforcement regulatory penalty "
            "whistleblower deferred prosecution agreement"
        ),
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


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed_tags(client: Any, tags: List[str]) -> Dict[str, List[float]]:
    """Embed a list of tag strings, trying models in order until one works.

    Returns a dict mapping tag -> embedding vector.
    """
    if not tags or client is None:
        return {}

    models_to_try = [EMBEDDING_MODEL] + EMBEDDING_MODEL_FALLBACKS
    active_model: Optional[str] = None

    # Probe with a single item to find a working model
    for model in models_to_try:
        try:
            client.embeddings.create(model=model, input=["probe"])
            active_model = model
            _stderr(f"[info] Using embedding model: {model}")
            break
        except Exception as exc:
            _stderr(f"[warn] Embedding model {model} unavailable: {exc}")

    if active_model is None:
        _stderr("[error] No embedding model available — trends will be empty")
        return {}

    embeddings: Dict[str, List[float]] = {}
    batch_size = 100
    for i in range(0, len(tags), batch_size):
        batch = tags[i : i + batch_size]
        try:
            response = client.embeddings.create(model=active_model, input=batch)
            for j, item in enumerate(response.data):
                embeddings[batch[j]] = item.embedding
        except Exception as exc:
            _stderr(f"[warn] embedding batch {i//batch_size} failed: {exc}")
    return embeddings


def _map_to_taxonomy(
    tag_counts: Dict[str, int],
    tag_embeddings: Dict[str, List[float]],
    taxonomy_embeddings: Dict[str, List[float]],
    threshold: float = TAXONOMY_MATCH_THRESHOLD,
) -> Dict[str, List[str]]:
    """Assign each tag to its best-matching taxonomy category.

    Returns dict mapping taxonomy_id -> list of tags assigned to it.
    Tags whose best match falls below threshold are dropped.
    """
    mapping: Dict[str, List[str]] = defaultdict(list)

    for tag in tag_counts:
        tag_vec = tag_embeddings.get(tag)
        if not tag_vec:
            continue

        best_id: Optional[str] = None
        best_sim = threshold  # must beat this to qualify

        for tax_id, tax_vec in taxonomy_embeddings.items():
            if not tax_vec:
                continue
            sim = _cosine_similarity(tag_vec, tax_vec)
            if sim > best_sim:
                best_sim = sim
                best_id = tax_id

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

    # Embed qualifying tags
    tag_list = sorted(qualifying_tags.keys())
    _stderr(f"[info] Embedding {len(tag_list)} tags...")
    if dry_run:
        embeddings: Dict[str, List[float]] = {}
    else:
        embeddings = _embed_tags(client, tag_list)
    _stderr(f"[info] Got embeddings for {len(embeddings)} tags")

    # Embed taxonomy category descriptions
    _stderr(f"[info] Embedding {len(TAXONOMY)} taxonomy categories...")
    if dry_run:
        taxonomy_embeddings: Dict[str, List[float]] = {}
    else:
        tax_texts = [tax["description"] for tax in TAXONOMY]
        tax_text_embeddings = _embed_tags(client, tax_texts)
        taxonomy_embeddings = {
            tax["id"]: tax_text_embeddings.get(tax["description"], [])
            for tax in TAXONOMY
        }

    # Map tags to taxonomy categories
    taxonomy_map = _map_to_taxonomy(qualifying_tags, embeddings, taxonomy_embeddings)
    _stderr(f"[info] Tags mapped to {len(taxonomy_map)} taxonomy categories")

    # In dry_run (no embeddings), fall back to assigning all tags to one bucket
    if dry_run and not taxonomy_map:
        taxonomy_map = {"sec-rulemaking-policy": list(qualifying_tags.keys())[:20]}

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
