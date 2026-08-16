"""CAPITAL_FORMATION vocabulary - Python side (see CLAUDE.md).

The data lives in apps/web/lib/capital-formation-vocabulary.json: ONE shared
config consumed by this module (neon_feeds.DEFAULT_TOPIC_RULES) and by the TS
side (apps/web/lib/capital-formation-vocabulary.ts, which feeds
topic-rule-recommendations.ts, theme-intelligence.ts, gdelt-doc.ts,
stored-category-evidence.ts, and intelbeta-dashboard.tsx).

Both writers seed the same rss_topic_rules table, so a forked copy here means
the live rule depends on which side initialized the database. That already
happened once - this module is the fix.

The JSON lives under apps/web (rather than a top-level config/ dir) so the
Vercel build is guaranteed to include it; the Python workflows all run from
the repo root, where reading into apps/web is free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

VOCABULARY_PATH = (
    Path(__file__).resolve().parent
    / "apps"
    / "web"
    / "lib"
    / "capital-formation-vocabulary.json"
)

# Mirrors the fallback in the TS reader's consumers: if the config cannot be
# read we degrade to the pre-consolidation keyword set rather than seeding an
# empty rule, which would silently stop routing capital formation articles.
_FALLBACK_KEYWORDS = [
    "capital formation",
    "initial public offering",
    "public offering",
    "private placement",
    "exempt offering",
    "regulation crowdfunding",
]

_payload: Dict[str, Any] | None = None
_load_warned = False


def _load() -> Dict[str, Any]:
    """Fail-soft: a missing or malformed config degrades to the fallback
    keyword list with a single stderr warning rather than taking down topic
    seeding over a data file - but tests assert the real file loads, so a
    packaging regression still fails CI loudly."""
    global _payload, _load_warned
    if _payload is not None:
        return _payload
    try:
        _payload = json.loads(VOCABULARY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exercised via tests with a bad path
        if not _load_warned:
            print(
                f"[capital_formation_vocabulary] failed to load {VOCABULARY_PATH}: {exc}",
                file=sys.stderr,
            )
            _load_warned = True
        _payload = {}
    return _payload


def topic_key() -> str:
    return str(_load().get("topicKey", "") or "CAPITAL_FORMATION")


def label() -> str:
    return str(_load().get("label", "") or "Capital Formation")


def sort_order() -> int:
    try:
        return int(_load().get("sortOrder", 12))
    except (TypeError, ValueError):
        return 12


def keywords() -> List[str]:
    raw = _load().get("keywords")
    if not isinstance(raw, list) or not raw:
        return list(_FALLBACK_KEYWORDS)
    return [str(item).strip() for item in raw if str(item).strip()]


def keywords_csv() -> str:
    """Comma-separated lowercase keywords, the exact shape rss_topic_rules
    stores and parseKeywords() in intel-topic-matching.ts expects."""
    return ", ".join(keyword.lower() for keyword in keywords())


def focus_area_ids() -> List[str]:
    areas = _load().get("focusAreas")
    if not isinstance(areas, list):
        return []
    return [str(area.get("id", "")) for area in areas if isinstance(area, dict) and area.get("id")]
