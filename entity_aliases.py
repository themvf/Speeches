"""Entity normalization / alias map - Python side (see CLAUDE.md).

The alias data lives in apps/web/lib/server/entity-aliases.json: ONE shared
config consumed by this module (used by _normalize_enrichment_payload in
run_financial_news_pipeline.py and app.py, and by
backfill_entity_mentions.py) and by the TS side
(apps/web/lib/server/entity-aliases.ts, used by neon.ts's
prepareMentionBatch). Never fork per-language copies of the data; both sides
must collapse the same alias pairs to the same normalized value or
watchlist/attention/trend counts fragment across name variants.

The JSON lives under apps/web (rather than a top-level config/ dir) so the
Vercel build is guaranteed to include it; the Python workflows all run from
the repo root, where reading into apps/web is free.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ALIAS_CONFIG_PATH = (
    Path(__file__).resolve().parent / "apps" / "web" / "lib" / "server" / "entity-aliases.json"
)

# Byte-for-byte port of normalizeMentionValue in entity-aliases.ts (which is
# the normalization intelligence_mentions.normalized_value has always used).
# The two implementations are pinned together by mirrored unit tests
# (tests/test_entity_aliases.py and apps/web/lib/server/entity-aliases.test.ts
# assert the same input/output fixture pairs).
_QUOTES_RE = re.compile("['\"“”‘’]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_SPACES_RE = re.compile(r"\s+")


def normalize_mention_value(value: Any) -> str:
    text = str(value or "").lower()
    text = _QUOTES_RE.sub("", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    return _SPACES_RE.sub(" ", text).strip()


_alias_lookup: Dict[str, str] | None = None
_load_warned = False


def _load_alias_lookup() -> Dict[str, str]:
    """normalized alias -> canonical display label. Fail-soft: a missing or
    malformed config degrades to no aliasing (entities pass through
    unchanged) with a single stderr warning, rather than taking down the
    whole enrichment pipeline over a data file - but tests assert the real
    file loads, so a packaging regression still fails CI loudly."""
    global _alias_lookup, _load_warned
    if _alias_lookup is not None:
        return _alias_lookup
    lookup: Dict[str, str] = {}
    try:
        payload = json.loads(ALIAS_CONFIG_PATH.read_text(encoding="utf-8"))
        for entry in payload.get("entities", []):
            if not isinstance(entry, dict):
                continue
            canonical = str(entry.get("canonical", "") or "").strip()
            if not canonical:
                continue
            # Canonical labels map to themselves so already-canonical
            # mentions stay stable regardless of source casing/punctuation.
            aliases = entry.get("aliases", [])
            keys = [canonical] + (aliases if isinstance(aliases, list) else [])
            for alias in keys:
                key = normalize_mention_value(alias)
                if key and key not in lookup:
                    lookup[key] = canonical
    except Exception as exc:  # pragma: no cover - exercised via tests with a bad path
        if not _load_warned:
            print(f"[entity_aliases] failed to load {ALIAS_CONFIG_PATH}: {exc}", file=sys.stderr)
            _load_warned = True
        lookup = {}
    _alias_lookup = lookup
    return lookup


def canonical_entity_label(value: Any) -> str:
    """Raw entity text -> canonical display label ("Securities and Exchange
    Commission" -> "SEC"), or the trimmed input unchanged when no alias
    matches."""
    trimmed = str(value or "").strip()
    if not trimmed:
        return trimmed
    return _load_alias_lookup().get(normalize_mention_value(trimmed), trimmed)


def canonical_normalized_entity_value(value: Any) -> str:
    """Canonical normalized form - what intelligence_mentions.normalized_value
    should hold for this entity."""
    return normalize_mention_value(canonical_entity_label(value))


def entity_alias_pairs() -> List[Tuple[str, str, str]]:
    """Every (alias_normalized, canonical_label, canonical_normalized) where
    the alias differs from its canonical form - the work list for
    backfill_entity_mentions.py."""
    pairs: List[Tuple[str, str, str]] = []
    for alias_normalized, canonical_label in _load_alias_lookup().items():
        canonical_normalized = normalize_mention_value(canonical_label)
        if alias_normalized != canonical_normalized:
            pairs.append((alias_normalized, canonical_label, canonical_normalized))
    return pairs
