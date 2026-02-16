#!/usr/bin/env python3
"""
Speaker parsing helpers.

The SEC listing sometimes returns multiple linked names in one table cell and
BeautifulSoup text extraction can collapse them into a single token stream
(e.g., "Gary GenslerHester M. Peirce..."). These helpers normalize and split
speaker strings for consistent display and counting.
"""

from __future__ import annotations

import re
from typing import List


_PERSON_RE = re.compile(
    r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:\s+[A-Z]\.)?\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+)\b"
)
_PERSON_BEFORE_COMMA_RE = re.compile(
    r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+(?:\s+[A-Z]\.)?\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]+)\s*,"
)

_NON_PERSON_FIRST_TOKENS = {
    "Acting",
    "Chair",
    "Chairman",
    "Commissioner",
    "Division",
    "Office",
    "Department",
    "Committee",
    "Commission",
    "Staff",
    "Federal",
    "Securities",
    "Exchange",
    "Treasury",
    "Financial",
    "General",
    "Chief",
    "Director",
    "Markets",
    "SEC",
    "U.S",
    "US",
    "Corporation",
}


def _is_initial_token(token: str) -> bool:
    t = token.strip()
    if t.endswith("."):
        t = t[:-1]
    return len(t) == 1 and t.isalpha() and t.isupper()


def _is_capitalized_token(token: str) -> bool:
    if not token:
        return False
    t = token.strip(".,;:")
    return bool(t) and t[0].isupper()


def _is_likely_person_candidate(candidate: str) -> bool:
    tokens = [t.strip(".,;:") for t in candidate.split() if t.strip(".,;:")]
    if not tokens:
        return False

    if len(tokens[0]) <= 2:
        return False

    if tokens[0] in _NON_PERSON_FIRST_TOKENS:
        return False

    # Reject organization-role fragments like "Grewal Director".
    if any(token in _NON_PERSON_FIRST_TOKENS for token in tokens[1:]):
        return False

    return True


def normalize_speaker_text(raw_speaker: str) -> str:
    """Normalize speaker text formatting from SEC listing/content."""
    if not raw_speaker:
        return ""

    text = str(raw_speaker).replace("\xa0", " ")
    text = re.sub(r"\[\d+\]", "", text)
    # Split collapsed names like "...GenslerHester..."
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;")
    return text


def extract_speakers(raw_speaker: str) -> List[str]:
    """Extract one or more likely person names from a speaker string."""
    text = normalize_speaker_text(raw_speaker)
    if not text:
        return []

    # Explicit delimiter path: authoritative for values we normalize from SEC links.
    if ";" in text:
        out: List[str] = []
        seen = set()
        for part in [p.strip() for p in text.split(";") if p.strip()]:
            part = re.sub(
                r"\b(?:Chair|Chairman|Acting Chair|Acting Chairman|Commissioner)\s+",
                "",
                part,
            ).strip()
            if "," in part:
                part = part.split(",", 1)[0].strip()
            if part and part not in seen:
                out.append(part)
                seen.add(part)
        if out:
            return out

    text_for_names = re.sub(
        r"\b(?:Chair|Chairman|Acting Chair|Acting Chairman|Commissioner)\s+",
        "",
        text,
    )

    names: List[str] = []
    seen = set()

    # Strong signal: name token immediately before a comma.
    for match in _PERSON_BEFORE_COMMA_RE.finditer(text_for_names):
        candidate = match.group(1).strip()
        if not _is_likely_person_candidate(candidate):
            continue
        if candidate not in seen:
            names.append(candidate)
            seen.add(candidate)

    if names:
        return names

    # Fallback for no-comma strings (often concatenated names).
    tokens = [t for t in text_for_names.split() if t]
    if len(tokens) >= 2:
        memo = {}

        def parse_from(idx: int):
            if idx == len(tokens):
                return (0.0, [])
            if idx in memo:
                return memo[idx]

            best = (-1e9, [])

            def consider(length: int, score_bonus: float):
                nonlocal best
                if idx + length > len(tokens):
                    return
                chunk = tokens[idx:idx + length]
                if any(not _is_capitalized_token(tok) for tok in chunk):
                    return
                if _is_initial_token(chunk[0]):
                    return
                if length == 2 and _is_initial_token(chunk[1]):
                    return
                if length == 3:
                    if _is_initial_token(chunk[1]):
                        pass
                    else:
                        if _is_initial_token(chunk[2]):
                            return
                candidate = " ".join(chunk)
                if not _is_likely_person_candidate(candidate):
                    return
                rest_score, rest_names = parse_from(idx + length)
                if rest_score < -1e8:
                    return
                total = score_bonus + rest_score
                names = [candidate] + rest_names
                if total > best[0]:
                    best = (total, names)

            # Prefer segmentations that can consume middle-initial names but
            # still allow plain first/last and three-token names.
            consider(3, 1.25 if idx + 1 < len(tokens) and _is_initial_token(tokens[idx + 1]) else 1.05)
            consider(2, 1.0)

            memo[idx] = best
            return best

        score, parsed_names = parse_from(0)
        if score > 0 and parsed_names:
            unique = []
            seen = set()
            for name in parsed_names:
                if name not in seen:
                    unique.append(name)
                    seen.add(name)
            return unique

    # Regex fallback for any remaining odd formats.
    for match in _PERSON_RE.finditer(text_for_names):
        candidate = match.group(1).strip()
        prev = text_for_names[max(0, match.start() - 4):match.start()].lower()

        # Skip organization fragments (e.g., "... of General Counsel").
        if prev.endswith("of "):
            continue
        if not _is_likely_person_candidate(candidate):
            continue

        if candidate not in seen:
            names.append(candidate)
            seen.add(candidate)

    if names:
        return names

    # Fallback: keep original normalized value as one speaker entry.
    return [text]


def format_speakers(raw_speaker: str) -> str:
    """Return display-friendly speaker text."""
    speakers = extract_speakers(raw_speaker)
    return "; ".join(speakers) if speakers else ""


def primary_speaker(raw_speaker: str) -> str:
    """Return primary speaker for single-value groupings/charts."""
    speakers = extract_speakers(raw_speaker)
    return speakers[0] if speakers else ""
