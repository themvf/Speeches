#!/usr/bin/env python3
"""Shared helpers for classifying comment positions in policy documents."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


COMMENT_POSITION_LABELS = {"supportive", "opposed", "mixed", "neutral", "unclear", "not_applicable"}

COMMENT_POSITION_INSTRUCTION = (
    "For public comments and comment letters, classify the submitter's overall position toward the notice, "
    "proposal, or rulemaking as supportive, opposed, mixed, neutral, or unclear. "
    "Use supportive when support, adoption, retention, or implementation cues outweigh criticisms. "
    "Use opposed when the comment urges narrowing, abandoning, delaying, rescinding, or otherwise criticizes "
    "the proposal overall. Use mixed when it clearly supports some elements and opposes others. "
    "Use neutral when it mainly requests clarification or offers process feedback without taking a clear side. "
    "For all other documents, set comment_position to {label:not_applicable, confidence:0, rationale:''}. "
)

_COMMENT_POSITION_SOURCE_KINDS = {"finra_comment_letter", "regulations_gov_comment"}
_COMMENT_POSITION_DOC_TYPES = {"comment letter", "public comment"}

_WeightedRule = Tuple[str, str, int]

_COMMENT_SUPPORT_RULES: Sequence[_WeightedRule] = (
    (r"\bstrongly support\b", "strongly support", 3),
    (r"\bsupport(?:s|ed|ing)?\b", "support", 2),
    (r"\bin favor of\b", "in favor of", 2),
    (r"\bagree with\b", "agree with", 2),
    (r"\bendorse(?:s|d|ment)?\b", "endorse", 2),
    (r"\bcommend(?:s|ed)?\b", "commend", 1),
    (r"\bwelcome(?:s|d)?\b", "welcome", 1),
    (r"\bapprove(?:s|d)?\b", "approve", 1),
    (r"\bfavor(?:s|ed)?\b", "favor", 1),
    (
        r"\burg(?:e|es|ed|ing)\b[\s\S]{0,120}\b(preserve|implement|adopt|finalize|retain|recognize|move forward)\b",
        "urge preserve or adopt",
        3,
    ),
    (
        r"\b(recommend|encourage)(?:s|d|ing)?\b[\s\S]{0,120}\b(adopt|approve|finalize|retain|implement)\b",
        "recommend adoption",
        2,
    ),
)
_COMMENT_OPPOSE_RULES: Sequence[_WeightedRule] = (
    (r"\bstrongly oppose\b", "strongly oppose", 3),
    (r"\boppose(?:s|d|ition)?\b", "oppose", 2),
    (r"\bobject(?:s|ed)? to\b", "object to", 2),
    (r"\bdisagree with\b", "disagree with", 2),
    (r"\brecommend against\b", "recommend against", 2),
    (r"\bshould not\b", "should not", 1),
    (r"\breject(?:s|ed)?\b", "reject", 2),
    (r"\bharmful\b", "harmful", 1),
    (r"\bunworkable\b", "unworkable", 1),
    (r"\bburdensome\b", "burdensome", 1),
    (
        r"\burg(?:e|es|ed|ing)\b[\s\S]{0,120}\b(narrow|abandon|withdraw|rescind|repeal|delay|pause|reconsider)\b",
        "urge narrow or abandon",
        3,
    ),
    (r"\bcritical of\b", "critical of", 2),
    (r"\bwarn(?:s|ed|ing)? that\b", "warn that", 1),
    (r"\bexceeds? statutory authority\b", "exceeds statutory authority", 2),
    (r"\bwithout clear congressional authorization\b", "without congressional authorization", 2),
    (r"\blacks? statutory authority\b", "lacks statutory authority", 2),
)
_COMMENT_MIXED_RULES: Sequence[_WeightedRule] = (
    (r"\bwhile we support\b", "while we support", 3),
    (r"\bsupport\b[\s\S]{0,160}\bbut\b", "support ... but", 3),
    (r"\bsupport\b[\s\S]{0,160}\bhowever\b", "support ... however", 3),
    (r"\bsupport\b[\s\S]{0,160}\bconcern(?:s)?\b", "support with concerns", 3),
    (r"\bcommend\b[\s\S]{0,160}\bbut\b", "commend ... but", 2),
    (r"\bsupport\b[\s\S]{0,160}\brecommend(?:s|ed|ing)?\b", "support with recommendations", 2),
)
_COMMENT_NEUTRAL_RULES: Sequence[_WeightedRule] = (
    (r"\brequest(?:s|ed)? clarification\b", "request clarification", 1),
    (r"\bseek(?:s|ing)? clarification\b", "seek clarification", 1),
    (r"\brecommend(?:s|ed|ing)? clarification\b", "recommend clarification", 1),
    (r"\bsuggest(?:s|ed|ing)? changes\b", "suggest changes", 1),
    (r"\bprovide feedback\b", "provide feedback", 1),
    (r"\boffer(?:s|ed|ing)? comments\b", "offer comments", 1),
)


def is_comment_position_document(doc: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(doc, dict):
        return False
    source_kind = str(doc.get("source_kind", "") or "").strip().lower()
    doc_type = str(doc.get("doc_type", "") or "").strip().lower()
    if source_kind in _COMMENT_POSITION_SOURCE_KINDS:
        return True
    if doc_type in _COMMENT_POSITION_DOC_TYPES:
        return True
    return source_kind.endswith("_comment") and "comment" in doc_type


def _collect_weighted_hits(text: str, rules: Sequence[_WeightedRule], limit: int = 4) -> Tuple[List[str], int]:
    hits: List[str] = []
    seen = set()
    score = 0
    for pattern, label, weight in rules:
        if label in seen:
            continue
        if re.search(pattern, text):
            seen.add(label)
            hits.append(label)
            score += int(weight)
            if len(hits) >= limit:
                break
    return hits, score


def infer_comment_position(doc: Optional[Dict[str, Any]], text: str) -> Dict[str, Any]:
    if not is_comment_position_document(doc):
        return {"label": "not_applicable", "confidence": 0.0, "rationale": ""}

    lower = str(text or "").lower()
    support_hits, support_score = _collect_weighted_hits(lower, _COMMENT_SUPPORT_RULES)
    oppose_hits, oppose_score = _collect_weighted_hits(lower, _COMMENT_OPPOSE_RULES)
    mixed_hits, mixed_score = _collect_weighted_hits(lower, _COMMENT_MIXED_RULES, limit=3)
    neutral_hits, neutral_score = _collect_weighted_hits(lower, _COMMENT_NEUTRAL_RULES, limit=3)

    if mixed_hits or (support_score and oppose_score and abs(support_score - oppose_score) <= 1):
        label = "mixed"
        confidence = 0.8 if mixed_hits else 0.72
        cues = mixed_hits or (support_hits + oppose_hits)[:4]
        rationale = "Detected material support and opposition cues: " + ", ".join(cues[:4])
    elif support_score and (not oppose_score or support_score >= oppose_score + 2):
        label = "supportive"
        confidence = min(0.92, 0.54 + 0.05 * support_score + 0.04 * max(0, support_score - oppose_score))
        cues = support_hits[:3]
        if oppose_hits:
            rationale = "Supportive cues outweigh opposing cues: " + ", ".join((support_hits + oppose_hits)[:4])
        else:
            rationale = "Supportive cues: " + ", ".join(cues)
    elif oppose_score:
        label = "opposed"
        confidence = min(0.92, 0.54 + 0.05 * oppose_score + 0.04 * max(0, oppose_score - support_score))
        cues = oppose_hits[:3]
        if support_hits:
            rationale = "Opposing cues outweigh supportive cues: " + ", ".join((oppose_hits + support_hits)[:4])
        else:
            rationale = "Opposing cues: " + ", ".join(cues)
    elif neutral_score:
        label = "neutral"
        confidence = min(0.7, 0.4 + 0.08 * len(neutral_hits))
        rationale = "Neutral/request-for-clarification cues: " + ", ".join(neutral_hits[:3])
    else:
        label = "unclear"
        confidence = 0.2
        rationale = "No clear support or opposition cues detected."

    if label == "mixed" and mixed_score >= 3 and confidence < 0.8:
        confidence = 0.8

    return {"label": label, "confidence": round(confidence, 2), "rationale": rationale[:240]}
