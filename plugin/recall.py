"""VERA v5 — Proactive recall trigger layer.

Closes the v4 core defect: recall was passive. prefetch() only reacted to the
literal words of the current message; nothing pulled project context when work
started. This module adds a deterministic, LLM-free intent detector plus one
bounded extra retrieval that runs INSIDE prefetch when a work-start signal
fires.

Design constraints honored:
- No LLM calls, no network, no threads. Pure regex + token sets.
- Everything flows through retriever.search() so SQLite governance
  (trust tiers, retrieval_enabled, principal visibility) stays in force.
- The proactive block is clearly labeled "[proactive]" and bounded (~400 chars)
  so it cannot flood the context budget.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

if __package__:
    from .retriever import search, format_for_context
else:
    from retriever import search, format_for_context  # type: ignore

logger = logging.getLogger(__name__)

# Work-start / continuation signals (deterministic, case-insensitive).
WORK_START_PATTERNS = (
    r"\blet'?s\s+(?:start|begin|get into|work on)\b",
    r"\b(?:start|begin|continue|resuming|resume|pick\s+up|finish|work\s+on)\b"
    r"(?!\s*(?:over|again\s+from\s+scratch))",
    r"\bcontinu(?:e|ing)\b",
    r"\bwhere\s+(?:did\s+we|we)\s+(?:leave[d]?\s+off|stop(?:ped)?|end(?:ed)?)\b",
    r"\b(?:carry\s+on|back\s+to|returning\s+to)\b",
    r"\b(?:execut|implement|build|deploy|upgrad|migrat)\w*\b",
)

STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "now", "lets", "let", "us", "we", "i", "you", "it",
    "its", "this", "that", "these", "those", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "doing", "would", "could", "ought", "im", "dont", "didnt", "whats",
    "hey", "hi", "hello", "thanks", "thank", "ok", "okay", "yeah", "yes",
    "please", "work", "working", "start", "starting", "continue",
    "continuing", "today", "tonight", "tomorrow", "still", "also",
})

MAX_PROJECT_TERMS = 6
PROACTIVE_MAX_CHARS = 400


def detect_intent(
    message: str,
    known_projects: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Return {"kind": "work_start", "project_terms": [...]} or None.

    Deterministic. Fires on work-start/continuation phrasing; extracts
    candidate project nouns from the message, preferring names that match
    known_projects (categories/projects already present in VERA).
    Smalltalk never fires.
    """
    if not message or not isinstance(message, str):
        return None
    text = message.strip()
    if len(text) < 8:
        return None

    lowered = text.lower()
    fired = False
    for pattern in WORK_START_PATTERNS:
        if re.search(pattern, lowered):
            fired = True
            break
    if not fired:
        return None

    # Neutral openers must not trigger even if a pattern word appears.
    tokens = [t for t in re.findall(r"[a-z0-9][\w.-]*", lowered)]
    content = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if not content:
        return None

    # Candidate project terms: content words, hyphenated/dotted kept whole.
    candidates: List[str] = []
    known_lower = {p.lower() for p in (known_projects or [])}
    # Prefer known project names first (exact token match).
    for token in content:
        if token in known_lower and token not in candidates:
            candidates.append(token)
    # Then remaining content words (skip bare numbers unless version-like).
    for token in content:
        if token in candidates:
            continue
        if re.fullmatch(r"\d+", token) and not re.search(r"v\d|ver\d|\d\.\d", lowered):
            continue
        candidates.append(token)
        if len(candidates) >= MAX_PROJECT_TERMS:
            break

    if not candidates:
        return None
    return {"kind": "work_start", "project_terms": candidates}


def expand_prefetch(db, query: str, *, semantic_backend: Any = None,
                    authenticated: bool = False, principal_id: Optional[int] = None,
                    agent_identity: str = "",
                    max_chars: int = PROACTIVE_MAX_CHARS) -> str:
    """One bounded extra retrieval when a work-start signal fires.

    Returns a labeled block ("[proactive] ...") or "" when nothing fired /
    nothing found. Uses retriever.search() so all governance applies.
    """
    try:
        raw_cats = db.get_categories() or []
        known = sorted({
            (c.get("category") if isinstance(c, dict) else str(c)) or ""
            for c in raw_cats
        } - {""})
    except Exception:
        known = []
    intent = detect_intent(query, known_projects=known)
    if not intent:
        return ""

    # Resolve the agent's own default principal when none was supplied —
    # mirrors handle_tool_call's fallback so proactive recall sees the same
    # rows an authenticated vera_search would (governance stays intact).
    if authenticated and principal_id is None:
        try:
            row = db._get_conn().execute(
                "SELECT id FROM principals WHERE platform='hermes' AND platform_subject_id=? "
                "AND status='active'",
                (agent_identity or "default",),
            ).fetchone()
            if row is not None:
                principal_id = int(row["id"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("VERA v5 recall: principal resolution skipped: %s", exc)

    probe_terms = intent["project_terms"]
    probe_query = " ".join(probe_terms[:MAX_PROJECT_TERMS])
    try:
        results = search(
            db,
            probe_query,
            limit=5,
            semantic_backend=semantic_backend,
            principal_id=principal_id,
            authenticated=bool(authenticated and principal_id is not None),
        )
    except Exception as exc:  # noqa: BLE001 — proactive pull must never break prefetch
        logger.debug("VERA v5 proactive recall failed quietly: %s", exc)
        return ""
    if not results:
        return ""
    # Truncate-per-entry: a single long memory must not void the whole block
    # (format_for_context skips entries that exceed the WHOLE budget).
    lines = []
    used = 0
    for m in results[:3]:
        content = " ".join(str(m.get("content", "")).split())
        if not content:
            continue
        per_entry = max(80, max_chars // 3)
        if len(content) > per_entry:
            content = content[:per_entry - 3].rstrip() + "..."
        entry = f"- [{m.get('category', 'general')}; evidence/{m.get('trust_tier', 'unverified')}] > {content}"
        created = str(m.get("created_at", ""))[:10]
        if created:
            entry += f" ({created})"
        if used + len(entry) > max_chars and lines:
            break
        lines.append(f"[proactive] {entry}")
        used += len(entry)
    return "\n".join(lines)
    labeled = "\n".join(
        f"[proactive] {line}" if line.strip() else line
        for line in block.splitlines()
    )
    logger.info(
        "VERA v5: proactive recall fired for terms=%s (%d memories)",
        probe_terms[:3], min(3, len(results)),
    )
    return labeled
