"""VERA v5 — Session summarizer, intake quality gate, category consolidation.

Closes three v4.5 defects:
1. The layered-summary path was wired to an empty table — no summarizer ever
   ran, so vague queries returned ~nothing. This module produces an honest
   EXTRACTIVE summary (top user lines by salience — no LLM dependency) and a
   key_facts list. on_session_end stores the summary on the session row.
2. The extractor stored raw typo'd chat verbatim as "facts". A cheap
   fluency gate now rejects rambling/low-information sentences BEFORE they
   enter storage (existing security patterns untouched).
3. 26 fragmented categories. CATEGORY_CONSOLIDATION maps them to the
   canonical six: decisions, facts, preferences, corrections, research, tasks.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- summaries

_FILLER = frozenset({
    "hey", "hello", "hi", "thanks", "thank", "you", "ok", "okay", "yeah",
    "yes", "no", "cool", "nice", "good", "great", "please", "sorry",
})

_SALIENCE_HINTS = re.compile(
    r"\b(?:need|must|require|decided|decision|plan|upgrade|deploy|fix|build|"
    r"package|migrat\w*|supersede|embed\w*|recall|owner|prefers|correction|never|"
    r"always)\b"
)


def _sentence_score(text: str) -> float:
    """Cheap salience: length + hint density − filler ratio."""
    tokens = re.findall(r"[^\W_]+", text.lower())
    if not tokens:
        return 0.0
    filler = sum(1 for t in tokens if t in _FILLER)
    hints = sum(1 for t in tokens if _SALIENCE_HINTS.search(t))
    return hints * 2.0 + min(len(tokens), 30) / 10.0 - filler * 1.5


def summarize_session(messages: List[Dict[str, Any]], *, max_chars: int = 400) -> Dict[str, Any]:
    """Extractive summary from USER messages only (assistant text is never
    evidence). Returns {"summary": str, "key_facts": [str]}."""
    scored: List[tuple] = []
    for msg in messages or []:
        if msg.get("role") != "user":
            continue
        text = " ".join(str(msg.get("content", "")).split())
        if len(text) < 15:
            continue
        # Skip injected/system-looking blobs (cron wrappers etc.)
        if text.startswith("[") or "<memory-context>" in text[:60]:
            continue
        score = _sentence_score(text)
        if score > 1.0:
            scored.append((score, text))
    if not scored:
        return {"summary": "", "key_facts": []}
    scored.sort(reverse=True)
    top = [t for _, t in scored[:3]]
    summary = " | ".join(top)[:max_chars]
    key_facts = top[:2]
    return {"summary": summary, "key_facts": key_facts}


# ----------------------------------------------------------- clean intake

_MIN_CONTENT_TOKENS = 4
_MAX_FILLER_RATIO = 0.35
_RAMBLE_PUNCT = re.compile(r"(?:!{2,}|\?{2,}|\.{3,})")
_RAMBLE_TOKENS = frozenset({
    "umm", "uhh", "uh", "um", "hmm", "idk", "lol", "lmao", "gonna", "wanna",
    "kinda", "sorta", "like", "maybe", "whatever", "stuff", "things",
})


def is_fluent_declaration(sentence: str) -> bool:
    """Cheap fluency gate for intake. True = worth classifying at all."""
    lowered = sentence.lower().strip()
    tokens = re.findall(r"[^\W_]+", lowered)
    if len(tokens) < _MIN_CONTENT_TOKENS:
        return False
    filler = sum(1 for t in tokens if t in _RAMBLE_TOKENS or t in _FILLER)
    if filler / len(tokens) > _MAX_FILLER_RATIO:
        return False
    if _RAMBLE_PUNCT.search(lowered) and filler > 0:
        return False
    # Must contain at least one non-filler content word of length >= 3
    content_words = [t for t in tokens if t not in _RAMBLE_TOKENS and t not in _FILLER]
    return any(len(w) >= 3 for w in content_words)


# ------------------------------------------------- category consolidation

CATEGORY_CONSOLIDATION: Dict[str, str] = {
    # canonical six are fixed points
    "decisions": "decisions",
    "facts": "facts",
    "preferences": "preferences",
    "corrections": "corrections",
    "research": "research",
    "tasks": "tasks",
    # legacy fragments -> canonical
    "general": "facts",
    "user": "preferences",
    "owner_preferences": "preferences",
    "environment": "facts",
    "active_projects": "tasks",
    "key_decisions": "decisions",
    "workflow_conventions": "preferences",
    "current_task": "tasks",
    "conversation": "facts",
    "exploration": "research",
    "debugging": "tasks",
    "research_results": "research",
    "greetings": "facts",
    "one_off_commands": "tasks",
    "test": "facts",
    "security": "facts",
    "incident": "tasks",
    "backup": "tasks",
    "monitoring": "tasks",
    # live fragments found in a production DB audit (25 distinct categories)
    "protocols": "facts",
    "lessons": "facts",
    "projects": "tasks",
    "technical": "facts",
    "preference": "preferences",
    "correction": "corrections",
    "troubleshooting": "tasks",
    "recovery": "tasks",
    "memory": "facts",
    "incidents": "tasks",
    "evidence": "research",
    "security-policy": "preferences",
    "requirements": "decisions",
    "owner-feedback": "corrections",
    "milestones": "tasks",
    "configuration": "facts",
}


def consolidate_category(category: str) -> str:
    return CATEGORY_CONSOLIDATION.get(str(category or "").strip().lower(), "facts")
