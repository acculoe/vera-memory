"""VERA v4.5 — Persistent Working Memory.

Two components:
1. PersistentMemoryManager — manages persistence.md (between-session memory)
2. WorkingMemoryBuffer — in-memory buffer for within-session continuity

Together they solve the "amnesia" problem:
- Between sessions: persistence.md is injected via prefetch() (relevant entries only)
- Within sessions: WorkingMemoryBuffer holds recent operations (instant recall)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # v5.5: dictation repair lives with the Jev gate, not duplicated here.
    if __package__:
        from .classify import clean_dictation as classify_clean
    else:
        from classify import clean_dictation as classify_clean
except Exception:  # pragma: no cover - degrade to raw text, never block
    def classify_clean(text: str) -> str:
        return " ".join((text or "").split())

logger = logging.getLogger(__name__)

# Categories that PERSIST across sessions
PERSIST_CATEGORIES = frozenset({
    "owner_preferences",
    "corrections",
    "environment",
    "active_projects",
    "key_decisions",
    "workflow_conventions",
    "current_task",
    "preferences",
})

# Categories that are EPHEMERAL (never persist)
EPHEMERAL_CATEGORIES = frozenset({
    "conversation",
    "exploration",
    "debugging",
    "research_results",
    "greetings",
    "one_off_commands",
    "test",
})

# Content signals that suggest persistence
PERSIST_SIGNALS = (
    "always", "never", "must", "required", "rule:",
    "preference:", "correction:", "owner said", "owner wants",
    "convention:", "project:", "phase", "deployed", "configured",
    "environment:", "host:", "model:", "platform:",
    "use ", "do not ", "don't ", "instead of",
)

# Content signals that suggest ephemerality
EPHEMERAL_SIGNALS = (
    "ran command", "checked", "explored", "loaded skill",
    "searched for", "found", "attempted", "tried",
    "hello", "hey", "thanks", "ok", "sure",
    "screenshot", "captured", "scrolled",
)

MAX_PERSISTENCE_CHARS = 2500
# v5.5: per-entry cap raised from 200 and made sentence-aware. The old
# `entry[:197] + "..."` chopped mid-word, so every long entry ended in
# unusable fragments ("...keep all the stuff related to bu..."). A memory
# that cannot be read on sight is not working memory.
MAX_ENTRY_CHARS = 400
# v5.5: seed/placeholder bullets that carry no information. They were being
# round-tripped forever and, worse, the briefing reported "Active task:
# (none)" as if it were real state.
PLACEHOLDER_ENTRIES = frozenset({
    "(none)",
    "none",
    "n/a",
    "tbd",
    "todo",
    "-",
    "(add your preferences here)",
    "(add your active task here)",
    "add your preferences here",
})


def _truncate_entry(entry: str, limit: int = MAX_ENTRY_CHARS) -> str:
    """Truncate at a sentence or word boundary — never mid-word."""
    text = " ".join((entry or "").split())
    if len(text) <= limit:
        return text
    window = text[:limit]
    # Prefer the last sentence end inside the window.
    for stop in (". ", "! ", "? ", "; ", ", "):
        idx = window.rfind(stop)
        if idx >= int(limit * 0.5):
            return window[: idx + 1].rstrip()
    # Otherwise fall back to the last whole word.
    return (window.rsplit(" ", 1)[0] if " " in window else window).rstrip(" ,;:") + "…"


def _is_placeholder(entry: str) -> bool:
    """True for seed bullets that should never be treated as real memory."""
    return " ".join((entry or "").split()).casefold().strip(" .") in PLACEHOLDER_ENTRIES
SECTION_HEADERS = (
    "Active Task",
    "Owner Preferences",
    "Corrections",
    "Environment",
    "Active Projects",
    "Key Decisions",
    "Workflow Conventions",
    "General",
)


def _content_hash(text: str) -> str:
    """Deterministic hash for deduplication."""
    normalized = " ".join(text.lower().strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _parse_persistence_md(content: str) -> Dict[str, List[str]]:
    """Parse persistence.md into {section: [entries]}.

    v5.5: placeholder bullets are dropped on read so they cannot round-trip
    through a load/write cycle and never reach the session briefing.
    """
    sections: Dict[str, List[str]] = {}
    current = "General"
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("## "):
            current = stripped[3:].strip()
            sections.setdefault(current, [])
        elif stripped.startswith("- ") and current:
            entry = stripped[2:].strip()
            if entry and not _is_placeholder(entry):
                sections[current].append(entry)
    return sections


def _build_persistence_md(sections: Dict[str, List[str]]) -> str:
    """Build persistence.md content from sections dict.

    v5.5: sentence-aware truncation instead of a hard 197-char chop, and
    placeholder bullets are never rendered back out.
    """
    lines = []
    for section in SECTION_HEADERS:
        entries = sections.get(section, [])
        entries = [e for e in entries if e and not _is_placeholder(e)]
        if entries:
            lines.append(f"## {section}")
            for entry in entries:
                lines.append(f"- {_truncate_entry(entry)}")
            lines.append("")
    # Add any sections not in the predefined order
    for section, entries in sections.items():
        if section not in SECTION_HEADERS and entries:
            entries = [e for e in entries if e and not _is_placeholder(e)]
            if not entries:
                continue
            lines.append(f"## {section}")
            for entry in entries:
                lines.append(f"- {_truncate_entry(entry)}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class PersistentMemoryManager:
    """Manages persistence.md — the between-session working memory file.

    This file is LEAN (max 1500 chars) and contains only high-value,
    cross-session-critical information: owner preferences, corrections,
    environment facts, active projects, key decisions.

    It is NOT injected wholesale into every prompt. Only relevant sections
    are pulled via get_relevant() based on the current query.
    """

    def __init__(self, vera_core_dir: Path, max_chars: int = MAX_PERSISTENCE_CHARS):
        self.persistence_file = vera_core_dir / "persistence.md"
        self.max_chars = max_chars
        self._entries: List[Dict[str, str]] = []  # {content, category, hash, added_at}
        self._sections: Dict[str, List[str]] = {}

    def load(self) -> None:
        """Load existing persistence.md."""
        if self.persistence_file.exists():
            raw = self.persistence_file.read_text(encoding="utf-8")
            self._sections = _parse_persistence_md(raw)
            # Rebuild flat entries from sections
            self._entries = []
            for section, items in self._sections.items():
                for item in items:
                    self._entries.append({
                        "content": item,
                        "category": self._section_to_category(section),
                        "hash": _content_hash(item),
                        "added_at": datetime.now(timezone.utc).isoformat(),
                    })
            logger.debug("VERA v4.5: Loaded %d persistence entries", len(self._entries))

    def should_persist(self, content: str, category: str, importance: float = 0.5) -> bool:
        """Decide if a memory should go into persistence.md.

        Returns True for high-value, cross-session-critical information.
        Returns False for ephemeral, session-scoped content.
        """
        if not content or len(content.strip()) < 10:
            return False

        # HIGH IMPORTANCE: Always persist
        if importance >= 0.8:
            return True

        # EXPLICIT CATEGORIES: Always persist
        if category in PERSIST_CATEGORIES:
            return True

        # EXPLICIT EPHEMERAL: Never persist
        if category in EPHEMERAL_CATEGORIES:
            return False

        # CONTENT HEURISTICS
        content_lower = content.lower()

        persist_score = sum(1 for s in PERSIST_SIGNALS if s in content_lower)
        ephemeral_score = sum(1 for s in EPHEMERAL_SIGNALS if s in content_lower)

        if persist_score > 0 and persist_score >= ephemeral_score:
            return True
        if ephemeral_score > persist_score:
            return False

        # Default: don't keep persistence lean
        return False

    def add_entry(self, memory: Dict[str, Any]) -> bool:
        """Add a memory entry if it should persist. Returns True if added.

        v5.5: the caller arrives already Jev-classified (see classify.py), so
        this method's job is admission control — not re-guessing the category.
        Placeholder bullets and too-short content are rejected outright.
        """
        content = classify_clean(memory.get("content", ""))
        category = memory.get("category", "general")
        importance = memory.get("importance", 0.5)

        if not content or _is_placeholder(content):
            return False

        if not self.should_persist(content, category, importance):
            return False

        content_h = _content_hash(content)

        # Deduplicate: check if same content already exists
        for existing in self._entries:
            if existing["hash"] == content_h:
                return False

        # Check if this supersedes an existing entry (correction pattern)
        if category == "corrections":
            self._entries = [
                e for e in self._entries
                if not self._is_correction_of(content, e["content"])
            ]

        self._entries.append({
            "content": content,
            "category": category,
            "hash": content_h,
            "added_at": datetime.now(timezone.utc).isoformat(),
        })

        # Rebuild sections
        self._sections = self._build_sections()
        self._enforce_limit()
        return True

    def get_relevant(self, query: str, max_chars: int = 400) -> str:
        """Get entries relevant to the current query.

        This is the key optimization: we don't dump the whole persistence.md.
        We return only entries whose content overlaps with the query.
        """
        if not self._entries or not query:
            return ""

        query_words = set(query.lower().split())
        # Remove very short words that match everything
        query_words = {w for w in query_words if len(w) > 2}

        if not query_words:
            # No meaningful query words — return first few entries as fallback
            result = []
            total = 0
            for entry in self._entries[:3]:
                text = f"- {entry['content']}"
                if total + len(text) <= max_chars:
                    result.append(text)
                    total += len(text)
            return "\n".join(result)

        # Score entries by word overlap with query
        scored: List[tuple] = []
        for entry in self._entries:
            entry_words = set(entry["content"].lower().split())
            overlap = len(query_words & entry_words)
            if overlap > 0:
                scored.append((overlap, entry))

        # Sort by relevance
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build result within budget
        result = []
        total = 0
        for score, entry in scored:
            text = f"- {entry['content']}"
            if total + len(text) <= max_chars:
                result.append(text)
                total += len(text)

        return "\n".join(result)

    def final_curation(self) -> None:
        """End-of-session cleanup: remove stale entries (7+ days old)."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        before = len(self._entries)
        self._entries = [
            e for e in self._entries
            if self._parse_timestamp(e.get("added_at", "")) > cutoff
        ]
        self._sections = self._build_sections()
        self._enforce_limit()
        after = len(self._entries)
        if before != after:
            logger.info("VERA v4.5: Persistence curation removed %d stale entries", before - after)

    def write(self) -> None:
        """Write current content to persistence.md."""
        self.persistence_file.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        header = f"# Working Memory — Auto-curated\n\nLast updated: {timestamp}\n\n"
        body = _build_persistence_md(self._sections)
        self.persistence_file.write_text(header + body, encoding="utf-8")

    def get_section(self, section_name: str) -> List[str]:
        """Get entries from a specific section."""
        return self._sections.get(section_name, [])

    # --- Internal helpers ---

    def _build_sections(self) -> Dict[str, List[str]]:
        """Rebuild sections from flat entries."""
        sections: Dict[str, List[str]] = {}
        for entry in self._entries:
            section = self._category_to_section(entry.get("category", "general"))
            sections.setdefault(section, [])
            # Deduplicate within section
            if entry["content"] not in sections[section]:
                sections[section].append(entry["content"])
        return sections

    def _enforce_limit(self) -> None:
        """Keep total content under max_chars.

        v5.5 fix: this evicted the NEWEST entry from each section (`list.pop()`
        removes the last element) while the docstring claimed it removed the
        oldest — so recent, still-relevant entries were dropped first and stale
        ones accumulated. It now evicts the oldest first, which is what
        between-session working memory actually needs.
        """
        content = _build_persistence_md(self._sections)
        while len(content) > self.max_chars and self._entries:
            removed = False
            for section in reversed(SECTION_HEADERS):
                entries_in_section = self._sections.get(section, [])
                if entries_in_section:
                    removed_entry = entries_in_section.pop(0)  # oldest first
                    self._entries = [
                        e for e in self._entries
                        if e["content"] != removed_entry
                    ]
                    removed = True
                    break
            if not removed:
                break
            content = _build_persistence_md(self._sections)

    def _is_correction_of(self, new: str, existing: str) -> bool:
        """Check if new memory corrects an existing one."""
        new_words = set(new.lower().split())
        existing_words = set(existing.lower().split())
        if not existing_words:
            return False
        overlap = len(new_words & existing_words)
        return overlap > len(existing_words) * 0.5

    def _category_to_section(self, category: str) -> str:
        """Map memory category to persistence section name."""
        mapping = {
            "owner_preferences": "Owner Preferences",
            "corrections": "Corrections",
            "preferences": "Owner Preferences",
            "environment": "Environment",
            "active_projects": "Active Projects",
            "tasks": "Active Projects",
            "key_decisions": "Key Decisions",
            "decisions": "Key Decisions",
            "workflow_conventions": "Workflow Conventions",
            "current_task": "Active Task",
        }
        return mapping.get(category, "General")

    def _section_to_category(self, section: str) -> str:
        """Reverse map section name to category."""
        mapping = {
            "Owner Preferences": "owner_preferences",
            "Corrections": "corrections",
            "Environment": "environment",
            "Active Projects": "active_projects",
            "Key Decisions": "key_decisions",
            "Workflow Conventions": "workflow_conventions",
            "Active Task": "current_task",
        }
        return mapping.get(section, "general")

    @staticmethod
    def _parse_timestamp(ts: str) -> datetime:
        """Parse ISO timestamp, return epoch on failure."""
        try:
            return datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return datetime(2000, 1, 1, tzinfo=timezone.utc)


class WorkingMemoryBuffer:
    """In-session working memory buffer.

    Holds the last N memory operations within the current session.
    Injected via prefetch() on every turn for within-session continuity.

    Solves the "amnesia within a session" problem:
    - Agent stores something via vera_store
    - 3 turns later it's buried in 577+ SQLite entries
    - The buffer keeps it hot and injectable
    """

    def __init__(self, max_entries: int = 10):
        self.max_entries = max_entries
        self.entries: List[Dict[str, str]] = []

    def add(self, content: str, category: str = "general", source: str = "session") -> None:
        """Add a recent memory operation to the buffer."""
        if not content or not content.strip():
            return

        entry = {
            "content": content.strip(),
            "category": category,
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Deduplicate: don't add if last entry is identical
        if self.entries and self.entries[0]["content"] == content.strip():
            return

        self.entries.insert(0, entry)
        # Keep only recent N
        self.entries = self.entries[:self.max_entries]

    def remove(self, content: str) -> bool:
        """v5.5: drop a just-withdrawn fact from this session's working memory.

        Without this, `vera_forget` stopped retrieval in SQLite while the
        in-session buffer kept echoing the retracted fact until the session
        ended — the owner would be told a memory was withdrawn and then see
        it again in the very next prefetch.
        """
        target = " ".join((content or "").split()).casefold()
        if not target:
            return False
        before = len(self.entries)
        self.entries = [
            e for e in self.entries
            if " ".join(str(e.get("content", "")).split()).casefold() != target
        ]
        return len(self.entries) != before

    def get_relevant(self, query: str, max_chars: int = 300) -> str:
        """Get entries relevant to the current query.

        Returns formatted entries sorted by relevance to the query,
        within the character budget.
        """
        if not self.entries or not query:
            return ""

        query_words = set(query.lower().split())
        query_words = {w for w in query_words if len(w) > 2}

        if not query_words:
            # No meaningful query — return the most recent entry
            entry = self.entries[0]
            text = f"[recent] {entry['content']}"
            return text if len(text) <= max_chars else text[:max_chars - 3] + "..."

        # Score by word overlap
        scored: List[tuple] = []
        for entry in self.entries:
            entry_words = set(entry["content"].lower().split())
            overlap = len(query_words & entry_words)
            scored.append((overlap, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Build result within budget
        result = []
        total = 0
        for score, entry in scored:
            text = f"[{entry['source']}] {entry['content']}"
            if total + len(text) <= max_chars:
                result.append(text)
                total += len(text)

        return "\n".join(result)

    def get_recent(self, n: int = 3) -> List[Dict[str, str]]:
        """Get the N most recent entries (unfiltered)."""
        return self.entries[:n]

    def clear(self) -> None:
        """Clear buffer on session end."""
        count = len(self.entries)
        self.entries.clear()
        if count:
            logger.debug("VERA v4.5: Working buffer cleared (%d entries)", count)

    def __len__(self) -> int:
        return len(self.entries)
