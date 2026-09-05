"""VERA Memory — Migration from VERA .md files to SQLite.

Non-destructive: copies shared files instead of moving them.
Per-profile archives with collision safety.
"""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__:
    from .models import MemoryDB
else:  # Direct source-tree import used by the isolated regression harness.
    from models import MemoryDB

logger = logging.getLogger(__name__)


def migrate_vera_files(
    db: MemoryDB,
    profile_dir: Path,
    shared_vera_dir: Optional[Path] = None,
    archive: bool = True,
    subject_principal_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Migrate VERA .md memory files to SQLite.

    Non-destructive: shared files are COPIED, not moved.
    Per-profile archives avoid collision between agents. When a verified
    principal is supplied, migrated rows are subject-bound and immediately
    eligible for authenticated retrieval.
    """
    report = {
        "files_found": 0,
        "entries_extracted": 0,
        "entries_stored": 0,
        "entries_skipped": 0,
        "files_archived": 0,
        "errors": [],
    }

    all_entries = []
    pending_profile_archives: List[Path] = []
    pending_shared_archives: List[Path] = []

    # 1. Per-profile VERA memory (move to archive after migration)
    profile_vera = profile_dir / "vera" / "memory"
    if profile_vera.exists():
        md_files = sorted(profile_vera.glob("*.md"))
        report["files_found"] += len(md_files)
        for md_file in md_files:
            try:
                entries = _parse_vera_file(md_file)
                all_entries.extend(entries)
                report["entries_extracted"] += len(entries)
                if archive:
                    pending_profile_archives.append(md_file)
            except Exception as e:
                report["errors"].append(f"Error parsing {md_file.name}: {e}")

    # 2. Shared VERA memory (COPY only, never move)
    if shared_vera_dir and shared_vera_dir.exists():
        shared_memory = shared_vera_dir / "memory"
        if shared_memory.exists():
            md_files = sorted(shared_memory.glob("*.md"))
            report["files_found"] += len(md_files)
            for md_file in md_files:
                try:
                    entries = _parse_vera_file(md_file)
                    all_entries.extend(entries)
                    report["entries_extracted"] += len(entries)
                    if archive:
                        pending_shared_archives.append(md_file)
                except Exception as e:
                    report["errors"].append(f"Error parsing shared {md_file.name}: {e}")

    # Store in one validated transaction, including the zero-entry case. This
    # keeps supplied-principal validation strict before any archive operation.
    if subject_principal_id is not None:
        for entry in all_entries:
            entry["subject_principal_id"] = int(subject_principal_id)
            entry["retrieval_enabled"] = True
    stored = db.bulk_store(
        all_entries,
        atomic=True,
        required_active_principal_id=subject_principal_id,
    )
    report["entries_stored"] = stored
    report["entries_skipped"] = len(all_entries) - stored

    # Archive only after the validated batch transaction has committed. Empty
    # files retain the legacy archive/copy behavior without bypassing principal
    # validation. If archive I/O fails, the source remains available and a
    # rerun deduplicates any rows already stored.
    if archive:
        for md_file in pending_profile_archives:
            try:
                _archive_file(md_file, profile_dir / "vera" / "archive", report)
            except Exception as exc:
                report["errors"].append(f"Error archiving {md_file.name}: {exc}")
        for md_file in pending_shared_archives:
            try:
                dest = profile_dir / "vera" / "archive" / md_file.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(str(md_file), str(dest))
                    report["files_archived"] += 1
            except Exception as exc:
                report["errors"].append(f"Error archiving shared {md_file.name}: {exc}")

    return report


def _archive_file(filepath: Path, archive_dir: Path, report: Dict) -> None:
    """Move a file to archive with collision safety."""
    dest = archive_dir / filepath.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        # Collision: append timestamp
        stem = filepath.stem
        suffix = filepath.suffix
        ts = datetime.now().strftime("%H%M%S")
        dest = archive_dir / f"{stem}-{ts}{suffix}"
    shutil.move(str(filepath), str(dest))
    report["files_archived"] += 1


def _parse_vera_file(filepath: Path) -> List[Dict[str, Any]]:
    """Parse a VERA .md memory file into memory entries."""
    content = filepath.read_text(encoding="utf-8")
    entries = []
    date_match = re.match(r'(\d{4}-\d{2}-\d{2})', filepath.stem)
    file_date = date_match.group(1) if date_match else None
    sections = re.split(r'\n##\s+', content)
    for section in sections:
        section = section.strip()
        if not section or len(section) < 20:
            continue
        timestamp_match = re.search(r'\[(\d{2}:\d{2}:\d{2})\]', section)
        timestamp = timestamp_match.group(1) if timestamp_match else None
        category = _categorize_section(section)
        lines = section.split('\n')
        meaningful_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if any(skip in line.lower() for skip in [
                'vera mirror', 'session', 'scope', 'completed',
                'notes', 'external side effects', 'auto-created'
            ]):
                continue
            meaningful_lines.append(line)
        if meaningful_lines:
            entry_content = ' '.join(meaningful_lines)
            if len(entry_content) > 2000:
                entry_content = entry_content[:1997] + "..."
            created_at = file_date or datetime.now().strftime("%Y-%m-%d")
            if timestamp:
                created_at = f"{created_at}T{timestamp}"
            entries.append({
                "content": entry_content,
                "category": category,
                "source": "vera-migration",
                "source_session": filepath.stem,
                "created_at": created_at,
                "importance": _estimate_importance(entry_content),
                "metadata": {"original_file": filepath.name},
            })
    return entries


def _categorize_section(section_text: str) -> str:
    text_lower = section_text.lower()
    if any(kw in text_lower for kw in ["decided", "agreed", "confirmed", "plan"]):
        return "decisions"
    if any(kw in text_lower for kw in ["domain", "email", "url", "config"]):
        return "facts"
    if any(kw in text_lower for kw in ["prefer", "like", "want", "always", "never"]):
        return "preferences"
    if any(kw in text_lower for kw in ["wrong", "incorrect", "actually", "fix"]):
        return "corrections"
    if any(kw in text_lower for kw in ["research", "analysis", "finding"]):
        return "research"
    if any(kw in text_lower for kw in ["task", "completed", "done", "shipped"]):
        return "tasks"
    return "general"


def _estimate_importance(text: str) -> float:
    score = 0.4
    text_lower = text.lower()
    if any(w in text_lower for w in ["important", "critical", "never forget"]):
        score += 0.3
    if any(w in text_lower for w in ["decided", "agreed", "confirmed"]):
        score += 0.2
    if len(text) > 200:
        score += 0.1
    return min(score, 1.0)
