"""VERA v5 — Session-start briefing + genuine persistence.md integration.

Fixes two v4.5 defects:
1. Nothing told the agent where it left off. initialize() now builds a
   <=150-token briefing: last session topic + top Active Task from
   persistence.md, injected once at session start.
2. persistence.md's "Active Task" section was never populated by anyone.
   The proactive-recall intent layer now owns that section: when a
   work-start signal fires, the Active Task is set/replaced automatically.

The briefing is built ONCE per session (initialize) and cached; it must NOT
be rebuilt per message (prompt-cache safety).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

BRIEFING_MAX_CHARS = 600  # ~150 tokens

ACTIVE_TASK_HEADER = "## Active Task"


def build_session_briefing(
    db,
    persistence_lines: Iterable[str],
    *,
    max_chars: int = BRIEFING_MAX_CHARS,
) -> str:
    """One-shot session-start block.

    Components (only when available):
    - [briefing] header
    - Last session topic (most recent ended session with a summary)
    - Current Active Task line from persistence.md
    """
    parts: List[str] = []

    last_topic = _last_session_topic(db)
    if last_topic:
        parts.append(f"Last session: {last_topic}")

    for line in persistence_lines:
        line = (line or "").strip()
        if line and len(line) < 300:
            parts.append(f"Active task: {line}")
            break  # only the top active-task line

    if not parts:
        return ""

    body = "\n".join(parts)
    if len(body) > max_chars:
        body = body[: max_chars - 3].rstrip() + "..."
    return f"[briefing] {body}"


def _last_session_topic(db) -> Optional[str]:
    """Most recent ENDED session with a summary, cleaned for injection."""
    try:
        rows = db._get_conn().execute(
            """SELECT summary FROM sessions
                WHERE ended_at IS NOT NULL AND summary IS NOT NULL
                ORDER BY id DESC LIMIT 1"""
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.debug("VERA v5 briefing: session lookup failed: %s", exc)
        return None
    if not rows or not rows[0] or not rows[0]["summary"]:
        return None
    raw = str(rows[0]["summary"])
    # v4 stored summaries like "Session ended. Last topic: <text>" and cron
    # sessions stored whole system-prompt blobs. Extract the human part.
    for marker in ("Last topic:", "last topic:"):
        idx = raw.find(marker)
        if idx >= 0:
            raw = raw[idx + len(marker):]
            break
    raw = " ".join(raw.split())
    # Drop obvious prompt-injection-looking prefixes from cron artifacts.
    if raw.startswith("[") and "]" in raw[:120]:
        inner_end = raw.find("]") + 1
        raw = raw[inner_end:].strip() or raw
    return raw[:200] or None


class BriefingPersistence:
    """Thin, honest read/write layer over the persistence.md file.

    Unlike the v4.5 word-overlap gate (which left Active Task permanently
    empty), this class gives the recall layer a direct, tested API to own
    exactly one section: Active Task.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def get_active_task(self) -> str:
        if not self.path.exists():
            return ""
        current_section = ""
        best = ""
        for line in self.path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                current_section = stripped[3:].strip()
                continue
            if current_section == "Active Task" and stripped.startswith("- "):
                best = stripped[2:].strip()
                break
        return best

    def set_active_task(self, task: str) -> bool:
        """Replace the single Active Task bullet. Returns True if file changed."""
        task = " ".join((task or "").split())[:280]
        if not task:
            return False
        text = self.path.read_text(encoding="utf-8") if self.path.exists() else (
            "# Working Memory — Auto-curated\n"
        )
        lines = text.splitlines()
        out: List[str] = []
        in_target = False
        replaced = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## "):
                if in_target and not replaced:
                    out.append(f"- {task}")
                    replaced = True
                in_target = stripped[3:].strip() == "Active Task"
                out.append(line)
                continue
            if in_target and (stripped.startswith("- ") or not stripped):
                continue  # drop old bullets/blank lines inside Active Task
            out.append(line)
        if in_target and not replaced:
            out.append(f"- {task}")
            replaced = True
        if not any(s.strip() == ACTIVE_TASK_HEADER for s in out):
            if out and out[-1].strip():
                out.append("")
            out.append(ACTIVE_TASK_HEADER)
            out.append(f"- {task}")
            replaced = True
        new_text = "\n".join(out).rstrip() + "\n"
        if new_text == text:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(new_text, encoding="utf-8")
        logger.info("VERA v5: persistence.md Active Task updated: %s", task[:80])
        return True


def set_active_task(mgr: BriefingPersistence, task: str) -> None:
    mgr.set_active_task(task)


def get_active_task(mgr: BriefingPersistence) -> str:
    return mgr.get_active_task()
