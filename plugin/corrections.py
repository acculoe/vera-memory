"""VERA v5 — Correction resolution for the explicit vera_store path.

The extractor path (store_extracted_memory) already handles supersession for
auto-extracted facts. The explicit tool path did not: handle_tool_call
("vera_store") called store_memory() directly, so an owner correction stored
explicitly never demoted the stale fact it corrected. This module closes that
gap with ONE function used by handle_tool_call.

Supersede semantics reuse MemoryDB._supersede_locked via store_memory's
supersedes_id parameter — no new schema, no destructive deletes.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

if __package__:
    from .models import TRUST_WEIGHTS
else:
    from models import TRUST_WEIGHTS  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY = 0.55  # Jaccard token overlap; corrections paraphrase


def _tokens(text: str) -> set:
    return set(re.findall(r"[^\W_]+", text.lower(), re.UNICODE))


def _similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    # Overlap coefficient: |A∩B| / min(|A|,|B|). Corrections paraphrase and
    # ADD words ("not X, actually Y"), which punishes Jaccard; the overlap
    # coefficient measures how much of the smaller claim is contained in the
    # newer one — exactly the correction shape.
    return len(ta & tb) / min(len(ta), len(tb))


def find_stale_predecessor(
    db,
    content: str,
    *,
    category: str,
    subject_principal_id: Optional[int],
    threshold: float = DEFAULT_SIMILARITY,
) -> Optional[Dict[str, Any]]:
    """Most-similar ACTIVE memory this content corrects/updates, or None."""
    try:
        rows = db._get_conn().execute(
            """SELECT id, content, trust_tier FROM memories
                WHERE status='active' AND category=?
                  AND COALESCE(subject_principal_id,-1)=COALESCE(?,-1)
                ORDER BY id DESC LIMIT 200""",
            (category, subject_principal_id),
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.debug("VERA v5 corrections: predecessor scan failed: %s", exc)
        return None
    best = None
    best_sim = 0.0
    for row in rows:
        sim = _similarity(content, row["content"])
        if sim >= threshold and sim > best_sim:
            best = dict(row)
            best_sim = sim
    return best


def store_with_resolution(
    db,
    *,
    content: str,
    category: str,
    trust_tier: str,
    source_type: str,
    subject_principal_id: Optional[int],
    sender_principal_id: Optional[int],
    source_session: str = "",
    importance: float = 0.5,
    similarity_threshold: float = DEFAULT_SIMILARITY,
    extractor_version: str = "authenticated-tool-v1",
) -> Dict[str, Any]:
    """Store an explicit memory; supersede a stale near-match when allowed.

    Supersede is permitted when the incoming memory is correction-typed OR its
    trust tier outranks/equal-with-higher-importance the old one — mirroring
    store_extracted_memory's precedence rules.
    """
    predecessor = find_stale_predecessor(
        db, content, category=category,
        subject_principal_id=subject_principal_id,
        threshold=similarity_threshold,
    )

    kwargs: Dict[str, Any] = dict(
        content=content,
        category=category,
        source_session=source_session,
        source="conversation",
        importance=importance,
        subject_principal_id=subject_principal_id,
        sender_principal_id=sender_principal_id,
        trust_tier=trust_tier,
        source_type=source_type,
        extractor_version=extractor_version,
    )

    action = "stored"
    if predecessor:
        old_weight = TRUST_WEIGHTS.get(predecessor.get("trust_tier", ""), 0.0)
        new_weight = TRUST_WEIGHTS.get(trust_tier, 0.0)
        is_correction = bool(re.search(
            r"\b(?:correction|actually|not\s+\w+|instead\s+of)\b", content.lower()))
        # A same-trust owner correction with >= importance demotes the old row.
        if (is_correction and new_weight >= old_weight) or new_weight > old_weight:
            kwargs["supersedes_id"] = int(predecessor["id"])
            action = "superseded"

    memory_id = db.store_memory(**kwargs)
    result = {"memory_id": int(memory_id), "action": action}
    if action == "superseded":
        result["superseded_id"] = int(predecessor["id"])  # type: ignore[index]
        logger.info(
            "VERA v5: correction superseded stale memory %s -> %s",
            predecessor["id"], memory_id,  # type: ignore[index]
        )
    return result
