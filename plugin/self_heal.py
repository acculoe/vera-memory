"""VERA v5 — Embedding self-heal.

Closes the v4 defect where embeddings only covered post-init memories and
failures were swallowed silently. Two pure operations:

1. count_embedding_gap(db, model_hash)
   -> (total_active, missing_ids) using the SAME join semantics as
   SQLiteVectorStore.load(): a vector only counts when its content_hash
   still matches the memory's current content_hash.

2. heal_embeddings(db, adapter, ...)
   -> embeds exactly the missing ids through the existing
   SQLiteVectorStore.upsert path, one retry per failed batch, and reports
   every residual failure LOUDLY (return value + logger.warning).
   No silent swallowing, ever.
"""

from __future__ import annotations

import logging
import time

if __package__:
    from .semantic import SQLiteVectorStore
else:
    from semantic import SQLiteVectorStore  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32
DEFAULT_TIMEOUT_SECONDS = 60.0


def count_embedding_gap(db, model_hash: str):
    """Return (total_active, [missing_ids]) for the pinned model hash.

    Mirrors store.load() validity rules: embedding must exist for this
    model_hash AND its content_hash must equal the memory's current hash.
    Stale vectors therefore count as missing and get rebuilt.
    """
    conn = db._get_conn()
    total = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE status='active'"
    ).fetchone()[0]
    rows = conn.execute(
        """SELECT m.id
             FROM memories m
            WHERE m.status='active'
              AND NOT EXISTS (
                    SELECT 1 FROM memory_embeddings e
                     WHERE e.memory_id = m.id
                       AND e.model_hash = ?
                       AND e.content_hash = m.content_hash
              )
            ORDER BY m.id""",
        (model_hash,),
    ).fetchall()
    return int(total), [int(r[0]) for r in rows]


def heal_embeddings(
    db,
    adapter,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_failures_logged: int = 50,
) -> dict:
    """Embed every missing active memory. Returns an honest report.

    - Uses the existing SQLiteVectorStore.upsert path (no bypass).
    - One retry per failed batch before recording failures.
    - Failures are returned in report["failed"] AND logged at WARNING.
    - Raises TimeoutError if the overall budget is exhausted mid-run;
      progress already committed stays committed (safe re-run later).
    """
    spec = adapter.spec
    store = SQLiteVectorStore(db, spec)
    started = time.perf_counter()
    total, missing_ids = count_embedding_gap(db, spec.model_hash)

    conn = db._get_conn()
    id_to_content = {
        int(r["id"]): r["content"]
        for r in conn.execute(
            f"SELECT id, content FROM memories WHERE status='active'"
            f" AND id IN ({','.join('?' for _ in missing_ids) or 'NULL'})",
            missing_ids,
        ).fetchall()
    }

    embedded = 0
    failed = []
    batches = [
        missing_ids[i : i + max(1, int(batch_size))]
        for i in range(0, len(missing_ids), max(1, int(batch_size)))
    ]

    for batch_ids in batches:
        remaining = timeout_seconds - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError(
                f"embedding self-heal exceeded {timeout_seconds}s budget "
                f"({embedded}/{len(missing_ids)} done)"
            )
        texts = [id_to_content[mid] for mid in batch_ids]
        vectors = None
        for attempt in (1, 2):  # exactly one retry
            try:
                vectors = adapter.encode(texts, timeout_seconds=remaining)
                break
            except Exception as exc:  # noqa: BLE001 — recorded loudly below
                if attempt == 2:
                    for mid in batch_ids:
                        failed.append({"id": mid, "reason": str(exc)[:200]})
                    logger.warning(
                        "VERA v5 self-heal: batch FAILED after retry "
                        "(ids=%s): %s", batch_ids[:8], str(exc)[:160]
                    )
                else:
                    logger.warning(
                        "VERA v5 self-heal: batch attempt 1 failed, retrying: %s",
                        str(exc)[:160],
                    )
        if vectors is None:
            continue
        try:
            written = store.upsert(
                (mid, _content_hash_of(conn, mid), vec)
                for mid, vec in zip(batch_ids, vectors)
            )
            embedded += written
        except Exception as exc:  # noqa: BLE001
            for mid in batch_ids:
                failed.append({"id": mid, "reason": f"upsert: {exc}"[:200]})
            logger.warning("VERA v5 self-heal: upsert FAILED (ids=%s): %s",
                           batch_ids[:8], str(exc)[:160])

    report = {
        "total": total,
        "missing_before": len(missing_ids),
        "embedded": embedded,
        "failed": failed[:max(0, int(max_failures_logged))],
        "failed_count": len(failed),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "model_hash": spec.model_hash,
    }
    logger.info(
        "VERA v5: embedding self-heal complete (embedded=%d, failed=%d, "
        "elapsed=%.1fs)", embedded, len(failed), report["elapsed_s"],
    )
    return report


def _content_hash_of(conn, memory_id: int) -> str:
    row = conn.execute(
        "SELECT content_hash FROM memories WHERE id=?", (memory_id,)
    ).fetchone()
    if not row or not row["content_hash"]:
        raise ValueError(f"memory {memory_id} has no content_hash")
    return row["content_hash"]
