"""VERA Memory — governed SQLite schema and CRUD operations.

SQLite remains the canonical record.  Search indexes and embeddings are
rebuildable derivatives.  Schema upgrades are ordered, recorded, and designed
to preserve existing rows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 8
VALID_LAYERS = {"l0", "l1", "l2", "l3"}
VALID_STATUSES = {"active", "historical", "superseded", "quarantined", "deleted"}
VALID_TRUST_TIERS = {
    "canonical",
    "owner_declared",
    "owner_approved",
    "verified_summary",
    "assistant_summary",
    "observed_chat",
    "imported_history",
}
VALID_VISIBILITIES = {"subject", "profile_global"}
PROFILE_GLOBAL_TRUST_TIERS = {"canonical", "owner_declared", "owner_approved", "verified_summary"}
# Only local configuration and validated summaries may be globally visible.  A
# caller-supplied source label or trust tier is never enough to widen scope.
PROFILE_GLOBAL_SOURCES = {"local-policy", "runtime-config", "generated-summary"}
# Identity-only capability: it is intentionally never constructed from tool,
# extraction, or transport inputs. It makes scope widening an internal call path,
# not a caller-controlled source/trust label.
_PROFILE_GLOBAL_CAPABILITY = object()
_OWNER_APPROVAL_CAPABILITY = object()
# Offline-only capability for an owner-approved legacy corpus repair. Runtime
# tool arguments cannot manufacture this object, so an authenticated chat turn
# cannot claim unbound historical rows merely by naming a principal.
_LEGACY_BINDING_CAPABILITY = object()

TRUST_WEIGHTS = {
    "canonical": 1.00,
    "owner_declared": 0.95,
    "owner_approved": 0.90,
    "verified_summary": 0.80,
    "assistant_summary": 0.55,
    "observed_chat": 0.35,
    "imported_history": 0.25,
}

BASE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    source_session TEXT,
    source TEXT NOT NULL DEFAULT 'conversation',
    importance REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    metadata TEXT,
    memory_layer TEXT NOT NULL DEFAULT 'l1',
    memory_type TEXT NOT NULL DEFAULT 'fact',
    subject_principal_id INTEGER,
    trust_tier TEXT NOT NULL DEFAULT 'observed_chat',
    status TEXT NOT NULL DEFAULT 'active',
    retrieval_enabled INTEGER NOT NULL DEFAULT 1,
    valid_from TEXT,
    valid_to TEXT,
    supersedes_id INTEGER,
    normalized_content TEXT,
    content_hash TEXT,
    embedding_status TEXT NOT NULL DEFAULT 'missing',
    embedding_model TEXT,
    visibility TEXT NOT NULL DEFAULT 'subject'
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    summary TEXT,
    turn_count INTEGER NOT NULL DEFAULT 0,
    memories_created INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    checksum TEXT NOT NULL
);
"""

GOVERNANCE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS principals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    platform TEXT NOT NULL,
    platform_subject_id TEXT NOT NULL,
    principal_type TEXT NOT NULL DEFAULT 'person',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(platform, platform_subject_id)
);

CREATE TABLE IF NOT EXISTS canonical_identity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    principal_id INTEGER NOT NULL,
    context_scope TEXT NOT NULL DEFAULT 'default',
    reference_name TEXT,
    direct_address TEXT,
    authority TEXT NOT NULL DEFAULT 'canonical',
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    supersedes_id INTEGER,
    status TEXT NOT NULL DEFAULT 'active',
    FOREIGN KEY(principal_id) REFERENCES principals(id),
    FOREIGN KEY(supersedes_id) REFERENCES canonical_identity(id)
);
CREATE UNIQUE INDEX IF NOT EXISTS canonical_identity_active_scope
ON canonical_identity(principal_id, context_scope)
WHERE status='active';

CREATE TABLE IF NOT EXISTS memory_provenance (
    memory_id INTEGER PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_locator TEXT,
    source_session TEXT,
    sender_principal_id INTEGER,
    observed_at TEXT NOT NULL,
    extractor_version TEXT,
    content_hash TEXT NOT NULL,
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY(sender_principal_id) REFERENCES principals(id)
);

CREATE TABLE IF NOT EXISTS memory_links (
    parent_id INTEGER NOT NULL,
    child_id INTEGER NOT NULL,
    relation TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(parent_id, child_id, relation),
    FOREIGN KEY(parent_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY(child_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector BLOB NOT NULL,
    vector_checksum TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(memory_id, model_id),
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS migration_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    migration_name TEXT NOT NULL,
    memory_id INTEGER,
    before_json TEXT,
    after_json TEXT,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conflict_review (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id_a INTEGER NOT NULL,
    memory_id_b INTEGER NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    FOREIGN KEY(memory_id_a) REFERENCES memories(id),
    FOREIGN KEY(memory_id_b) REFERENCES memories(id)
);
"""

FTS_SCHEMA_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS fts_memories USING fts5(
    content,
    category,
    source,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories
WHEN new.retrieval_enabled=1 AND new.status='active'
BEGIN
    INSERT INTO fts_memories(rowid, content, category, source)
    VALUES (new.id, new.content, new.category, new.source);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories
BEGIN
    DELETE FROM fts_memories WHERE rowid=old.id;
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories
BEGIN
    DELETE FROM fts_memories WHERE rowid=old.id;
    INSERT INTO fts_memories(rowid, content, category, source)
    SELECT new.id, new.content, new.category, new.source
    WHERE new.retrieval_enabled=1 AND new.status='active';
END;
"""

EMBEDDING_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector BLOB NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(memory_id, model_hash),
    FOREIGN KEY(memory_id) REFERENCES memories(id)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_model
    ON memory_embeddings(model_hash, memory_id);
"""

LAYERED_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS summary_generations (
    summary_id INTEGER PRIMARY KEY,
    series_key TEXT NOT NULL,
    layer TEXT NOT NULL,
    generation INTEGER NOT NULL,
    generator_version TEXT NOT NULL,
    owner_review_status TEXT NOT NULL DEFAULT 'not_required',
    identity_adjacent INTEGER NOT NULL DEFAULT 0,
    supersedes_summary_id INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE(series_key, generation),
    FOREIGN KEY(summary_id) REFERENCES memories(id),
    FOREIGN KEY(supersedes_summary_id) REFERENCES memories(id)
);
CREATE INDEX IF NOT EXISTS idx_summary_generations_series
    ON summary_generations(series_key, generation);
"""

_COLUMN_MIGRATIONS = {
    "memory_layer": "TEXT NOT NULL DEFAULT 'l1'",
    "memory_type": "TEXT NOT NULL DEFAULT 'fact'",
    "subject_principal_id": "INTEGER",
    "trust_tier": "TEXT NOT NULL DEFAULT 'observed_chat'",
    "status": "TEXT NOT NULL DEFAULT 'active'",
    "retrieval_enabled": "INTEGER NOT NULL DEFAULT 1",
    "valid_from": "TEXT",
    "valid_to": "TEXT",
    "supersedes_id": "INTEGER",
    "normalized_content": "TEXT",
    "content_hash": "TEXT",
    "embedding_status": "TEXT NOT NULL DEFAULT 'missing'",
    "embedding_model": "TEXT",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_content(content: str) -> str:
    normalized = unicodedata.normalize("NFKC", content or "")
    return " ".join(normalized.split()).strip().casefold()


def content_digest(content: str) -> str:
    return hashlib.sha256(normalize_content(content).encode("utf-8")).hexdigest()


def claim_key(value: str) -> Optional[str]:
    """Return a conservative subject key for explicit `subject is value` claims."""
    text = normalize_content(value)
    text = re.sub(
        r"^(?:actually|correction|i meant|the correct answer is|the right value is)[,:\s]+",
        "",
        text,
    )
    match = re.match(
        r"^(.{2,120}?)\s+(?:is|are|was|were|should be|will be|=)\s+.{1,300}$",
        text,
    )
    if not match:
        return None
    subject = " ".join(match.group(1).split()).strip(" ,:.-")
    if len(subject.split()) > 12:
        return None
    return subject or None


def _migration_checksum(version: int, description: str) -> str:
    return hashlib.sha256(f"{version}:{description}".encode("utf-8")).hexdigest()


class MemoryDB:
    """Thread-safe, profile-local VERA database."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()

    @staticmethod
    def needs_migration_at_path(db_path: Path) -> bool:
        """Read-only preflight: never initialize or mutate a live database."""
        path = Path(db_path)
        if not path.exists():
            return False
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "schema_migrations" not in tables:
                return True
            row = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()
            return int(row[0]) < SCHEMA_VERSION
        finally:
            conn.close()

    def connect(self, *, apply_migrations: bool = True) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(BASE_SCHEMA_SQL)
        self._apply_migrations()
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    def _get_conn(self) -> sqlite3.Connection:
        if not self._conn:
            raise RuntimeError("MemoryDB not connected")
        return self._conn

    def _apply_migrations(self) -> None:
        conn = self._get_conn()
        migrations = [
            (1, "baseline tables", self._migration_baseline),
            (2, "legacy schema marker compatibility", self._migration_legacy_marker),
            (3, "memory governance columns", self._migration_governance_columns),
            (4, "principal provenance lifecycle tables", self._migration_governance_tables),
            (5, "active-only contentless fts index", self._migration_fts),
            (6, "rebuildable sqlite embedding storage", self._migration_embeddings),
            (7, "versioned layered summary generations", self._migration_layered_summaries),
            (8, "subject scoped retrieval visibility", self._migration_subject_visibility),
        ]
        applied_rows = {
            int(row["version"]): str(row["checksum"])
            for row in conn.execute("SELECT version, checksum FROM schema_migrations")
        }
        expected = {version: _migration_checksum(version, description) for version, description, _ in migrations}
        if set(applied_rows) - set(expected):
            raise RuntimeError("database records unknown schema migration")
        for version, recorded in applied_rows.items():
            if recorded != expected[version]:
                raise RuntimeError(f"database migration checksum mismatch: {version}")
        try:
            conn.execute("BEGIN")
            for version, description, func in migrations:
                if version in applied_rows:
                    continue
                func(conn)
                conn.execute(
                    "INSERT INTO schema_migrations(version, applied_at, checksum) VALUES (?, ?, ?)",
                    (version, utcnow(), _migration_checksum(version, description)),
                )
            self._backfill_governance(conn)
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @staticmethod
    def _migration_baseline(conn: sqlite3.Connection) -> None:
        conn.executescript(BASE_SCHEMA_SQL)

    @staticmethod
    def _migration_legacy_marker(conn: sqlite3.Connection) -> None:
        # Version 2 was the last legacy schema.  The row is retained so ordered
        # migration history is explicit even for fresh databases.
        conn.execute("SELECT 1")

    @staticmethod
    def _migration_governance_columns(conn: sqlite3.Connection) -> None:
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(memories)")
        }
        for name, definition in _COLUMN_MIGRATIONS.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE memories ADD COLUMN {name} {definition}")

    @staticmethod
    def _migration_governance_tables(conn: sqlite3.Connection) -> None:
        conn.executescript(GOVERNANCE_SCHEMA_SQL)

    @staticmethod
    def _migration_fts(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            DROP TRIGGER IF EXISTS memories_ai;
            DROP TRIGGER IF EXISTS memories_ad;
            DROP TRIGGER IF EXISTS memories_au;
            DROP TABLE IF EXISTS fts_memories;
            """
        )
        conn.executescript(FTS_SCHEMA_SQL)
        conn.execute(
            """INSERT INTO fts_memories(rowid, content, category, source)
               SELECT id, content, category, source FROM memories
               WHERE retrieval_enabled=1 AND status='active'"""
        )

    @staticmethod
    def _migration_embeddings(conn: sqlite3.Connection) -> None:
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(memory_embeddings)")
        }
        if columns and "model_hash" not in columns:
            conn.execute("ALTER TABLE memory_embeddings RENAME TO memory_embeddings_legacy_v5")
            conn.executescript(EMBEDDING_SCHEMA_SQL)
            rows = conn.execute(
                """SELECT e.memory_id, e.model_id, e.dimensions, e.vector,
                          e.created_at, m.content_hash
                   FROM memory_embeddings_legacy_v5 e
                   JOIN memories m ON m.id=e.memory_id"""
            ).fetchall()
            conn.executemany(
                """INSERT INTO memory_embeddings
                   (memory_id, model_name, model_hash, dimensions, vector,
                    content_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        row["memory_id"],
                        row["model_id"],
                        hashlib.sha256(str(row["model_id"]).encode("utf-8")).hexdigest(),
                        row["dimensions"],
                        row["vector"],
                        row["content_hash"],
                        row["created_at"],
                    )
                    for row in rows
                ],
            )
            conn.execute("DROP TABLE memory_embeddings_legacy_v5")
        else:
            conn.executescript(EMBEDDING_SCHEMA_SQL)

    @staticmethod
    def _migration_layered_summaries(conn: sqlite3.Connection) -> None:
        conn.executescript(LAYERED_SCHEMA_SQL)

    @staticmethod
    def _migration_subject_visibility(conn: sqlite3.Connection) -> None:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(memories)")}
        if "visibility" not in columns:
            conn.execute("ALTER TABLE memories ADD COLUMN visibility TEXT NOT NULL DEFAULT 'subject'")
        # Existing unbound rows are not known to be safe shared profile facts.
        conn.execute(
            """UPDATE memories SET visibility='profile_global'
               WHERE subject_principal_id IS NULL
                 AND trust_tier IN ('canonical', 'owner_declared', 'owner_approved')
                 AND source IN ('explicit', 'local-policy', 'runtime-config')"""
        )
        conn.execute(
            """UPDATE memories SET retrieval_enabled=0
               WHERE subject_principal_id IS NULL AND visibility!='profile_global'
                 AND source NOT IN ('explicit', 'local-policy', 'runtime-config')"""
        )

    @staticmethod
    def _backfill_governance(conn: sqlite3.Connection) -> None:
        conn.execute(
            """UPDATE memories
               SET normalized_content=lower(trim(content))
               WHERE normalized_content IS NULL OR normalized_content=''"""
        )
        rows = conn.execute(
            "SELECT id, content FROM memories WHERE content_hash IS NULL OR content_hash=''"
        ).fetchall()
        conn.executemany(
            "UPDATE memories SET normalized_content=?, content_hash=? WHERE id=?",
            [(normalize_content(r["content"]), content_digest(r["content"]), r["id"]) for r in rows],
        )
        conn.execute(
            """UPDATE memories SET trust_tier = CASE
                 WHEN source IN ('vera-migration','migration') AND trust_tier='observed_chat'
                 THEN 'imported_history'
                 WHEN trust_tier IS NULL OR trust_tier='' OR trust_tier NOT IN
                    ('canonical','owner_declared','owner_approved','verified_summary',
                     'assistant_summary','observed_chat','imported_history')
                 THEN CASE
                    WHEN source='explicit' THEN 'owner_declared'
                    WHEN source IN ('vera-migration','migration') THEN 'imported_history'
                    WHEN source='hermes-memory-tool' THEN 'assistant_summary'
                    ELSE 'observed_chat' END
                 ELSE trust_tier END"""
        )
        now = utcnow()
        missing = conn.execute(
            """SELECT id, source, source_session, content_hash, created_at
               FROM memories
               WHERE id NOT IN (SELECT memory_id FROM memory_provenance)"""
        ).fetchall()
        conn.executemany(
            """INSERT INTO memory_provenance
               (memory_id, source_type, source_locator, source_session,
                sender_principal_id, observed_at, extractor_version, content_hash)
               VALUES (?, ?, NULL, ?, NULL, ?, 'legacy-backfill-v1', ?)""",
            [
                (
                    row["id"],
                    row["source"] or "legacy",
                    row["source_session"],
                    row["created_at"] or now,
                    row["content_hash"],
                )
                for row in missing
            ],
        )

    # --- Meta and health ---

    def get_schema_version(self) -> int:
        row = self._get_conn().execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        return int(row["value"]) if row else 0

    def integrity_check(self) -> str:
        with self._lock:
            return self._get_conn().execute("PRAGMA integrity_check").fetchone()[0]

    def rebuild_fts(self, *, commit: bool = True) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM fts_memories")
            conn.execute(
                """INSERT INTO fts_memories(rowid, content, category, source)
                   SELECT id, content, category, source FROM memories
                   WHERE retrieval_enabled=1 AND status='active'"""
            )
            if commit:
                conn.commit()

    def bind_legacy_memories_to_principal(
        self,
        principal_id: int,
        *,
        memory_ids: Sequence[int],
        approval_run_id: str,
        _legacy_binding_capability: Any = None,
    ) -> Dict[str, int]:
        """Bind an explicit, owner-approved legacy row manifest to a principal.

        This is deliberately not an automatic startup migration. The caller
        must preflight the exact row IDs and supply a non-sensitive approval
        run identifier plus the module-private maintenance capability.
        """
        if _legacy_binding_capability is not _LEGACY_BINDING_CAPABILITY:
            raise ValueError("legacy binding requires owner-approved capability")
        manifest = sorted({int(memory_id) for memory_id in memory_ids})
        if not manifest:
            raise ValueError("legacy binding requires a non-empty memory manifest")
        if len(manifest) > 900:
            raise ValueError("legacy binding manifest exceeds the 900-row transaction limit")
        run_id = str(approval_run_id).strip()
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", run_id):
            raise ValueError("approval run id must be a non-sensitive identifier")

        target_principal = int(principal_id)
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                principal = conn.execute(
                    "SELECT id FROM principals WHERE id=? AND status='active'",
                    (target_principal,),
                ).fetchone()
                if principal is None:
                    raise ValueError("target principal does not exist or is inactive")

                placeholders = ",".join("?" for _ in manifest)
                rows = conn.execute(
                    f"""SELECT id, subject_principal_id, visibility,
                               retrieval_enabled, status, source, trust_tier,
                               content_hash
                        FROM memories WHERE id IN ({placeholders}) ORDER BY id""",
                    manifest,
                ).fetchall()
                if len(rows) != len(manifest):
                    raise ValueError("legacy binding manifest contains missing rows")

                to_bind = []
                already_bound = 0
                for row in rows:
                    if row["status"] != "active" or row["visibility"] != "subject":
                        raise ValueError("legacy binding manifest contains an ineligible row")
                    subject = row["subject_principal_id"]
                    if subject is None:
                        to_bind.append(row)
                    elif int(subject) == target_principal:
                        already_bound += 1
                    else:
                        raise ValueError("legacy binding manifest contains a row owned by another principal")

                now = utcnow()
                newly_enabled = sum(
                    1 for row in to_bind if int(row["retrieval_enabled"]) != 1
                )
                for row in to_bind:
                    provenance = {
                        "source": row["source"],
                        "trust_tier": row["trust_tier"],
                        "content_hash": row["content_hash"],
                        "approval_run_id": run_id,
                    }
                    before = {
                        **provenance,
                        "subject_principal_id": row["subject_principal_id"],
                        "visibility": row["visibility"],
                        "retrieval_enabled": int(row["retrieval_enabled"]),
                        "status": row["status"],
                    }
                    after = {
                        **provenance,
                        "subject_principal_id": target_principal,
                        "visibility": "subject",
                        "retrieval_enabled": 1,
                        "status": "active",
                    }
                    conn.execute(
                        """UPDATE memories
                           SET subject_principal_id=?, visibility='subject',
                               retrieval_enabled=1, updated_at=?
                           WHERE id=?""",
                        (target_principal, now, int(row["id"])),
                    )
                    conn.execute(
                        """INSERT INTO migration_audit
                           (migration_name, memory_id, before_json, after_json, applied_at)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            "owner-approved-legacy-principal-binding-v1",
                            int(row["id"]),
                            json.dumps(before, sort_keys=True),
                            json.dumps(after, sort_keys=True),
                            now,
                        ),
                    )

                self.rebuild_fts(commit=False)
                conn.commit()
                return {
                    "principal_id": target_principal,
                    "bound": len(to_bind),
                    "retrieval_enabled": newly_enabled,
                    "already_bound": already_bound,
                }
            except Exception:
                conn.rollback()
                raise

    def get_active_retrieval_rows(
        self, *, principal_id: Optional[int] = None, authenticated: bool = False
    ) -> List[Dict[str, Any]]:
        """Return only SQLite-authorized semantic candidates.

        Policy: subject rows require the exact authenticated principal.  A row
        with no subject is shareable only when explicitly profile-global and
        backed by a high-trust local owner/canonical source.
        """
        if not authenticated or principal_id is None:
            return []
        with self._lock:
            where = ["status='active'", "retrieval_enabled=1"]
            params: List[Any] = []
            if authenticated and principal_id is not None:
                where.append(
                    "(subject_principal_id=? OR (subject_principal_id IS NULL "
                    "AND visibility='profile_global' AND trust_tier IN ('canonical','owner_declared','owner_approved','verified_summary')))"
                )
                params.append(int(principal_id))
            else:
                where.append("subject_principal_id IS NULL AND visibility='profile_global' AND trust_tier IN ('canonical','owner_declared','owner_approved','verified_summary')")
            rows = self._get_conn().execute(
                f"""SELECT id, content, category, importance, memory_layer, memory_type,
                          subject_principal_id, trust_tier, content_hash, created_at, visibility
                   FROM memories WHERE {' AND '.join(where)} ORDER BY id""",
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def get_active_index_rows(self) -> List[Dict[str, Any]]:
        """Return active rows for local derivative-index maintenance only.

        This is not a retrieval API: callers must authorize IDs again before
        exposing results.  It exists so a rebuild does not weaken retrieval
        authorization or depend on a synthetic principal.
        """
        with self._lock:
            rows = self._get_conn().execute(
                """SELECT id, content, category, importance, memory_layer, memory_type,
                          subject_principal_id, trust_tier, content_hash, created_at, visibility
                   FROM memories WHERE status='active' AND retrieval_enabled=1 ORDER BY id"""
            ).fetchall()
            return [dict(row) for row in rows]

    def get_active_memories_by_ids(
        self, memory_ids: Sequence[int], *, principal_id: Optional[int] = None, authenticated: bool = False
    ) -> List[Dict[str, Any]]:
        if not authenticated or principal_id is None:
            return []
        ids = sorted({int(value) for value in memory_ids})
        if not ids:
            return []
        with self._lock:
            where = [
                f"id IN ({','.join('?' for _ in ids)})",
                "status='active'", "retrieval_enabled=1",
            ]
            params: List[Any] = list(ids)
            if authenticated and principal_id is not None:
                where.append("(subject_principal_id=? OR (subject_principal_id IS NULL AND visibility='profile_global' AND trust_tier IN ('canonical','owner_declared','owner_approved','verified_summary')))")
                params.append(int(principal_id))
            else:
                where.append("subject_principal_id IS NULL AND visibility='profile_global' AND trust_tier IN ('canonical','owner_declared','owner_approved','verified_summary')")
            rows = self._get_conn().execute(
                f"SELECT * FROM memories WHERE {' AND '.join(where)}", params
            ).fetchall()
            by_id = {int(row["id"]): dict(row) for row in rows}
            return [by_id[value] for value in ids if value in by_id]

    # --- Principals and canonical identity ---

    def upsert_principal(
        self,
        *,
        platform: str,
        platform_subject_id: str,
        principal_type: str = "person",
        status: str = "active",
    ) -> int:
        """Provision a principal from trusted local configuration.

        Retrieval and extraction must never call this method for unknown input.
        """
        with self._lock:
            conn = self._get_conn()
            now = utcnow()
            conn.execute(
                """INSERT INTO principals
                   (platform, platform_subject_id, principal_type, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(platform, platform_subject_id) DO UPDATE SET
                     principal_type=excluded.principal_type,
                     status=excluded.status,
                     updated_at=excluded.updated_at""",
                (platform.lower(), str(platform_subject_id), principal_type, status, now, now),
            )
            row = conn.execute(
                "SELECT id FROM principals WHERE platform=? AND platform_subject_id=?",
                (platform.lower(), str(platform_subject_id)),
            ).fetchone()
            conn.commit()
            return int(row["id"])

    def resolve_principal(self, platform: str, platform_subject_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not platform_subject_id:
            return None
        with self._lock:
            row = self._get_conn().execute(
                """SELECT * FROM principals
                   WHERE platform=? AND platform_subject_id=? AND status='active'""",
                ((platform or "unknown").lower(), str(platform_subject_id)),
            ).fetchone()
            return dict(row) if row else None

    def set_canonical_identity(
        self,
        principal_id: int,
        *,
        context_scope: str = "default",
        reference_name: Optional[str] = None,
        direct_address: Optional[str] = None,
        authority: str = "canonical",
    ) -> int:
        with self._lock:
            conn = self._get_conn()
            now = utcnow()
            old = conn.execute(
                """SELECT id FROM canonical_identity
                   WHERE principal_id=? AND context_scope=? AND status='active'""",
                (principal_id, context_scope),
            ).fetchone()
            old_id = int(old["id"]) if old else None
            if old_id:
                conn.execute(
                    "UPDATE canonical_identity SET status='superseded', valid_to=? WHERE id=?",
                    (now, old_id),
                )
            cursor = conn.execute(
                """INSERT INTO canonical_identity
                   (principal_id, context_scope, reference_name, direct_address,
                    authority, valid_from, supersedes_id, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'active')""",
                (principal_id, context_scope, reference_name, direct_address, authority, now, old_id),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def resolve_identity(self, principal_id: int, context_scope: str = "default") -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._get_conn().execute(
                """SELECT ci.*, p.platform, p.platform_subject_id
                   FROM canonical_identity ci
                   JOIN principals p ON p.id=ci.principal_id
                   WHERE ci.principal_id=? AND ci.status='active'
                     AND ci.context_scope IN (?, 'default')
                   ORDER BY CASE WHEN ci.context_scope=? THEN 0 ELSE 1 END, ci.id DESC
                   LIMIT 1""",
                (principal_id, context_scope, context_scope),
            ).fetchone()
            return dict(row) if row else None

    # --- Memories ---

    def store_memory(
        self,
        content: str,
        category: str = "general",
        source_session: Optional[str] = None,
        source: str = "explicit",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        memory_layer: str = "l1",
        memory_type: str = "fact",
        subject_principal_id: Optional[int] = None,
        trust_tier: Optional[str] = None,
        status: str = "active",
        retrieval_enabled: bool = True,
        supersedes_id: Optional[int] = None,
        source_type: Optional[str] = None,
        source_locator: Optional[str] = None,
        sender_principal_id: Optional[int] = None,
        extractor_version: str = "manual-v1",
        visibility: Optional[str] = None,
        deduplicate: bool = True,
        commit: bool = True,
        _profile_global_capability: Optional[object] = None,
    ) -> int:
        clean = " ".join((content or "").split()).strip()
        if not clean:
            raise ValueError("memory content is required")
        if re.search(
            r"(?i)\b(?:bearer\s+[\w._~+/-]{12,}|eyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}|(?:cookie|set-cookie)\s*[:=]|(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://|(?:password|passwd|pwd|api[_-]?key|apikey|token|secret|auth)\s*(?::|=|\bis\b|\bwas\b|\bto be\b)\s*\S+|(?:password|passwd|pwd|api[_-]?key|apikey)\s+[A-Za-z0-9._~+/-]{8,}|(?:token|secret|auth)\s+[A-Za-z0-9._~+/-]{8,})",
            clean,
        ):
            raise ValueError("credential-like content is not storable")
        if re.search(r"(?i)\b(?:ignore|override|disregard)\b.{0,80}\b(?:instruction|policy|system|prompt)\b", clean):
            raise ValueError("instruction-like content is not storable")
        if memory_layer not in VALID_LAYERS:
            raise ValueError(f"invalid memory layer: {memory_layer}")
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid memory status: {status}")
        trust = trust_tier or self._default_trust(source)
        if trust not in VALID_TRUST_TIERS:
            raise ValueError(f"invalid trust tier: {trust}")
        can_widen_profile_scope = (
            _profile_global_capability is _PROFILE_GLOBAL_CAPABILITY
            and subject_principal_id is None
            and trust in PROFILE_GLOBAL_TRUST_TIERS
            and source in PROFILE_GLOBAL_SOURCES
        )
        effective_visibility = visibility or ("profile_global" if can_widen_profile_scope else "subject")
        if effective_visibility not in VALID_VISIBILITIES:
            raise ValueError(f"invalid memory visibility: {effective_visibility}")
        if effective_visibility == "profile_global" and not can_widen_profile_scope:
            raise ValueError("profile-global memory requires internal high-trust local provenance")
        if effective_visibility == "subject" and subject_principal_id is None:
            retrieval_enabled = False
        importance = max(0.0, min(float(importance), 1.0))
        normalized = normalize_content(clean)
        digest = content_digest(clean)
        now = utcnow()

        with self._lock:
            conn = self._get_conn()
            existing = conn.execute(
                """SELECT id FROM memories
                   WHERE content_hash=? AND status='active'
                     AND COALESCE(subject_principal_id, -1)=COALESCE(?, -1)
                   LIMIT 1""",
                (digest, subject_principal_id),
            ).fetchone()
            if existing and deduplicate:
                existing_id = int(existing["id"])
                existing_row = conn.execute(
                    "SELECT trust_tier, importance, retrieval_enabled, visibility FROM memories WHERE id=?",
                    (existing_id,),
                ).fetchone()
                if existing_row and TRUST_WEIGHTS.get(trust, 0.0) > TRUST_WEIGHTS.get(existing_row["trust_tier"], 0.0):
                    conn.execute(
                        """UPDATE memories SET trust_tier=?, importance=MAX(importance, ?),
                           retrieval_enabled=?, visibility=?, updated_at=? WHERE id=?""",
                        (trust, importance, 1 if retrieval_enabled else 0, effective_visibility, now, existing_id),
                    )
                    conn.execute(
                        """UPDATE memory_provenance SET sender_principal_id=COALESCE(sender_principal_id, ?),
                           extractor_version=?, observed_at=? WHERE memory_id=?""",
                        (sender_principal_id, extractor_version, now, existing_id),
                    )
                    if commit:
                        conn.commit()
                return existing_id
            try:
                cursor = conn.execute(
                    """INSERT INTO memories
                       (content, category, source_session, source, importance,
                        created_at, updated_at, metadata, memory_layer, memory_type,
                        subject_principal_id, trust_tier, status, retrieval_enabled,
                        valid_from, supersedes_id, normalized_content, content_hash, visibility)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        clean,
                        category,
                        source_session,
                        source,
                        importance,
                        now,
                        now,
                        json.dumps(metadata, sort_keys=True) if metadata else None,
                        memory_layer,
                        memory_type,
                        subject_principal_id,
                        trust,
                        status,
                        1 if retrieval_enabled else 0,
                        now,
                        supersedes_id,
                        normalized,
                        digest,
                        effective_visibility,
                    ),
                )
                memory_id = int(cursor.lastrowid)
                conn.execute(
                    """INSERT INTO memory_provenance
                       (memory_id, source_type, source_locator, source_session,
                        sender_principal_id, observed_at, extractor_version, content_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        memory_id,
                        source_type or source,
                        source_locator,
                        source_session,
                        sender_principal_id,
                        now,
                        extractor_version,
                        digest,
                    ),
                )
                if supersedes_id:
                    self._supersede_locked(conn, supersedes_id, memory_id, now)
                if commit:
                    conn.commit()
                return memory_id
            except Exception:
                conn.rollback()
                raise

    @staticmethod
    def _default_trust(source: str) -> str:
        if source == "explicit":
            return "owner_declared"
        if source in {"migration", "vera-migration"}:
            return "imported_history"
        if source == "hermes-memory-tool":
            return "assistant_summary"
        return "observed_chat"

    @staticmethod
    def _supersede_locked(conn: sqlite3.Connection, old_id: int, new_id: int, now: str) -> None:
        conn.execute(
            """UPDATE memories
               SET status='superseded', retrieval_enabled=0, valid_to=?, updated_at=?
               WHERE id=? AND status='active'""",
            (now, now, old_id),
        )
        conn.execute(
            """INSERT OR IGNORE INTO memory_links(parent_id, child_id, relation, created_at)
               VALUES (?, ?, 'superseded_by', ?)""",
            (old_id, new_id, now),
        )

    def search_memories(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
        *,
        allowed_ids: Optional[Sequence[int]] = None,
        principal_id: Optional[int] = None,
        authenticated: bool = False,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        if not query or not authenticated or principal_id is None:
            # An unauthenticated caller receives no memory context, including
            # profile-global rows.  A profile-global record is global only among
            # authenticated principals, never public context.
            return []
        with self._lock:
            conn = self._get_conn()
            where = ["fts_memories MATCH ?", "m.status='active'", "m.retrieval_enabled=1"]
            params: List[Any] = [query]
            if category:
                where.append("m.category=?")
                params.append(category)
            if authenticated and principal_id is not None:
                where.append("(m.subject_principal_id=? OR (m.subject_principal_id IS NULL AND m.visibility='profile_global' AND m.trust_tier IN ('canonical','owner_declared','owner_approved','verified_summary')))")
                params.append(int(principal_id))
            else:
                where.append("m.subject_principal_id IS NULL AND m.visibility='profile_global' AND m.trust_tier IN ('canonical','owner_declared','owner_approved','verified_summary')")
            if allowed_ids is not None:
                ids = [int(value) for value in allowed_ids]
                if not ids:
                    return []
                where.append("m.id IN (%s)" % ",".join("?" for _ in ids))
                params.extend(ids)
            params.append(limit)
            try:
                rows = conn.execute(
                    f"""SELECT m.id, m.content, m.category, m.source, m.importance,
                               m.created_at, m.access_count, m.memory_layer, m.memory_type,
                               m.subject_principal_id, m.trust_tier, m.status,
                               m.retrieval_enabled, m.supersedes_id,
                               bm25(fts_memories) AS rank
                        FROM fts_memories
                        JOIN memories m ON m.id=fts_memories.rowid
                        WHERE {' AND '.join(where)}
                        ORDER BY rank ASC LIMIT ?""",
                    params,
                ).fetchall()
            except sqlite3.OperationalError as exc:
                logger.debug("FTS query failed safely: %s", exc)
                return []
            results = [dict(row) for row in rows]
            if results:
                ids = [row["id"] for row in results]
                conn.execute(
                    f"""UPDATE memories SET access_count=access_count+1, last_accessed=?
                         WHERE id IN ({','.join('?' for _ in ids)})""",
                    [utcnow(), *ids],
                )
                conn.commit()
            return results

    def get_recent_memories(
        self, limit: int = 20, category: Optional[str] = None, *,
        principal_id: Optional[int] = None, authenticated: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return recent rows only for an authenticated authorized principal."""
        if not authenticated or principal_id is None:
            return []
        limit = max(1, min(int(limit), 100))
        with self._lock:
            params: List[Any] = []
            where = ["status='active'", "retrieval_enabled=1",
                "(subject_principal_id=? OR (subject_principal_id IS NULL AND visibility='profile_global' AND trust_tier IN ('canonical','owner_declared','owner_approved','verified_summary')))"]
            params.append(int(principal_id))
            if category:
                where.append("category=?")
                params.append(category)
            params.append(limit)
            rows = self._get_conn().execute(
                f"""SELECT * FROM memories WHERE {' AND '.join(where)}
                    ORDER BY created_at DESC LIMIT ?""",
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def get_memory(
        self, memory_id: int, *, principal_id: Optional[int] = None, authenticated: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Return one retrievable row only for an authenticated authorized principal."""
        if not authenticated or principal_id is None:
            return None
        rows = self.get_active_memories_by_ids(
            [memory_id], principal_id=principal_id, authenticated=True
        )
        return rows[0] if rows else None

    def _get_memory_internal(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Private record access for local maintenance; never expose to tool paths."""
        with self._lock:
            row = self._get_conn().execute(
                "SELECT * FROM memories WHERE id=?", (int(memory_id),)
            ).fetchone()
            return dict(row) if row else None

    def get_memory_count(self, *, active_only: bool = False) -> int:
        with self._lock:
            sql = "SELECT COUNT(*) AS cnt FROM memories"
            if active_only:
                sql += " WHERE status='active' AND retrieval_enabled=1"
            return int(self._get_conn().execute(sql).fetchone()["cnt"])

    def get_categories(self) -> List[Dict[str, Any]]:
        rows = self._get_conn().execute(
            """SELECT category, COUNT(*) AS count FROM memories
               GROUP BY category ORDER BY count DESC"""
        ).fetchall()
        return [dict(row) for row in rows]

    def delete_memory(self, memory_id: int) -> bool:
        """Soft-delete a memory so audit history is retained."""
        with self._lock:
            cursor = self._get_conn().execute(
                """UPDATE memories SET status='deleted', retrieval_enabled=0,
                   valid_to=?, updated_at=? WHERE id=? AND status!='deleted'""",
                (utcnow(), utcnow(), int(memory_id)),
            )
            self._get_conn().commit()
            return cursor.rowcount > 0

    def add_memory_link(self, parent_id: int, child_id: int, relation: str) -> None:
        if relation not in {
            "evidence_for", "superseded_by", "scenario_member",
            "profile_source", "restored_from",
        }:
            raise ValueError("invalid memory relation")
        with self._lock:
            self._get_conn().execute(
                """INSERT OR IGNORE INTO memory_links(parent_id, child_id, relation, created_at)
                   VALUES (?, ?, ?, ?)""",
                (int(parent_id), int(child_id), relation, utcnow()),
            )
            self._get_conn().commit()

    def create_layered_summary(
        self,
        content: str,
        *,
        layer: str,
        source_ids: Sequence[int],
        category: str = "summary",
        series_key: Optional[str] = None,
        generator_version: str = "layered-summary-v2",
        owner_reviewed: bool = False,
        identity_adjacent: bool = False,
        _owner_approval_capability: Optional[object] = None,
    ) -> int:
        if layer not in {"l2", "l3"}:
            raise ValueError("layered summaries must be l2 or l3")
        unique_ids = sorted({int(item) for item in source_ids})
        if not unique_ids:
            raise ValueError("layered summary requires source memories")
        sources: List[Dict[str, Any]] = []
        for memory_id in unique_ids:
            source = self._get_memory_internal(memory_id)
            if not source or source["status"] in {"deleted", "quarantined"}:
                raise ValueError(f"source memory is unavailable: {memory_id}")
            sources.append(source)
        # A summary can be shared only when every input was already shareable
        # by authenticated principals.  Otherwise it stays review-gated and
        # cannot launder subject/audit-only evidence into global retrieval.
        globally_shareable_sources = all(
            source["subject_principal_id"] is None
            and source["visibility"] == "profile_global"
            and source["trust_tier"] in PROFILE_GLOBAL_TRUST_TIERS
            and source["source"] in PROFILE_GLOBAL_SOURCES
            for source in sources
        )
        source_principals = {source["subject_principal_id"] for source in sources}
        summary_subject_id = (
            next(iter(source_principals))
            if len(source_principals) == 1 and next(iter(source_principals)) is not None
            else None
        )

        approved_by_owner = (
            bool(owner_reviewed) and _owner_approval_capability is _OWNER_APPROVAL_CAPABILITY
        )
        key = (series_key or f"{layer}:{category}").strip()
        if not key or len(key) > 200:
            raise ValueError("invalid summary series key")
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                previous = conn.execute(
                    """SELECT sg.summary_id, sg.generation
                       FROM summary_generations sg JOIN memories m ON m.id=sg.summary_id
                       WHERE sg.series_key=? AND m.status='active' AND m.retrieval_enabled=1
                       ORDER BY sg.generation DESC LIMIT 1""", (key,)
                ).fetchone()
                generation = int(conn.execute(
                    "SELECT COALESCE(MAX(generation), 0) + 1 FROM summary_generations WHERE series_key=?", (key,)
                ).fetchone()[0])
                review_required = (
                    not globally_shareable_sources
                    or bool(layer == "l3" and identity_adjacent and not approved_by_owner)
                )
                review_status = "pending" if review_required else ("approved" if approved_by_owner else "not_required")
                supersedes_id = int(previous["summary_id"]) if previous and not review_required else None
                trust_tier = "owner_approved" if approved_by_owner and not review_required else (
                    "assistant_summary" if review_required else "verified_summary"
                )
                summary_id = self.store_memory(
                    content, category=category, source="generated-summary",
                    memory_layer=layer, memory_type="summary", trust_tier=trust_tier,
                    subject_principal_id=None if not review_required else summary_subject_id,
                    retrieval_enabled=not review_required,
                    visibility="profile_global" if not review_required else "subject",
                    deduplicate=False, commit=False, supersedes_id=supersedes_id,
                    _profile_global_capability=_PROFILE_GLOBAL_CAPABILITY,
                    extractor_version=generator_version,
                    metadata={"series_key": key, "generation": generation,
                              "identity_adjacent": bool(identity_adjacent)},
                )
                conn.execute(
                    """INSERT INTO summary_generations
                       (summary_id, series_key, layer, generation, generator_version,
                        owner_review_status, identity_adjacent, supersedes_summary_id, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (summary_id, key, layer, generation, generator_version, review_status,
                     1 if identity_adjacent else 0, supersedes_id, utcnow()),
                )
                relation = "scenario_member" if layer == "l2" else "profile_source"
                conn.executemany(
                    """INSERT OR IGNORE INTO memory_links(parent_id, child_id, relation, created_at)
                       VALUES (?, ?, ?, ?)""",
                    [(source_id, summary_id, relation, utcnow()) for source_id in unique_ids],
                )
                conn.commit()
                return summary_id
            except Exception:
                conn.rollback()
                raise

    def review_layered_summary(
        self, summary_id: int, *, approved: bool, _owner_approval_capability: Optional[object] = None
    ) -> Dict[str, Any]:
        """Approve or reject an identity-adjacent L3 summary atomically."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                generation = conn.execute(
                    "SELECT * FROM summary_generations WHERE summary_id=?", (int(summary_id),)
                ).fetchone()
                if not generation:
                    raise ValueError("summary generation does not exist")
                if generation["owner_review_status"] != "pending":
                    raise ValueError("summary is not pending owner review")
                if approved and _owner_approval_capability is not _OWNER_APPROVAL_CAPABILITY:
                    raise ValueError("summary approval requires trusted owner approval capability")
                source_rows = conn.execute(
                    """SELECT m.subject_principal_id, m.visibility, m.trust_tier, m.source
                       FROM memory_links l JOIN memories m ON m.id=l.parent_id
                       WHERE l.child_id=?""", (int(summary_id),)
                ).fetchall()
                if approved and (not source_rows or any(
                    row["subject_principal_id"] is not None
                    or row["visibility"] != "profile_global"
                    or row["trust_tier"] not in PROFILE_GLOBAL_TRUST_TIERS
                    or row["source"] not in PROFILE_GLOBAL_SOURCES
                    for row in source_rows
                )):
                    raise ValueError("summary sources are not eligible for profile-global approval")
                now = utcnow()
                if not approved:
                    conn.execute("UPDATE memories SET status='quarantined', retrieval_enabled=0, valid_to=?, updated_at=? WHERE id=?", (now, now, int(summary_id)))
                    conn.execute("UPDATE summary_generations SET owner_review_status='rejected' WHERE summary_id=?", (int(summary_id),))
                    conn.commit()
                    return {"summary_id": int(summary_id), "status": "rejected"}
                previous = conn.execute(
                    """SELECT sg.summary_id FROM summary_generations sg JOIN memories m ON m.id=sg.summary_id
                       WHERE sg.series_key=? AND sg.summary_id!=? AND m.status='active' AND m.retrieval_enabled=1
                       ORDER BY sg.generation DESC LIMIT 1""", (generation["series_key"], int(summary_id))
                ).fetchone()
                if previous:
                    self._supersede_locked(conn, int(previous["summary_id"]), int(summary_id), now)
                conn.execute("""UPDATE memories SET status='active', retrieval_enabled=1,
                   trust_tier='owner_approved', visibility='profile_global', valid_from=?, valid_to=NULL, updated_at=? WHERE id=?""", (now, now, int(summary_id)))
                conn.execute("UPDATE summary_generations SET owner_review_status='approved', supersedes_summary_id=? WHERE summary_id=?", (int(previous["summary_id"]) if previous else None, int(summary_id)))
                conn.commit()
                return {"summary_id": int(summary_id), "status": "approved"}
            except Exception:
                conn.rollback()
                raise

    def restore_layered_summary(self, summary_id: int) -> Dict[str, Any]:
        """Re-activate a retained generation and supersede the current one atomically."""
        target_id = int(summary_id)
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                target = conn.execute(
                    """SELECT sg.*, m.status FROM summary_generations sg
                       JOIN memories m ON m.id=sg.summary_id WHERE sg.summary_id=?""", (target_id,)
                ).fetchone()
                if not target or target["owner_review_status"] == "rejected":
                    raise ValueError("summary generation is not restorable")
                current = conn.execute(
                    """SELECT sg.summary_id FROM summary_generations sg JOIN memories m ON m.id=sg.summary_id
                       WHERE sg.series_key=? AND m.status='active' AND m.retrieval_enabled=1
                       ORDER BY sg.generation DESC LIMIT 1""", (target["series_key"],)
                ).fetchone()
                if current and int(current["summary_id"]) == target_id:
                    conn.commit()
                    return {"summary_id": target_id, "status": "already_active"}
                now = utcnow()
                if current:
                    self._supersede_locked(conn, int(current["summary_id"]), target_id, now)
                    conn.execute("""INSERT OR IGNORE INTO memory_links (parent_id, child_id, relation, created_at)
                       VALUES (?, ?, 'restored_from', ?)""", (int(current["summary_id"]), target_id, now))
                conn.execute("""UPDATE memories SET status='active', retrieval_enabled=1,
                   valid_from=?, valid_to=NULL, updated_at=? WHERE id=?""", (now, now, target_id))
                conn.commit()
                return {"summary_id": target_id, "status": "restored"}
            except Exception:
                conn.rollback()
                raise

    def find_near_duplicates(
        self,
        content: str,
        threshold: float = 0.85,
        limit: int = 20,
        *,
        category: Optional[str] = None,
        subject_principal_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        target = set(re.findall(r"[^\W_]+", normalize_content(content), re.UNICODE))
        if not target:
            return []
        where = ["status='active'"]
        params: List[Any] = []
        if category is not None:
            where.append("category=?")
            params.append(category)
            if subject_principal_id is None:
                where.append("subject_principal_id IS NULL")
        if subject_principal_id is not None:
            where.append("subject_principal_id=?")
            params.append(int(subject_principal_id))
        params.append(max(1, min(int(limit), 200)))
        rows = self._get_conn().execute(
            f"""SELECT id, content, content_hash FROM memories
                WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT ?""",
            params,
        ).fetchall()
        matches = []
        for row in rows:
            tokens = set(re.findall(r"[^\W_]+", normalize_content(row["content"]), re.UNICODE))
            union = target | tokens
            similarity = len(target & tokens) / len(union) if union else 0.0
            if similarity >= threshold:
                matches.append({**dict(row), "similarity": similarity})
        return sorted(matches, key=lambda item: item["similarity"], reverse=True)

    def create_conflict_review(self, memory_id_a: int, memory_id_b: int, reason: str) -> int:
        with self._lock:
            cursor = self._get_conn().execute(
                """INSERT INTO conflict_review(memory_id_a, memory_id_b, reason, created_at)
                   VALUES (?, ?, ?, ?)""",
                (int(memory_id_a), int(memory_id_b), reason[:500], utcnow()),
            )
            self._get_conn().commit()
            return int(cursor.lastrowid)

    def store_extracted_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Store one extractor result with duplicate and correction precedence."""
        content = str(memory.get("content", "")).strip()
        if not content:
            raise ValueError("memory content is required")
        category = str(memory.get("category", "general"))
        subject_principal_id = memory.get("subject_principal_id")
        trust = memory.get("trust_tier") or self._default_trust(
            str(memory.get("source", "conversation"))
        )

        near = self.find_near_duplicates(
            content,
            threshold=0.90,
            limit=100,
            category=category,
            subject_principal_id=subject_principal_id,
        )
        if near:
            return {"memory_id": int(near[0]["id"]), "action": "duplicate"}

        key = claim_key(content)
        predecessor = None
        if key:
            rows = self._get_conn().execute(
                """SELECT id, content, trust_tier, memory_type FROM memories
                   WHERE status='active' AND category=?
                     AND COALESCE(subject_principal_id, -1)=COALESCE(?, -1)
                   ORDER BY id DESC LIMIT 200""",
                (category, subject_principal_id),
            ).fetchall()
            predecessor = next(
                (dict(row) for row in rows if claim_key(row["content"]) == key),
                None,
            )

        memory_to_store = dict(memory)
        action = "stored"
        if predecessor:
            old_weight = TRUST_WEIGHTS.get(predecessor["trust_tier"], 0.0)
            new_weight = TRUST_WEIGHTS.get(trust, 0.0)
            is_correction = memory.get("memory_type") == "correction"
            if (is_correction and new_weight >= old_weight) or new_weight > old_weight:
                memory_to_store["supersedes_id"] = predecessor["id"]
                action = "superseded"
            else:
                memory_to_store["retrieval_enabled"] = False
                action = "conflict_review"

        memory_id = self.store_memory(**memory_to_store)
        if predecessor and action == "conflict_review":
            self.create_conflict_review(
                predecessor["id"],
                memory_id,
                f"same claim subject ({key}) with non-superior authority",
            )
        return {"memory_id": memory_id, "action": action}

    def bulk_store(
        self,
        memories: Iterable[Dict[str, Any]],
        *,
        atomic: bool = False,
        required_active_principal_id: Optional[int] = None,
    ) -> int:
        if required_active_principal_id is not None and not atomic:
            raise ValueError("principal-validated bulk storage must be atomic")
        count = 0
        with self._lock:
            conn = self._get_conn()
            try:
                if atomic:
                    conn.execute("BEGIN IMMEDIATE")
                if required_active_principal_id is not None:
                    principal = conn.execute(
                        "SELECT id FROM principals WHERE id=? AND status='active'",
                        (int(required_active_principal_id),),
                    ).fetchone()
                    if principal is None:
                        raise ValueError("migration subject principal does not exist or is inactive")
                for memory in memories:
                    content = str(memory.get("content", "")).strip()
                    if not content:
                        continue
                    before = self.get_memory_count()
                    self.store_memory(
                        content=content,
                        category=memory.get("category", "general"),
                        source_session=memory.get("source_session"),
                        source=memory.get("source", "migration"),
                        importance=memory.get("importance", 0.5),
                        metadata=memory.get("metadata"),
                        memory_layer=memory.get("memory_layer", "l1"),
                        memory_type=memory.get("memory_type", "fact"),
                        subject_principal_id=memory.get("subject_principal_id"),
                        trust_tier=memory.get("trust_tier"),
                        status=memory.get("status", "active"),
                        retrieval_enabled=memory.get("retrieval_enabled", True),
                        supersedes_id=memory.get("supersedes_id"),
                        source_type=memory.get("source_type"),
                        source_locator=memory.get("source_locator"),
                        sender_principal_id=memory.get("sender_principal_id"),
                        extractor_version=memory.get("extractor_version", "bulk-v1"),
                        commit=not atomic,
                    )
                    if self.get_memory_count() > before:
                        count += 1
                if atomic:
                    conn.commit()
                return count
            except Exception:
                if atomic:
                    conn.rollback()
                raise

    # --- Sessions ---

    def start_session(self, session_id: str) -> None:
        with self._lock:
            self._get_conn().execute(
                "INSERT OR IGNORE INTO sessions(session_id, started_at) VALUES (?, ?)",
                (session_id, utcnow()),
            )
            self._get_conn().commit()

    def end_session(self, session_id: str, summary: Optional[str] = None, turn_count: int = 0) -> None:
        with self._lock:
            conn = self._get_conn()
            memories_created = conn.execute(
                "SELECT COUNT(*) AS cnt FROM memories WHERE source_session=?",
                (session_id,),
            ).fetchone()["cnt"]
            conn.execute(
                """UPDATE sessions SET ended_at=?, summary=?, turn_count=?, memories_created=?
                   WHERE session_id=?""",
                (utcnow(), summary, int(turn_count), memories_created, session_id),
            )
            conn.commit()
