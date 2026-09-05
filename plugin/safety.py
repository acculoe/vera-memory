"""Non-destructive backup, copied-database migration, and cleanup utilities."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable

if __package__:
    from .models import MemoryDB, utcnow
else:
    from models import MemoryDB, utcnow

_TRANSPORT_PREFIX = re.compile(r"^\[[^\]\r\n|]{1,128}\|[0-9]{1,32}\]\s*")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sqlite_integrity(path: Path) -> str:
    conn = sqlite3.connect(str(path))
    try:
        return str(conn.execute("PRAGMA integrity_check").fetchone()[0])
    finally:
        conn.close()


def logical_fingerprint(path: Path) -> str:
    """Hash canonical table contents while ignoring SQLite file layout."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        tables = [
            row[0]
            for row in conn.execute(
                """SELECT name FROM sqlite_master
                   WHERE type='table' AND name NOT LIKE 'fts_%'
                     AND name NOT LIKE 'sqlite_%' ORDER BY name"""
            )
        ]
        digest = hashlib.sha256()
        for table in tables:
            columns = [row[1] for row in conn.execute(f"PRAGMA table_info('{table}')")]
            order = " ORDER BY rowid" if columns else ""
            digest.update(table.encode("utf-8"))
            digest.update(json.dumps(columns).encode("utf-8"))
            for row in conn.execute(f'SELECT * FROM "{table}"{order}'):
                values = [row[column] for column in columns]
                digest.update(json.dumps(values, sort_keys=True, default=str).encode("utf-8"))
        return digest.hexdigest()
    finally:
        conn.close()


def backup_database(source: Path, destination: Path) -> Dict[str, Any]:
    """Create a consistent SQLite backup without modifying the source."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination:
        raise ValueError("backup destination must differ from source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.unlink(missing_ok=True)
    src = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    dst = sqlite3.connect(str(temp))
    try:
        src.backup(dst)
        dst.commit()
    finally:
        dst.close()
        src.close()
    if sqlite_integrity(temp) != "ok":
        temp.unlink(missing_ok=True)
        raise RuntimeError("backup failed SQLite integrity check")
    with temp.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, destination)
    return {
        "source": str(source),
        "backup": str(destination),
        "integrity": "ok",
        "logical_fingerprint": logical_fingerprint(destination),
        "file_sha256": file_sha256(destination),
    }


def restore_to_copy(backup: Path, destination: Path) -> Dict[str, Any]:
    """Restore a backup into a separate path and verify the result."""
    backup = Path(backup).resolve()
    destination = Path(destination).resolve()
    if backup == destination:
        raise ValueError("restore destination must differ from backup")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".restore.tmp")
    temp.unlink(missing_ok=True)
    shutil.copy2(backup, temp)
    if sqlite_integrity(temp) != "ok":
        temp.unlink(missing_ok=True)
        raise RuntimeError("restored copy failed SQLite integrity check")
    with temp.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, destination)
    source_fingerprint = logical_fingerprint(backup)
    restored_fingerprint = logical_fingerprint(destination)
    if source_fingerprint != restored_fingerprint:
        raise RuntimeError("restored copy does not match backup")
    return {
        "restored": str(destination),
        "integrity": "ok",
        "logical_fingerprint": restored_fingerprint,
    }


def prepare_live_migration_package(source: Path, package_dir: Path) -> Dict[str, Any]:
    """Create and verify a backup plus separate restoration package.

    This is deliberately explicit; callers must opt in before changing a live
    database path.  The returned manifest contains no memory content.
    """
    source = Path(source).resolve()
    package_dir = Path(package_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    package_dir.mkdir(parents=True, exist_ok=True)
    backup = package_dir / "memory-preupgrade.db"
    restored = package_dir / "memory-restore-verified.db"
    backup_report = backup_database(source, backup)
    restore_report = restore_to_copy(backup, restored)
    if backup_report["logical_fingerprint"] != restore_report["logical_fingerprint"]:
        raise RuntimeError("restoration package fingerprint mismatch")
    manifest = {
        "backup": str(backup),
        "restore_copy": str(restored),
        "integrity": "ok",
        "verified": True,
    }
    (package_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest


def classify_legacy_rows(db: MemoryDB, *, apply: bool = False) -> Dict[str, Any]:
    """Identify raw transport-labelled legacy rows and optionally quarantine them.

    Rows are preserved.  Applying the classifier marks matching imported or
    observed history as historical and retrieval-disabled, with an audit row.
    """
    conn = db._get_conn()
    rows = conn.execute(
        """SELECT id, content, source, status, retrieval_enabled
           FROM memories WHERE status='active'"""
    ).fetchall()
    candidates = [
        dict(row)
        for row in rows
        if _TRANSPORT_PREFIX.match(row["content"] or "")
        and row["source"] in {
            "conversation", "migration", "vera-migration", "hermes-memory-tool"
        }
    ]
    report = {
        "scanned": len(rows),
        "classified_historical": len(candidates),
        "applied": bool(apply),
        "deleted": 0,
    }
    if not apply:
        return report
    now = utcnow()
    try:
        for row in candidates:
            before = {
                "status": row["status"],
                "retrieval_enabled": row["retrieval_enabled"],
            }
            after = {"status": "historical", "retrieval_enabled": 0}
            conn.execute(
                """UPDATE memories SET status='historical', retrieval_enabled=0,
                   valid_to=?, updated_at=? WHERE id=?""",
                (now, now, row["id"]),
            )
            conn.execute(
                """INSERT INTO migration_audit
                   (migration_name, memory_id, before_json, after_json, applied_at)
                   VALUES ('legacy-transport-history-v1', ?, ?, ?, ?)""",
                (
                    row["id"],
                    json.dumps(before, sort_keys=True),
                    json.dumps(after, sort_keys=True),
                    now,
                ),
            )
        db.rebuild_fts(commit=False)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return report


def migrate_database_copy(
    source: Path,
    destination: Path,
    *,
    dry_run: bool = True,
    classify_history: bool = True,
) -> Dict[str, Any]:
    """Migrate only a copy; the source path is opened read-only for backup."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination:
        raise ValueError("migration destination must differ from source")
    source_hash_before = file_sha256(source)
    source_fingerprint_before = logical_fingerprint(source)

    if dry_run:
        temp_dir = Path(tempfile.mkdtemp(prefix="vera-migration-dry-run-"))
        working = temp_dir / "memory.db"
    else:
        working = destination
    try:
        backup_database(source, working)
        db = MemoryDB(working)
        db.connect()
        try:
            cleanup = classify_legacy_rows(db, apply=classify_history)
            result = {
                "dry_run": bool(dry_run),
                "source_unchanged": False,
                "schema_version": db.get_schema_version(),
                "integrity": db.integrity_check(),
                "total_memories": db.get_memory_count(),
                "active_memories": db.get_memory_count(active_only=True),
                "cleanup": cleanup,
                "migrated_fingerprint": logical_fingerprint(working),
            }
        finally:
            db.close()
        result["source_unchanged"] = (
            source_hash_before == file_sha256(source)
            and source_fingerprint_before == logical_fingerprint(source)
        )
        if not result["source_unchanged"]:
            raise RuntimeError("source database changed during copied migration")
        return result
    finally:
        if dry_run:
            shutil.rmtree(temp_dir, ignore_errors=True)
