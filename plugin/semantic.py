"""Replaceable semantic retrieval components with SQLite authority.

No model is bundled or downloaded by this module. Production embeddings must be
provided by an owner-approved, hash-pinned adapter. SQLite controls which memory
IDs may participate; vector artifacts are always rebuildable derivatives.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

if __package__:
    from .models import MemoryDB, utcnow
else:
    from models import MemoryDB, utcnow


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str
    model_hash: str
    dimensions: int

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("embedding model name is required")
        if len(self.model_hash) < 16:
            raise ValueError("embedding model hash must be pinned")
        if self.dimensions <= 0 or self.dimensions > 65536:
            raise ValueError("invalid embedding dimensions")


class EmbeddingAdapter(Protocol):
    spec: EmbeddingSpec

    def encode(self, texts: Sequence[str], *, timeout_seconds: float) -> List[List[float]]:
        ...


class SemanticBackend(Protocol):
    def search(
        self,
        query: str,
        *,
        allowed_ids: Sequence[int],
        limit: int,
        timeout_seconds: float,
    ) -> List[Dict[str, Any]]:
        ...


def _validate_vector(vector: Sequence[float], dimensions: int) -> List[float]:
    if len(vector) != dimensions:
        raise ValueError(f"expected {dimensions} dimensions, got {len(vector)}")
    output = [float(value) for value in vector]
    if not all(math.isfinite(value) for value in output):
        raise ValueError("embedding contains non-finite values")
    return output


def _pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes, dimensions: int) -> List[float]:
    expected = dimensions * 4
    if len(blob) != expected:
        raise ValueError("stored vector byte length is invalid")
    return list(struct.unpack(f"<{dimensions}f", blob))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return -1.0
    return dot / (norm_a * norm_b)


class SQLiteVectorStore:
    """Rebuildable vector storage keyed by stable external memory IDs."""

    def __init__(self, db: MemoryDB, spec: EmbeddingSpec):
        spec.validate()
        self.db = db
        self.spec = spec

    def upsert(self, rows: Iterable[Tuple[int, str, Sequence[float]]]) -> int:
        prepared = []
        for memory_id, content_hash, vector in rows:
            clean = _validate_vector(vector, self.spec.dimensions)
            prepared.append(
                (
                    int(memory_id),
                    self.spec.name,
                    self.spec.model_hash,
                    self.spec.dimensions,
                    _pack(clean),
                    str(content_hash),
                    utcnow(),
                )
            )
        if not prepared:
            return 0
        conn = self.db._get_conn()
        try:
            conn.executemany(
                """INSERT INTO memory_embeddings
                   (memory_id, model_name, model_hash, dimensions, vector,
                    content_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(memory_id, model_hash) DO UPDATE SET
                     model_name=excluded.model_name,
                     dimensions=excluded.dimensions,
                     vector=excluded.vector,
                     content_hash=excluded.content_hash,
                     created_at=excluded.created_at""",
                prepared,
            )
            conn.executemany(
                """UPDATE memories SET embedding_status='ready', embedding_model=?
                   WHERE id=?""",
                [(self.spec.model_hash, row[0]) for row in prepared],
            )
            conn.commit()
            return len(prepared)
        except Exception:
            conn.rollback()
            raise

    def load(self, allowed_ids: Sequence[int]) -> Dict[int, List[float]]:
        ids = sorted({int(value) for value in allowed_ids})
        if not ids:
            return {}
        rows = self.db._get_conn().execute(
            f"""SELECT e.memory_id, e.dimensions, e.vector
                 FROM memory_embeddings e
                 JOIN memories m ON m.id=e.memory_id
                 WHERE e.model_hash=?
                   AND e.memory_id IN ({','.join('?' for _ in ids)})
                   AND e.content_hash=m.content_hash
                   AND m.status='active' AND m.retrieval_enabled=1""",
            [self.spec.model_hash, *ids],
        ).fetchall()
        return {
            int(row["memory_id"]): _unpack(row["vector"], int(row["dimensions"]))
            for row in rows
            if int(row["dimensions"]) == self.spec.dimensions
        }

    def remove(self, memory_ids: Sequence[int]) -> int:
        ids = sorted({int(value) for value in memory_ids})
        if not ids:
            return 0
        conn = self.db._get_conn()
        cursor = conn.execute(
            f"DELETE FROM memory_embeddings WHERE model_hash=? AND memory_id IN ({','.join('?' for _ in ids)})",
            [self.spec.model_hash, *ids],
        )
        conn.commit()
        return int(cursor.rowcount)

    def deterministic_hash(self) -> str:
        rows = self.db._get_conn().execute(
            """SELECT memory_id, dimensions, vector, content_hash
               FROM memory_embeddings WHERE model_hash=? ORDER BY memory_id""",
            (self.spec.model_hash,),
        ).fetchall()
        digest = hashlib.sha256()
        digest.update(self.spec.model_hash.encode("utf-8"))
        for row in rows:
            digest.update(struct.pack("<QI", int(row["memory_id"]), int(row["dimensions"])))
            digest.update(row["vector"])
            digest.update(row["content_hash"].encode("utf-8"))
        return digest.hexdigest()


class ExactVectorIndex:
    """Small-corpus exact index with allowlisted search and atomic persistence."""

    FORMAT_VERSION = 1

    def __init__(self, spec: EmbeddingSpec):
        spec.validate()
        self.spec = spec
        self._vectors: Dict[int, List[float]] = {}

    def add(self, memory_id: int, vector: Sequence[float]) -> None:
        self._vectors[int(memory_id)] = _validate_vector(vector, self.spec.dimensions)

    def remove(self, memory_ids: Sequence[int]) -> None:
        for memory_id in memory_ids:
            self._vectors.pop(int(memory_id), None)

    def search(
        self,
        vector: Sequence[float],
        *,
        allowed_ids: Sequence[int],
        limit: int,
        min_score: float = -1.0,
    ) -> List[Dict[str, Any]]:
        query = _validate_vector(vector, self.spec.dimensions)
        allowed = {int(value) for value in allowed_ids}
        scored = [
            {"id": memory_id, "score": _cosine(query, candidate)}
            for memory_id, candidate in self._vectors.items()
            if memory_id in allowed
        ]
        return [
            item
            for item in sorted(scored, key=lambda item: (-item["score"], item["id"]))
            if item["score"] >= min_score
        ][: max(1, min(int(limit), 100))]

    def save(self, path: Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.FORMAT_VERSION,
            "model_name": self.spec.name,
            "model_hash": self.spec.model_hash,
            "dimensions": self.spec.dimensions,
            "vectors": {
                str(memory_id): _pack(vector).hex()
                for memory_id, vector in sorted(self._vectors.items())
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def load(cls, path: Path, expected_spec: EmbeddingSpec) -> "ExactVectorIndex":
        encoded = Path(path).read_bytes()
        payload = json.loads(encoded.decode("utf-8"))
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported vector index format")
        if (
            payload.get("model_hash") != expected_spec.model_hash
            or int(payload.get("dimensions", 0)) != expected_spec.dimensions
        ):
            raise ValueError("vector index model metadata mismatch")
        index = cls(expected_spec)
        for memory_id, encoded_vector in payload.get("vectors", {}).items():
            index.add(int(memory_id), _unpack(bytes.fromhex(encoded_vector), expected_spec.dimensions))
        return index


class SubprocessEmbeddingAdapter:
    """Hard-deadline adapter for an owner-approved local embedding command.

    The command receives one JSON object on stdin and must emit
    `{"vectors": [[...], ...]}` on stdout. It is started per request and is
    always waited for or killed, leaving no intentional idle worker.
    """

    def __init__(
        self,
        spec: EmbeddingSpec,
        command: Sequence[str],
        *,
        environment: Optional[Mapping[str, str]] = None,
    ):
        spec.validate()
        if not command:
            raise ValueError("embedding command is required")
        self.spec = spec
        self.command = [str(value) for value in command]
        self.environment = dict(environment or {})

    def encode(self, texts: Sequence[str], *, timeout_seconds: float) -> List[List[float]]:
        payload = json.dumps({"texts": [str(text) for text in texts]})
        env = {"PATH": os.environ.get("PATH", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
        env.update(self.environment)
        proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            stdout, _stderr = proc.communicate(payload, timeout=max(0.01, timeout_seconds))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise TimeoutError("embedding worker exceeded deadline")
        if proc.returncode != 0:
            raise RuntimeError("embedding worker failed")
        response = json.loads(stdout)
        vectors = response.get("vectors")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ValueError("embedding worker returned invalid batch shape")
        return [_validate_vector(vector, self.spec.dimensions) for vector in vectors]


class SQLiteExactSemanticBackend:
    def __init__(
        self,
        db: MemoryDB,
        adapter: EmbeddingAdapter,
        *,
        min_score: float = 0.55,
    ):
        adapter.spec.validate()
        self.db = db
        self.adapter = adapter
        self.store = SQLiteVectorStore(db, adapter.spec)
        self.min_score = float(min_score)

    def search(
        self,
        query: str,
        *,
        allowed_ids: Sequence[int],
        limit: int,
        timeout_seconds: float,
    ) -> List[Dict[str, Any]]:
        started = time.perf_counter()
        vectors = self.adapter.encode([query], timeout_seconds=timeout_seconds)
        remaining = timeout_seconds - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError("semantic retrieval deadline exhausted")
        index = ExactVectorIndex(self.adapter.spec)
        for memory_id, vector in self.store.load(allowed_ids).items():
            index.add(memory_id, vector)
        return index.search(
            vectors[0],
            allowed_ids=allowed_ids,
            limit=limit,
            min_score=self.min_score,
        )


def rebuild_embeddings(
    db: MemoryDB,
    adapter: EmbeddingAdapter,
    *,
    timeout_seconds: float = 30.0,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Deterministically rebuild active vectors from canonical SQLite text."""
    rows = db.get_active_index_rows()
    store = SQLiteVectorStore(db, adapter.spec)
    started = time.perf_counter()
    written = 0
    for offset in range(0, len(rows), max(1, int(batch_size))):
        elapsed = time.perf_counter() - started
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            raise TimeoutError("embedding rebuild exceeded deadline")
        batch = rows[offset : offset + max(1, int(batch_size))]
        vectors = adapter.encode(
            [row["content"] for row in batch],
            timeout_seconds=remaining,
        )
        written += store.upsert(
            (row["id"], row["content_hash"], vector)
            for row, vector in zip(batch, vectors)
        )
    return {
        "rows": len(rows),
        "written": written,
        "model_name": adapter.spec.name,
        "model_hash": adapter.spec.model_hash,
        "dimensions": adapter.spec.dimensions,
        "index_hash": store.deterministic_hash(),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }
