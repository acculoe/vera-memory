#!/usr/bin/env python3
"""VERA V4 ONNX Embedding Backfill Script

Embeds all existing memories that lack vectors.
Cross-platform: works on Linux and Windows.
No external dependencies beyond onnxruntime + numpy (already required by VERA).

Usage:
    python3 vera-onnx-backfill.py [path/to/memory.db]

If no path given, discovers DB via HERMES_HOME env or defaults to
~/.hermes/profiles/default/vera-memory/memory.db

Safe to re-run: skips memories that already have embeddings for this model.
"""

from __future__ import annotations

import os
import sys
import time
import struct
import hashlib
import math
import sqlite3
import json
from pathlib import Path
from typing import List, Sequence, Tuple

# ── Config ──────────────────────────────────────────────────────────────
MODEL_DIR_NAME = "models"
MODEL_FILE = "model_int8.onnx"
TOKENIZER_FILE = "tokenizer.json"
MODEL_NAME = "all-MiniLM-L6-v2-int8"
MODEL_HASH = "xenova-minilm-l6-v2-int8-v1"
MODEL_DIMENSIONS = 384
MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 64  # memories per encoding batch


# ── Vector packing (must match semantic.py) ─────────────────────────────
def _pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _validate_vector(vector: Sequence[float], dimensions: int) -> List[float]:
    output = [float(v) for v in vector]
    if len(output) != dimensions:
        raise ValueError(f"expected {dimensions} dims, got {len(output)}")
    if not all(math.isfinite(v) for v in output):
        raise ValueError("non-finite values in embedding")
    return output


# ── Tokenizer (reused from embedding_adapter.py) ───────────────────────
class WordPieceTokenizer:
    def __init__(self, tokenizer_path: Path):
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tok_config = json.load(f)
        self.vocab: dict = tok_config["model"]["vocab"]
        self.max_length = MAX_SEQUENCE_LENGTH
        self.cls_token_id = self.vocab.get("[CLS]", 101)
        self.sep_token_id = self.vocab.get("[SEP]", 102)
        self.pad_token_id = self.vocab.get("[PAD]", 0)
        self.unk_token_id = self.vocab.get("[UNK]", 100)

    def tokenize(self, text: str) -> dict:
        import re
        text = text.lower().strip()
        tokens = re.findall(r"[a-z0-9]+|[^\s\w]", text)
        token_ids = []
        for token in tokens[: self.max_length - 2]:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.extend(self._wordpiece(token))
        input_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        pad_len = self.max_length - len(input_ids)
        input_ids += [self.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        token_type_ids += [0] * pad_len
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def _wordpiece(self, token: str) -> List[int]:
        ids = []
        start = 0
        while start < len(token):
            end = len(token)
            found = False
            while start < end:
                substr = token[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    ids.append(self.vocab[substr])
                    found = True
                    break
                end -= 1
            if not found:
                ids.append(self.unk_token_id)
                start += 1
            else:
                start = end
        return ids if ids else [self.unk_token_id]


# ── ONNX Encoder ────────────────────────────────────────────────────────
class ONNXEncoder:
    def __init__(self, plugin_dir: Path):
        model_path = plugin_dir / MODEL_DIR_NAME / MODEL_FILE
        tokenizer_path = plugin_dir / MODEL_DIR_NAME / TOKENIZER_FILE

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        import onnxruntime as ort
        import numpy as np
        self.np = np

        print(f"Loading ONNX model from {model_path}...")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = WordPieceTokenizer(tokenizer_path)
        print(f"Model loaded ({model_path.stat().st_size // (1024*1024)}MB)")

    def encode(self, texts: List[str]) -> List[List[float]]:
        np = self.np
        vectors = []
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            input_ids = np.array([tokens["input_ids"]], dtype=np.int64)
            attention_mask = np.array([tokens["attention_mask"]], dtype=np.int64)
            token_type_ids = np.array([tokens["token_type_ids"]], dtype=np.int64)

            outputs = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )
            last_hidden = outputs[0]
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            sum_embeddings = np.sum(last_hidden * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            pooled = sum_embeddings / sum_mask
            norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            normalized = pooled / np.clip(norm, a_min=1e-9, a_max=None)
            vectors.append(normalized[0].tolist())
        return vectors


# ── Backfill ────────────────────────────────────────────────────────────
def discover_db() -> Path:
    """Find memory.db from env or defaults."""
    hermes_home = os.environ.get("HERMES_HOME", "")
    if hermes_home:
        p = Path(hermes_home)
        # Handle profile-scoped HERMES_HOME
        db = p / "vera-memory" / "memory.db"
        if db.exists():
            return db
        db = p / "profiles" / "default" / "vera-memory" / "memory.db"
        if db.exists():
            return db
    default = Path.home() / ".hermes" / "profiles" / "default" / "vera-memory" / "memory.db"
    if default.exists():
        return default
    raise FileNotFoundError(
        "Cannot find memory.db. Set HERMES_HOME or pass path as argument."
    )


def discover_plugin_dir() -> Path:
    """Find the vera memory plugin directory."""
    hermes_home = os.environ.get("HERMES_HOME", "")
    candidates = []
    if hermes_home:
        p = Path(hermes_home)
        candidates.append(p / "plugins" / "vera_memory")
        candidates.append(p / "profiles" / "default" / "plugins" / "vera_memory")
    home = Path.home() / ".hermes"
    candidates.append(home / "plugins" / "vera_memory")
    candidates.append(home / "profiles" / "default" / "plugins" / "vera_memory")
    for c in candidates:
        if c.exists() and (c / "embedding_adapter.py").exists():
            return c
    raise FileNotFoundError("Cannot find vera_memory plugin directory.")


def get_embeddable_memories(conn: sqlite3.Connection) -> List[Tuple[int, str, str]]:
    """Get memories that need embedding (no existing embedding for this model)."""
    rows = conn.execute(
        """
        SELECT m.id, m.content, m.content_hash
        FROM memories m
        WHERE m.status = 'active'
          AND m.retrieval_enabled = 1
          AND m.id NOT IN (
              SELECT e.memory_id FROM memory_embeddings e
              WHERE e.model_hash = ?
          )
        ORDER BY m.id
        """,
        (MODEL_HASH,),
    ).fetchall()
    return rows


def backfill(db_path: Path, plugin_dir: Path):
    print(f"Database: {db_path}")
    print(f"Plugin:   {plugin_dir}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Count totals
    total_memories = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE status='active' AND retrieval_enabled=1"
    ).fetchone()[0]
    already_embedded = conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings WHERE model_hash=?", (MODEL_HASH,)
    ).fetchone()[0]
    to_embed = get_embeddable_memories(conn)

    print(f"\n--- Backfill Status ---")
    print(f"Total active memories:  {total_memories}")
    print(f"Already embedded:       {already_embedded}")
    print(f"Need embedding:         {len(to_embed)}")

    if not to_embed:
        print("\nNothing to do — all memories already embedded.")
        conn.close()
        return

    # Load encoder
    encoder = ONNXEncoder(plugin_dir)

    # Batch process
    start_time = time.time()
    embedded_count = 0
    errors = 0

    for i in range(0, len(to_embed), BATCH_SIZE):
        batch = to_embed[i : i + BATCH_SIZE]
        texts = [row[1] for row in batch]  # content
        ids = [row[0] for row in batch]
        hashes = [row[2] for row in batch]

        try:
            vectors = encoder.encode(texts)
            # Upsert
            prepared = []
            for mid, chash, vec in zip(ids, hashes, vectors):
                clean = _validate_vector(vec, MODEL_DIMENSIONS)
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                prepared.append((
                    int(mid),
                    MODEL_NAME,
                    MODEL_HASH,
                    MODEL_DIMENSIONS,
                    _pack(clean),
                    str(chash),
                    now,
                ))
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
            # Update embedding_status on memories
            conn.executemany(
                "UPDATE memories SET embedding_status='ready', embedding_model=? WHERE id=?",
                [(MODEL_HASH, mid) for mid in ids],
            )
            conn.commit()
            embedded_count += len(batch)
            elapsed = time.time() - start_time
            rate = embedded_count / elapsed if elapsed > 0 else 0
            print(
                f"  [{embedded_count}/{len(to_embed)}] "
                f"embedded, {rate:.0f} mem/s, "
                f"{elapsed:.1f}s elapsed"
            )
        except Exception as e:
            errors += 1
            print(f"  ERROR in batch starting at id {ids[0]}: {e}")
            conn.rollback()

    # Final verification
    final_count = conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings WHERE model_hash=?", (MODEL_HASH,)
    ).fetchone()[0]
    elapsed = time.time() - start_time

    print(f"\n--- Backfill Complete ---")
    print(f"Embedded:     {embedded_count} memories")
    print(f"Errors:       {errors} batches")
    print(f"Time:         {elapsed:.1f}s")
    print(f"Total in DB:  {final_count} embeddings (model: {MODEL_NAME})")
    print(f"Total active: {total_memories} memories")

    if final_count == total_memories:
        print("\n✓ All memories embedded — semantic search fully operational.")
    else:
        print(f"\n⚠ Mismatch: {total_memories - final_count} memories still lack embeddings.")

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])
    else:
        db_path = discover_db()
    plugin_dir = discover_plugin_dir()
    backfill(db_path, plugin_dir)
