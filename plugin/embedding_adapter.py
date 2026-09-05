"""ONNX quantized embedding adapter for VERA Memory V4.

Uses all-MiniLM-L6-v2 INT8 quantized (23MB) for local semantic search.
No internet required after initial model download.
No external services. No daemons. Self-contained per agent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Model configuration
MODEL_DIR = Path(__file__).parent / "models"
MODEL_FILE = "model_int8.onnx"
TOKENIZER_FILE = "tokenizer.json"
MODEL_NAME = "all-MiniLM-L6-v2-int8"
MODELDimensions = 384
MAX_SEQUENCE_LENGTH = 256
MODEL_HASH = "xenova-minilm-l6-v2-int8-v1"  # Pinned hash for reproducibility


def _get_model_path() -> Path:
    """Resolve the ONNX model path."""
    path = MODEL_DIR / MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {path}. "
            f"Download from: https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_int8.onnx"
        )
    return path


def _get_tokenizer_path() -> Path:
    """Resolve the tokenizer path."""
    path = MODEL_DIR / TOKENIZER_FILE
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {path}")
    return path


class WordPieceTokenizer:
    """Simple WordPiece tokenizer for embedding inference."""

    def __init__(self, tokenizer_path: Path):
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tok_config = json.load(f)

        self.vocab: Dict[str, int] = tok_config["model"]["vocab"]
        self.ids_to_tokens: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.max_length = MAX_SEQUENCE_LENGTH

        # Special tokens
        self.cls_token_id = self.vocab.get("[CLS]", 101)
        self.sep_token_id = self.vocab.get("[SEP]", 102)
        self.pad_token_id = self.vocab.get("[PAD]", 0)
        self.unk_token_id = self.vocab.get("[UNK]", 100)

    def tokenize(self, text: str) -> Dict[str, List[int]]:
        """Tokenize text and return input_ids, attention_mask, token_type_ids."""
        # Basic whitespace + punctuation tokenization
        tokens = self._basic_tokenize(text)
        # WordPiece subword tokenization
        token_ids = []
        for token in tokens[: self.max_length - 2]:  # Reserve for CLS + SEP
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # WordPiece: split into subwords
                subword_ids = self._wordpiece(token)
                token_ids.extend(subword_ids)

        # Add special tokens
        input_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # Pad to max_length
        pad_len = self.max_length - len(input_ids)
        input_ids += [self.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        token_type_ids += [0] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic tokenization: lowercase, split on whitespace/punctuation."""
        import re
        text = text.lower().strip()
        # Split on whitespace and punctuation, keep the tokens
        tokens = re.findall(r"[a-z0-9]+|[^\s\w]", text)
        return tokens

    def _wordpiece(self, token: str) -> List[int]:
        """WordPiece subword tokenization."""
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


class ONNXEmbeddingAdapter:
    """ONNX Runtime embedding adapter implementing the EmbeddingAdapter protocol.

    Loads all-MiniLM-L6-v2 INT8 quantized model (~23MB) on first use.
    Runs entirely local — no internet, no API calls, no daemons.
    """

    def __init__(self):
        self._session = None
        self._tokenizer = None
        self._loaded = False
        self._load_lock = None  # Lazy-init to avoid import-time issues

    def _ensure_loaded(self):
        """Lazy-load model and tokenizer on first encode() call."""
        if self._loaded:
            return

        import threading
        if self._load_lock is None:
            self._load_lock = threading.Lock()

        with self._load_lock:
            if self._loaded:  # Double-check after acquiring lock
                return

            try:
                import onnxruntime as ort
            except ImportError:
                raise RuntimeError(
                    "onnxruntime is required for VERA V4 semantic search. "
                    "Install with: pip install onnxruntime"
                )

            model_path = _get_model_path()
            tokenizer_path = _get_tokenizer_path()

            logger.info("VERA V4: Loading ONNX embedding model from %s", model_path)

            # Load ONNX session with optimization
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1

            self._session = ort.InferenceSession(
                str(model_path),
                sess_options,
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = WordPieceTokenizer(tokenizer_path)
            self._loaded = True

            logger.info(
                "VERA V4: Embedding model loaded (%s, %dMB)",
                MODEL_NAME,
                os.path.getsize(model_path) // (1024 * 1024),
            )

    @property
    def spec(self):
        """Return the EmbeddingSpec for this adapter."""
        if __package__:
            from .semantic import EmbeddingSpec
        else:
            from semantic import EmbeddingSpec
        return EmbeddingSpec(
            name=MODEL_NAME,
            model_hash=MODEL_HASH,
            dimensions=MODELDimensions,
        )

    def encode(self, texts: Sequence[str], *, timeout_seconds: float = 5.0) -> List[List[float]]:
        """Encode texts into embedding vectors.

        Args:
            texts: Sequence of text strings to encode.
            timeout_seconds: Maximum time for encoding.

        Returns:
            List of vectors, one per input text.
        """
        self._ensure_loaded()

        if not texts:
            return []

        started = time.perf_counter()
        vectors = []

        for text in texts:
            elapsed = time.perf_counter() - started
            if elapsed > timeout_seconds:
                raise TimeoutError(f"Embedding encoding exceeded {timeout_seconds}s deadline")

            vector = self._encode_single(str(text))
            vectors.append(vector)

        return vectors

    def _encode_single(self, text: str) -> List[float]:
        """Encode a single text into an embedding vector."""
        import numpy as np

        # Tokenize
        tokens = self._tokenizer.tokenize(text)
        input_ids = np.array([tokens["input_ids"]], dtype=np.int64)
        attention_mask = np.array([tokens["attention_mask"]], dtype=np.int64)
        token_type_ids = np.array([tokens["token_type_ids"]], dtype=np.int64)

        # Run inference
        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

        # Get last_hidden_state: shape (1, seq_len, 384)
        last_hidden = outputs[0]

        # Mean pooling: average over non-padding tokens
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(last_hidden * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = sum_embeddings / sum_mask

        # L2 normalize
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        normalized = pooled / np.clip(norm, a_min=1e-9, a_max=None)

        return normalized[0].tolist()
