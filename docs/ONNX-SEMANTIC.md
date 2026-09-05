# Semantic Search + the ONNX Backfill Worker

VERA's retrieval is hybrid: SQLite FTS5 (lexical) is authoritative and always on; the ONNX semantic layer adds paraphrase matching on top. No model bundled, no cloud calls, no daemon.

## How it works

- Embedding model: **all-MiniLM-L6-v2, INT8-quantized ONNX** (~23 MB), 384 dimensions, 256-token max sequence.
- Runtime: `onnxruntime` on CPU (`CPUExecutionProvider`) — runs on a Raspberry Pi-class machine.
- Vectors are stored in SQLite (`memory_embeddings` table), keyed by a pinned model hash (`xenova-minilm-l6-v2-int8-v1`). Changing models re-embeds cleanly; old vectors are never silently mixed.
- SQLite decides **which** memory IDs may participate (authority); vectors are disposable derivatives that can be rebuilt at any time.

## Graceful degradation

No model file → the provider logs `Semantic backend unavailable: ... (lexical-only mode)` and continues. FTS5 search, extraction, briefing, persistence, corrections — everything except paraphrase matching — works without the model. The model file itself is also never downloaded automatically; you fetch it deliberately:

```bash
cd ~/.hermes/plugins/vera_memory/models
curl -L -o model_int8.onnx \
  "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_int8.onnx"
```

Tokenizer assets (`tokenizer.json`, `vocab.txt`) ship in the repo — no Hugging Face login needed for those.

## The backfill worker (`vera-onnx-backfill.py`)

New memories get embedded automatically as they're stored. The backfill worker embeds everything that predates the model or missed its turn (e.g. you installed the model later):

```bash
python3 ~/.hermes/plugins/vera_memory/vera-onnx-backfill.py [path/to/memory.db]
```

- **Path discovery:** tries `$HERMES_HOME/vera-memory/memory.db`, then `profiles/default/vera-memory/memory.db`, then `~/.hermes/profiles/default/vera-memory/memory.db`. For a named profile or a custom location, pass the path explicitly:

  ```bash
  python3 vera-onnx-backfill.py ~/.hermes/profiles/<agent-name>/vera-memory/memory.db
  ```

- **Idempotent:** memories that already have embeddings for this model hash are skipped. Safe to re-run any time.
- **Output:** progress per batch (64 memories), final report of embedded/failed counts. Failures are listed loudly, never swallowed.
- **Dependencies:** only `onnxruntime` + `numpy`.

## Embedding self-heal (v5)

At every startup the provider counts memories vs. available vectors. If coverage has drifted (failures, crash mid-write), it re-embeds the gap through the same worker path, bounded so startup stays responsive. Residual failures are logged at WARNING — they never silently grow again. If you see repeated heal warnings at startup, fix the root cause (usually: model file missing/corrupt) rather than ignoring the log.

## Verifying semantic search

```bash
# In a live session:
# 1.  vera_store(content="The quarterly report ships on the fifteenth", category="facts")
# 2.  vera_search(query="when is the report due")     <- different words entirely
# A semantic hit proves the full pipeline: tokenizer -> ONNX -> vectors -> cosine ranking.
```

Zero hits on paraphrase queries with an empty log → check `lexical-only mode` in the gateway log, then re-check the model file.

## Performance notes (measured)

- Model load: ~1–2 s once per process (lazy — first encode call).
- Encoding: hundreds of memories/minute on a laptop CPU (INT8). The backfill worker batches at 64.
- Query cost: one tokenizer pass + one ONNX inference (~10 ms class) + cosine over candidate IDs. Effectively free next to the model call itself.
- Storage: 384 float32 per memory ≈ 1.5 KB — a five-figure memory count costs tens of MB.
