# VERA Memory V5

**Persistent, governed memory for Hermes Agent profiles.**

**Current release: V5.6.1** — see [PATCH-NOTES.md](PATCH-NOTES.md) for every
change since V5.0.0 (duplicate guard, recall triggers, canonical hashing,
classification judge, `vera_forget`, data hygiene).

VERA (Verified Episodic Registry Architecture) replaces the default MEMORY.md
memory system with a per-agent SQLite database: full-text search (FTS5), optional
fully-local semantic search (ONNX embeddings), automatic fact extraction from
every conversation, session-start briefing, working-memory persistence, and
correction supersession. No cloud. No daemons. No external services. Your
agent's memory stays on your machine.

Built by [Acculoe FutureWorks](https://github.com/acculoe). This
repository is the standalone, genericized public release — none of our internal
persona, branding, or data is included.

---

## Features

- **SQLite + FTS5 storage** — one database per agent profile, zero server
- **Automatic memory extraction** — facts, decisions, preferences, and corrections harvested from every turn
- **Governed identity** — every memory belongs to a *principal* (the agent itself, or an enrolled human sender). Memories are never served to the wrong identity.
- **Three agent tools** — `vera_search` (retrieve), `vera_store` (save), `vera_context` (focused context block)
- **Proactive recall** — relevant memories injected automatically before each model call
- **Session briefing** — a "here's where you left off" block at every session start
- **Persistent working memory** — a curated `persistence.md` that survives restarts
- **Correction supersession** — storing a correction automatically demotes the stale fact it corrects
- **Optional classification judge** — before content enters the durable store, a judge model decides store-vs-skip and category. Reference judge: **Jev** (`typesafe/jev-1.13`), or any compatible open-weight judge via `VERA_JUDGE_SCRIPT`. Fail-open to a local regex classifier when no judge is configured
- **`vera_forget`** — a real withdrawal verb: retract a fact, it leaves every index, the audited row stays
- **Duplicate guard** — exact-duplicate stores (same content, any subject) return the existing memory; no twin rows competing for recall slots
- **Term-ledger recall triggers** — "mentioned N times since DATE" answers from indexed lookups, no LLM call
- **Optional local semantic search** — all-MiniLM-L6-v2 INT8 (23 MB) runs 100% local via ONNX; graceful lexical-only fallback if the model is absent
- **Self-healing embeddings** — startup coverage check rebuilds missing vectors automatically

## Requirements

- [Hermes Agent](https://hermes-agent.nousresearch.com/docs) — macOS, Linux, or Windows
- Python 3.10+
- `onnxruntime` and `numpy` (see `requirements.txt`) — for semantic search

## Install (human, ~5 minutes)

Full guide: [`docs/CONFIG.md`](docs/CONFIG.md). Short version:

```bash
git clone https://github.com/acculoe/vera-memory.git
mkdir -p ~/.hermes/plugins/vera_memory
cp -r vera-memory/plugin/* ~/.hermes/plugins/vera_memory/
pip install -r vera-memory/requirements.txt
```

Add to your profile's `config.yaml`:

```yaml
memory:
  memory_enabled: false
  user_profile_enabled: false
  provider: vera_memory
```

Download the embedding model (optional, but recommended):

```bash
cd ~/.hermes/plugins/vera_memory/models
curl -L -o model_int8.onnx \
  "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_int8.onnx"
```

Restart the Hermes gateway. Done — VERA is now the exclusive memory provider.

## Install (your AI agent, hands-free)

Point your Hermes agent at **[`INSTALL-FOR-AGENTS.md`](INSTALL-FOR-AGENTS.md)**.
It is written for an LLM agent to follow without human help: environment
survey, install, config, identity provisioning, a full verification battery,
and a build-your-own startup-skill step so the agent reliably uses VERA at the
start of every session.

## Documentation

| File | Contents |
|------|----------|
| [`PATCH-NOTES.md`](PATCH-NOTES.md) | **What changed in 5.5.x / 5.6.x and how to upgrade** |
| [`INSTALL-FOR-AGENTS.md`](INSTALL-FOR-AGENTS.md) | Step-by-step install script for an AI agent to execute |
| [`docs/CONFIG.md`](docs/CONFIG.md) | Hermes config changes + eliminating the MEMORY.md conflict |
| [`docs/USAGE.md`](docs/USAGE.md) | The agent protocol: identity, tools, persistence, session startup |
| [`docs/ONNX-SEMANTIC.md`](docs/ONNX-SEMANTIC.md) | Semantic search setup + the embedding backfill worker |
| [`templates/`](templates/) | Starter `identity.md`, `protocols.md`, and a vera-startup skill template |

## Security model (read this once)

- **Principals.** Every memory row belongs to a principal: `(platform, subject_id)`
  — for example `("hermes", "myagent")` for the agent itself, or
  `("telegram", "123456")` for an enrolled human.
- **Untrusted labels.** Display names from chat platforms are treated as
  untrusted metadata. Senders are resolved only against pre-registered
  principal records. Unknown senders are never enrolled automatically.
- **Fail closed.** Retrieval and storage require a resolved, provisioned
  principal. If identity cannot be established, the tools refuse rather than
  guess.
- **Profile isolation.** Each Hermes profile gets its own database. Agents
  cannot read each other's memories.
- **Owner-controlled enrollment.** Human principals are provisioned explicitly
  by you (one Python line, shown in `docs/CONFIG.md`), never by the agent
  deciding someone is trusted.

## Migrating from the old file-based VERA?

VERA was originally a set of Markdown files (`vera/memory/*.md`). The SQLite
plugin does **not** import those files automatically — startup refuses to touch
a profile with legacy files until you run an explicit, backed-up offline
migration. See `migrator.py` and `safety.py` for the migration tooling, and
`docs/CONFIG.md` for the warning you'll see if legacy files are present.

If you previously used `vera-memory-system` (the file-based version): this
plugin supersedes it. You do not need the old file structure.

## License

MIT — see [LICENSE](LICENSE).

## Credits

- **Acculoe FutureWorks** — architecture, engineering, and field testing
- Built for [Hermes Agent](https://hermes-agent.nousresearch.com/docs) by Nous Research
- Embedding model: [all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) (Apache-2.0), INT8 ONNX build by [Xenova](https://huggingface.co/Xenova)
