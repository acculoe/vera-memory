# VERA Memory — Patch Notes

This file describes every change since the original public **V5.0.0** release.
Current version: **5.6.1**.

---

## V5.6.1 — Duplicate guard (current)

**The fix:** VERA's store-time duplicate check previously required both the
same content **and** the same subject attribution. Identical content saved
under two different principals (e.g. once by you, once by your agent) slipped
past the check and created a second active row — two copies competing for the
same recall slots in every query.

- The guard now dedupes on **content alone**: an exact-duplicate store returns
  the existing memory instead of inserting a new one.
- Attribution of a surviving duplicate stays with the first writer. A
  same-subject duplicate still gets the existing trust-upgrade behavior.
- The extractor path inherits the same guard (it falls through to the store
  path), so both automatic and explicit saves are covered.
- `deduplicate=False` still exists as an explicit opt-out for repair tooling.
- No schema change; this is a pure code fix.

---

## V5.6.0 — Recall triggers ("say it, see it")

- New **term ledger**: every stored memory indexes its entity-grade terms
  (capitalized spans, acronyms, quoted/backticked spans) with ±40 chars of
  context and the memory's original creation date.
- When you mention a term your agent has seen before, recall answers
  *"mentioned N times since DATE"* from two indexed lookups — no LLM call, no
  network, sub-millisecond.
- Query-stopword tuning so the trigger stays quiet on chit-chat.
- Ledger writes ride the same transaction as the store (fail-open: a ledger
  problem never loses a store) and are purged on withdraw/delete.

---

## V5.5.1 — Canonical content hashes

- `content_hash` had two incompatible formats in the wild (legacy 16-char
  digests vs canonical 64-char SHA-256). Exact-duplicate detection could not
  match legacy rows — roughly 40% of a typical corpus was invisible to it.
  Every row is now rehashed to the canonical digest, mirrored into the
  embedding store in the same transaction (zero re-embeddings required).
- Duplicate lineage: superseded rows get their `superseded_by` link records
  backfilled so replacement history is queryable.

---

## V5.5.0 — Classification judge (Jev), `vera_forget`, data hygiene

### The classification judge — you need Jev (or an equivalent open-weight judge)

VERA now runs a **pre-store classification gate**: before content enters your
durable store, a judge model decides *store or skip* and *which category*.
This is what keeps chit-chat, one-off command output, and raw transcripts out
of a memory database that is supposed to hold durable knowledge.

**The reference judge is [Jev](https://openrouter.ai/typesafe/jev-1.13)
(`typesafe/jev-1.13`), an open-weight judge model.** To enable the gate you
need **Jev, or any other open-weight AI of the same type as Jev**, exposed
behind one small CLI contract:

```bash
your-judge --decide --question "..." --options "a,b,c" --state "..."
# → JSON on stdout: {"results": [{"choice": "...", "gate": "PASS|FLAG|REVIEW",
#                                 "confidence": 0.0-1.0, ...}]}
```

Then point VERA at it:

```bash
export VERA_JUDGE_SCRIPT=/path/to/your/judge-cli
```

Design guarantees:

- **Fail-open.** No judge configured, judge slow, judge down, judge returns
  garbage → VERA silently falls back to the built-in local regex classifier.
  A store never fails because a judge call failed.
- **Explicit stores are never dropped.** When a human or agent deliberately
  calls `vera_store`, the judge may re-categorize and re-weight, but it cannot
  discard the memory. Only the *automatic* extraction path can be gated shut.
- **Cheap by construction.** At the reference judge's observed rates a call
  costs ~$0.00002 / ~400ms; content under 40 chars never calls out, and a
  per-session budget caps the tail.
- **Your key stays yours.** VERA ships with **no API keys and no credentials
  of any kind**. The judge CLI reads whatever key *you* configure, from *your*
  environment.

### `vera_forget` — a real withdrawal verb

Before 5.5, an agent could store but never forget. `vera_forget` sets a memory
to `withdrawn`: removed from every search index immediately, row kept in
SQLite with the reason and actor recorded for audit. Distinct from
`superseded` (a newer fact replaced it) and `historical` (aged out).

### Data hygiene migration (automatic, idempotent, audited)

On first start after upgrading, migration 9 runs against your existing DB:

- importance values clamped to 0..1 and rounded to 2dp (kills float slop)
- `category` → `memory_type` sync (fixes rows filed under the wrong type)
- `valid_from` backfilled from `created_at` (enables staleness/gap analysis)
- same-subject byte-identical duplicates superseded automatically
- cross-subject byte-identical duplicates **queued for review**, not merged —
  attribution across identities is an ownership decision, so VERA records the
  conflict for you instead of silently rewriting it

Every step is written to the `migration_audit` table. Content text is never
rewritten.

### Also in 5.5.x

- Persistence file (`persistence.md`) cap raised to 2,500 with sentence-aware
  truncation and placeholder filtering.
- Session briefing: fixed Active-Task section lookup.
- Dictation cleanup: mechanical speech-to-text artifacts (doubled words,
  split words) are collapsed on store — meaning is never reworded.

---

## Upgrading from V5.0.0 (the original public release)

VERA upgrades **in place** — same plugin, same provider, no config change:

```bash
cd /path/to/vera-memory && git pull
cp -r plugin/* ~/.hermes/plugins/vera_memory/
find ~/.hermes/plugins/vera_memory -name '__pycache__' -type d -exec rm -rf {} +
# BACK UP your database first:
cp ~/.hermes/profiles/<agent>/vera-memory/memory.db{,.pre-561.bak}
# restart the gateway — migrations 9/10/11 run automatically at first start
```

Verify after restart: `plugin.yaml` shows `version: 5.6.1`, the gateway log
shows the migration lines, and `PRAGMA integrity_check` on your DB returns
`ok`.

Rollback: restore the `.pre-561.bak` file and the previous plugin files.

---

## File integrity note

The bundled tokenizer files (`vocab.txt`, `tokenizer.json`) are the standard
[all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) vocabulary
(Apache-2.0). Common English words appearing in it — including any that look
like project names — are part of that universal model vocabulary, not
project content.
