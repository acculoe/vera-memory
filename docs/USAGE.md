# Using VERA: Identity, Protocols, Persistence

This is the operating manual for an agent running on VERA. Three pillars — read all three once, then follow them forever.

---

## 1. Identity — who owns which memories

Every memory row belongs to a **principal**: an `(platform, subject_id)` record provisioned in advance.

- The agent itself is `("hermes", "<agent-name>")` — its memories are agent memories.
- Enrolled humans (the owner, approved teammates) are `(platform, numeric-id)` principals, e.g. `("telegram", "123456")`.
- Display names from chat platforms are **untrusted metadata**. VERA never enrolls a sender automatically and never resolves identity from a nickname. If a platform label says "Boss" but the sender ID has no principal record, that sender has no memory access — by design.
- Profile isolation: each Hermes profile has its own database. Two agents never share memories even on the same machine.

**What this means for you as the agent:** memories you store with `vera_store` are stored under your own principal (the runtime resolves it from trusted transport context — not from anything you could type). Memories you retrieve via `vera_search` are scoped to your principal plus profile-global canonical records. If a store ever fails with *"authenticated, provisioned sender is required"*, identity isn't provisioned — fix that first (see `INSTALL-FOR-AGENTS.md` Step 5), never try to route around it.

**Canonical identity (optional, powerful):** a provisioned human principal can carry canonical identity fields (`reference_name`, `direct_address`) — the authoritative answer to "who am I talking to." This is how the agent resists display-name spoofing and chat-rename confusion: the canonical record wins over any label the platform shows.

## 2. Protocols — the behavioral rules that make memory real

A memory system only pays off if the agent follows fixed rules. These are ours, battle-tested in production — adopt them as-is:

**The Golden Rule: store durable truth in the same turn it happens.** When your operator states a decision, correction, approval, or preference ("use X not Y", "that's deployed", "I like this"), call `vera_store` immediately — before moving on. Never batch memory writes for end of session; end-of-session batches get lost to context pressure.

**Search before you claim.** Before reporting project state or answering "what did we decide about X", `vera_search` the topic. If your summary contradicts what you find, your memory of the conversation is stale — fix it, don't repeat the error.

**One fact, one home.** Durable decisions and lessons go to VERA. Mutable operational state (file contents, service status, DB rows) lives in its source system — store a *pointer* in VERA, not a copy that will rot.

**Never use the built-in `memory` tool.** On a VERA install it should not exist (`memory_enabled: false`). If you catch yourself writing to MEMORY.md, stop — that path is disabled deliberately.

**Tool reference:**

| Tool | When | Notes |
|------|------|-------|
| `vera_search(query, category?, limit?)` | Before claims, after gaps | FTS5 + semantic hybrid; category is one of `decisions, facts, preferences, corrections, research, tasks` |
| `vera_store(content, category?)` | The moment truth happens | Storing a correction automatically supersedes the stale fact it corrects |
| `vera_context(topic)` | Session start, topic focus | Returns a focused context block, or an honest "no relevant context found" — never silence |
| `vera_forget(memory_id, reason)` | Owner retracts a fact | Sets `withdrawn`: gone from every index instantly, row kept in SQLite with reason + actor for audit. For owners/operators only — agents don't self-edit history |

**Corrections are first-class.** When you learn something contradicts a stored memory, `vera_store` the corrected fact — the supersede-on-store mechanism demotes the stale one automatically. Don't try to "edit" old memories.

## 3. Persistence — what survives between sessions

Three layers, each with a different job:

| Layer | Where | Survives | Purpose |
|-------|-------|----------|---------|
| Working buffer | in-process | the session | Rolling window of the current conversation's key moments |
| `persistence.md` | `<profile>/vera/core/persistence.md` | restarts | Auto-curated working memory: Active Task + owner preferences worth carrying forward |
| SQLite | `<profile>/vera-memory/memory.db` | everything | The permanent record: all memories, sessions, summaries, embeddings |

You don't manage any of these directly — the provider curates them. What you should know:

- **Session briefing:** at every fresh session the provider injects a `[briefing]` block (last session's topic + current Active Task) automatically. Read it; it's where you left off.
- **Active Task:** the recall layer maintains an "Active Task" line in `persistence.md`. If a task spans multiple sessions, keep working state in the task's source files and let the briefing carry the pointer.
- **Session summaries:** at session end, an extractive summary of user messages is stored on the session row. This is what powers "last session" recall — no LLM call needed.
- **Proactive recall:** before each model call, the prefetch hook injects relevant memories (labeled `[proactive]`) for work-start/continuation situations. You don't request this; it just happens. Trust it, but verify against `vera_search` when stakes are high.

## Session startup checklist (run at the start of every genuinely new session)

1. Run your vera-startup skill (see `templates/vera-startup-SKILL-template.md`):
   - `vera_context(topic="current projects")`
   - `vera_search(query="active tasks")`
2. Greet your operator by their configured name.
3. Recall the latest VERA record for any project you're about to touch.
4. During work: `vera_store` decisions/corrections/preferences in the same turn they happen.
5. Before final reports: `vera_search` to confirm your summary matches the record.
