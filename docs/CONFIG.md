# Hermes Configuration Guide

Everything VERA needs from Hermes, verified against production deployments. This is the *exact* working configuration — no guesswork.

## The three keys that matter

In the profile's `config.yaml`:

```yaml
memory:
  memory_enabled: false          # 1. disables the built-in `memory` tool
  user_profile_enabled: false    # 2. disables built-in user-profile injection
  provider: vera_memory          # 3. activates VERA as the exclusive provider
```

Why each one:

| Key | If you skip it |
|-----|----------------|
| `memory_enabled: false` | The built-in `memory` tool stays in the agent's toolset. The framework injects `MEMORY.md` into every turn and it **overrides VERA** — the agent writes to MEMORY.md instead of VERA, and your SQLite memory quietly starves. This is the #1 misconfiguration. |
| `user_profile_enabled: false` | The built-in user-profile system fights VERA over "who the user is." VERA's principal model handles identity with proper isolation. Running both means two conflicting sources of truth. |
| `provider: vera_memory` | This is the **only** activation switch. `plugins.enabled` is neither required nor sufficient for an exclusive provider — it may show as "not enabled" in `hermes plugins list`; that is normal. The authority is the provider selector. |

After editing: **restart the gateway** (config is read at startup), then run `/new` in the chat so the agent gets a fresh session with the new memory system.

## Killing the MEMORY.md conflict completely

Three layers, all three should be applied:

1. `memory_enabled: false` (above) — removes the tool.
2. Remove `- memory` from every list under `platform_toolsets:` (e.g. `platform_toolsets.telegram:`) so it never loads on any platform. Do **not** touch `known_builtin_toolsets` — that is Hermes' registry of what *can* exist, not what loads.
3. Point `memories/MEMORY.md` at VERA (backup first):

```bash
MEMDIR="$HOME/.hermes/memories"    # profile-scoped installs: "$HERMES_HOME/memories"
[ -f "$MEMDIR/MEMORY.md" ] && cp "$MEMDIR/MEMORY.md" "$MEMDIR/MEMORY.md.pre-vera.bak"
printf '%s\n' '> VERA is the single source of truth for memory. Use `vera_search` to retrieve and `vera_store` to save. Do NOT use the `memory` tool.' > "$MEMDIR/MEMORY.md"
```

The one-liner matters: even with the tool disabled, a stale MEMORY.md full of old notes keeps injecting pre-VERA context into every turn.

## Verification

After restart + `/new`:

```bash
# 1. Plugin discovered
hermes plugins list            # vera_memory should appear (as exclusive provider)

# 2. Live tools — in a chat with the agent:
#    ask it to call  vera_search(query="test")  →  should respond, even with "no results"
#    ask it to call  vera_store(content="config verification", category="facts")
#    then            vera_search(query="config verification")  →  finds it

# 3. Database exists at the profile path
ls -la ~/.hermes/profiles/<agent-name>/vera-memory/memory.db

# 4. Gateway log shows provider init
grep -i "VERA" ~/.hermes/logs/*.log | tail -5
```

If step 2 fails with *"authenticated, provisioned sender is required"* — the identity principal isn't provisioned yet. See Step 5 of `INSTALL-FOR-AGENTS.md`.

## Multi-profile notes (multiple agents)

- Each profile gets its **own** database: `~/.hermes/profiles/<name>/vera-memory/memory.db`. Never point two profiles at one DB file — VERA enforces per-principal isolation and shared files break it.
- Provision each profile's agent principal separately (`("hermes", "<profile-name>")` per profile).
- The plugin auto-detects profile-scoped `HERMES_HOME` — no path config needed. It also handles Windows-style paths (backslash normalization is built in).
- A profile with legacy `vera/memory/*.md` files will **refuse to start** VERA until an explicit offline migration is done (by design — startup never mutates your history as a side effect).

## Optional: the classification judge (Jev or a compatible open-weight model)

Since 5.5, VERA can gate the automatic extraction path with an LLM judge that
decides *store-or-skip* and *category* before content enters the durable
store. **The reference judge is [Jev](https://openrouter.ai/typesafe/jev-1.13)
(`typesafe/jev-1.13`); any other open-weight AI of the same type works** as
long as it is exposed behind the same CLI contract:

```bash
# your-judge --decide --question "..." --options "a,b,c" --state "..."
# prints JSON: {"results": [{"choice": "...", "gate": "...", "confidence": 0.0-1.0}]}
export VERA_JUDGE_SCRIPT=/path/to/your/judge-cli     # add to the profile's environment
```

Behavior:

- **Not set / judge fails / times out / returns garbage** → VERA silently uses
  the built-in local regex classifier. Nothing breaks; no store ever fails
  because a judge call failed.
- Explicit `vera_store` calls are **never dropped** by the judge — it may
  re-categorize and re-weight, not discard. Only the automatic extraction
  path can be gated shut.
- The judge CLI is responsible for its own credentials. **VERA ships with no
  API keys** and never writes credentials anywhere.

## Reference: a production memory block

From a production profile (the `memory:` section):

```yaml
memory:
  memory_enabled: false
  user_profile_enabled: false
  provider: vera_memory
```

That's the whole contract. VERA adds no other required config keys; tuning knobs (`extraction_enabled`, `max_prefetch_chars`, `working_buffer_size`) are passed by Hermes defaults and don't need to be set.
