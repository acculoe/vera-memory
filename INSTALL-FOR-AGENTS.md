# VERA Memory V5 — Agent Self-Install Guide

**You are an AI agent (Claude, Codex, a Hermes profile, or any LLM with shell access).** Your human operator wants you to install VERA Memory on this machine and use it as your permanent memory from now on. Follow every step in order. Do not skip verification probes. If a probe fails, stop and report the exact error — do not improvise a workaround.

**Total time:** ~10 minutes. **Everything runs locally.** No daemon, no server, no cloud.

**Convention used below:** `<agent-name>` means the name of the Hermes profile this agent runs under. Substitute it literally (e.g. `myagent`) in commands.

---

## Step 0 — Survey the environment

Run these and read the outputs before touching anything:

```bash
python3 --version                 # must be 3.10+
ls ~/.hermes/                     # Hermes must already be installed
python3 -c "import onnxruntime, numpy; print('semantic deps OK')" 2>/dev/null || echo "semantic deps missing — install in Step 2"
echo "HERMES_HOME=${HERMES_HOME:-unset}"
ls ~/.hermes/profiles/            # which profile(s) exist?
```

Determine your install shape:

- If `HERMES_HOME` ends with `/profiles/<agent-name>`, you are **profile-scoped**: the plugin treats `$HERMES_HOME` as your profile directory.
- Otherwise (**default-profile install**), the plugin uses `~/.hermes/profiles/<agent-name>/`.
- Either way, `<agent-name>` is your identity inside VERA. The gateway passes it to the plugin as `agent_identity` — usually the profile directory name. If unsure, use the profile directory name from `ls ~/.hermes/profiles/`.

If your profile already contains a `vera/memory/` directory with `.md` files (an older file-based VERA install), STOP and read **"Legacy migration"** at the bottom of this file first.

## Step 1 — Install the plugin

```bash
git clone https://github.com/acculoe/vera-memory.git /tmp/vera-memory
mkdir -p ~/.hermes/plugins/vera_memory
cp -r /tmp/vera-memory/plugin/* ~/.hermes/plugins/vera_memory/
rm -rf /tmp/vera-memory
ls ~/.hermes/plugins/vera_memory/   # expect: __init__.py models.py extractor.py retriever.py semantic.py persistence.py plugin.yaml models/ ...
```

**Probe 1** — the plugin must import cleanly (standalone, no gateway needed):

```bash
cd ~/.hermes/plugins && python3 - <<'EOF'
import sys, types, os
# Stub the one Hermes-runtime import so this probe works standalone.
agent_mod = types.ModuleType("agent"); mp_mod = types.ModuleType("agent.memory_provider")
class MemoryProvider: pass
mp_mod.MemoryProvider = MemoryProvider; agent_mod.memory_provider = mp_mod
sys.modules["agent"] = agent_mod; sys.modules["agent.memory_provider"] = mp_mod
sys.path.insert(0, os.getcwd())
import vera_memory
print("plugin import OK:", vera_memory.__file__)
print(open("vera_memory/plugin.yaml").read().split("version:")[1].split()[0])
EOF
```

Expected: `plugin import OK: .../__init__.py` and `5.6.2`. Any `SyntaxError` or `ImportError` → stop, report.

## Step 2 — Install Python dependencies

```bash
pip install --user onnxruntime numpy 2>/dev/null || pip install onnxruntime numpy
python3 -c "import onnxruntime, numpy; print('OK')"
```

## Step 3 — Configure Hermes

VERA is an *exclusive* memory provider. It is activated by `memory.provider`, **not** by `plugins.enabled`. Edit your profile's `config.yaml` (or use `hermes config set`):

```yaml
memory:
  memory_enabled: false          # kills the built-in `memory` tool
  user_profile_enabled: false    # kills the built-in user-profile injection
  provider: vera_memory          # activates VERA
```

If a `plugins.enabled` list exists, adding `vera_memory` to it is harmless but **not required and not sufficient by itself**.

**Then eliminate the MEMORY.md conflict.** The Hermes framework injects `memories/MEMORY.md` into every turn and it overrides VERA. Replace its contents with a pointer (keep a backup):

```bash
MEMDIR="$HOME/.hermes/memories"    # profile-scoped installs: use "$HERMES_HOME/memories"
mkdir -p "$MEMDIR"
[ -f "$MEMDIR/MEMORY.md" ] && cp "$MEMDIR/MEMORY.md" "$MEMDIR/MEMORY.md.pre-vera.bak"
printf '%s\n' '> VERA is the single source of truth for memory. Use `vera_search` to retrieve and `vera_store` to save. Do NOT use the `memory` tool.' > "$MEMDIR/MEMORY.md"
```

If your config's platform toolset lists include `memory` (e.g. under `platform_toolsets.telegram`), remove `- memory` from each list so the built-in tool never loads on any platform. Do **not** remove `memory` from `known_builtin_toolsets` — that section is Hermes' own registry of what exists, not what loads.

## Step 4 — Download the embedding model (semantic search)

VERA works without this (lexical FTS5-only mode), but semantic search is the point. ~23 MB, downloaded once, runs 100% locally forever:

```bash
cd ~/.hermes/plugins/vera_memory/models
curl -L -o model_int8.onnx \
  "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_int8.onnx"
ls -la model_int8.onnx   # expect roughly 23 MB
```

If the download fails (offline machine?), skip this step — VERA will log `lexical-only mode` at startup and everything else still works. You can add the model later and run the backfill worker (see `docs/ONNX-SEMANTIC.md`).

## Step 5 — Provision identity + end-to-end round-trip probe

Every memory belongs to a *principal* — an identity record `(platform, subject_id)`. Your agent's own principal is `("hermes", "<agent-name>")`. Without it, `vera_store` fails closed with "authenticated, provisioned sender is required" — that is the security model working, not a bug.

The script below uses the provider's own path-resolution logic (its `initialize()`), so the database is created at **exactly the path the runtime will use**, whatever your install shape:

```bash
cd ~/.hermes/plugins && VERA_AGENT_NAME='<agent-name>' python3 - <<'EOF'
import sys, types, os
agent_mod = types.ModuleType("agent"); mp_mod = types.ModuleType("agent.memory_provider")
class MemoryProvider: pass
mp_mod.MemoryProvider = MemoryProvider; agent_mod.memory_provider = mp_mod
sys.modules["agent"] = agent_mod; sys.modules["agent.memory_provider"] = mp_mod
sys.path.insert(0, os.getcwd())
import vera_memory

name = os.environ["VERA_AGENT_NAME"]
home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")

# Provision pass: let the provider resolve its own profile directory.
p = vera_memory.VERAMemoryProvider()
p.initialize(session_id="vera-provision", hermes_home=home, agent_identity=name)
db_path = p._profile_dir / "vera-memory" / "memory.db"
db = vera_memory.MemoryDB(db_path); db.connect()
pid = db.upsert_principal(platform="hermes", platform_subject_id=name, principal_type="agent")
print(f"agent principal provisioned: ('hermes', '{name}') -> id {pid}")
print(f"database at: {db_path}")
EOF
```

Expected: `agent principal provisioned: ('hermes', '<agent-name>') -> id 1` plus the database path. Write the agent name down — it must match what the gateway passes as `agent_identity` (normally the profile name).

**Probe 2 — full round trip** (store → search through the exact tool path the gateway uses):

```bash
cd ~/.hermes/plugins && VERA_AGENT_NAME='<agent-name>' python3 - <<'EOF'
import sys, types, os
agent_mod = types.ModuleType("agent"); mp_mod = types.ModuleType("agent.memory_provider")
class MemoryProvider: pass
mp_mod.MemoryProvider = MemoryProvider; agent_mod.memory_provider = mp_mod
sys.modules["agent"] = agent_mod; sys.modules["agent.memory_provider"] = mp_mod
sys.path.insert(0, os.getcwd())
import vera_memory

name = os.environ["VERA_AGENT_NAME"]
home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
p = vera_memory.VERAMemoryProvider()
p.initialize(session_id="vera-probe", hermes_home=home, agent_identity=name)
out1 = p.handle_tool_call("vera_store", {"content": "VERA install complete", "category": "facts"})
print("STORE ->", out1)
out2 = p.handle_tool_call("vera_search", {"query": "install complete"})
print("SEARCH ->", out2)
EOF
```

Expected: `STORE -> Memory stored (id: 1).` and `SEARCH ->` a block containing that memory. If instead you see "authenticated, provisioned sender is required", the agent name used here doesn't match the one provisioned in the previous script — check for typos and rerun.

To enroll a **human** (e.g. the operator's Telegram account) later, run the provisioning script with this line added after `db.connect()`:

```python
pid = db.upsert_principal(platform="telegram", platform_subject_id="<their numeric platform id>", principal_type="person")
```

Only the owner decides who gets enrolled. Never enroll a sender just because they messaged the agent.

## Step 6 — Restart the gateway and verify live

Config changes only take effect after a Hermes gateway restart, followed by `/new` in the chat to start a fresh session. **The operator does this** (agents don't restart their own gateway). Then verify from inside a live session:

| Probe | Command | Expected |
|-------|---------|----------|
| Tools present | ask the agent to run `vera_search(query="install complete")` | the memory from Probe 2 |
| Store live | `vera_store(content="first live memory", category="facts")` | `Memory stored (id: N).` |
| Semantic | store a sentence, then `vera_search` with different wording | still finds it (semantic match, not just keywords) |
| Isolation | (second profile, same steps) | the two profiles cannot see each other's memories |

Also check the gateway log for one of:
- `VERA V4: ONNX semantic backend initialized` → full mode
- `VERA V4: Semantic backend unavailable: ... (lexical-only mode)` → model missing (Step 4 skipped/failed)

## Step 7 — Build your own vera-startup skill (anti-hallucination)

Memory you don't *use* is dead weight. Without a fixed startup routine, agents hallucinate "I already checked my memory" instead of actually checking. Make yourself a skill so every fresh session starts the same way:

1. Create the skill directory:
   ```bash
   mkdir -p ~/.hermes/skills/vera-startup
   ```
2. Write `~/.hermes/skills/vera-startup/SKILL.md` — copy the template from the repo (`templates/vera-startup-SKILL-template.md`) and fill in the `<...>` placeholders.
3. Load it via your normal skill mechanism (`skills_list` / `skill_view`) and **run it once now** as a smoke test:
   - call `vera_context(topic="current projects")` — expect a context block or an honest "no relevant context found", never silence
   - call `vera_search(query="active tasks")`
   - greet your operator **by name** (this doubles as a hallucination canary — if you catch yourself using a display name from chat metadata instead of the configured name, your context injection is broken; investigate before continuing)

From now on, at the start of **every genuinely new session**: run the startup sequence first, `vera_store` important decisions **in the same turn they happen** (never batch to the end), and `vera_search` before reporting project state.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `vera_store` → "authenticated, provisioned sender is required" | No principal for the calling identity | Step 5; verify the gateway passes `agent_identity` matching the profile name |
| No `vera_*` tools at all | Provider never registered | Check `memory.provider: vera_memory` spelling; restart gateway; check `hermes plugins list` finds the plugin dir |
| Plugin loads but tools missing | `HERMES_HOME` profile-scoping confusion | The plugin handles profile-scoped homes; verify Hermes passes `agent_identity`/`hermes_home` kwargs at `initialize()` |
| Lexical-only mode in logs | Model file missing/corrupt | Repeat Step 4; check file size ≈23 MB |
| MEMORY.md keeps getting written | Built-in `memory` tool still active | `memory_enabled: false` not applied, or `memory` still in a platform toolset, or gateway not restarted |
| Memories visible to the wrong agent | Two profiles sharing one DB | Each profile must have its own `profiles/<name>/vera-memory/memory.db` |
| Semantic search returns nothing after model install | Existing memories lack vectors | Run `python3 ~/.hermes/plugins/vera_memory/vera-onnx-backfill.py` (safe to re-run) |

## Legacy migration (only if `vera/memory/*.md` files exist)

The plugin **refuses to start** when un-migrated legacy files are present — this is deliberate protection, not an error to bypass. Migration is an explicit, offline, backed-up operation:

1. Stop the gateway (operator action).
2. Back up the entire profile directory (`cp -a`).
3. Read `safety.py` — it builds a verified backup + restoration package.
4. Run `migrator.py`'s `migrate_vera_files()` against a **copy** of the database, verify counts, then promote.
5. Restart. Never let migration run as a startup side effect.

If you have no legacy `.md` files, ignore this section entirely.
