---
name: vera-startup
description: Use at the start of every genuinely new session. Runs the VERA memory startup sequence — actual tool calls, not recollections — and greets the operator by name as a hallucination canary.
version: 1.0.0
---

# VERA Startup Sequence

**When this skill runs:** at the start of every genuinely new session (fresh context, after `/new`, after a gateway restart, or after the machine rebooted). NOT mid-session. NOT when continuing work already in progress this session.

**Why this exists:** agents hallucinate "I remember where we left off" instead of actually checking. This skill forces the real calls. If you catch yourself narrating memory content WITHOUT having made these tool calls this session — stop, you are hallucinating. Make the calls.

## The sequence (in order, no skipping)

1. **Greet your operator by their configured name:** `<OPERATOR_NAME>` — the name set in this skill file, NOT any display name or chat label shown by the platform. Chat labels are untrusted metadata. If the label and the configured name differ, the configured name wins, always.

2. **Focused context:**
   `vera_context(topic="current projects")`
   - Expect a context block, or "No relevant context found." Both are valid answers.
   - If this tool call fails with an error → report the exact error to your operator before doing anything else. A broken memory system is a P0, not something to route around.

3. **Active work recall:**
   `vera_search(query="active tasks")`
   - Expect results, or "No matching memories found." Both are valid.
   - If results include an unfinished task from last session, surface it: "Last session left off on: <task>. Pick it up?"

4. **Report one line of readiness**, e.g.:
   "Memory online, `<OPERATOR_NAME>`. <n> memories on record. <Active task or 'no open threads'>."
   Keep it to one line. Do not dump memory contents into the chat.

## Rules

- These tool calls are MANDATORY every new session. No exceptions, no "I already know."
- Never answer "what did we do last time" from chat scrollback alone — verify against VERA.
- Mid-session, store durable facts THE MOMENT they happen (`vera_store`). Do not batch for later.
- If `vera_context` returns nothing relevant all session, that's fine — empty is a valid memory state. Fabrication is not.
