"""VERA Memory — Hermes Memory Provider Plugin.

Unified memory system replacing both MEMORY.md and VERA files.
SQLite-backed with FTS5 search, automatic extraction, per-agent isolation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

if __package__:
    from .extractor import extract_facts_from_turn, is_injected_payload
    from .migrator import migrate_vera_files
    from .models import MemoryDB
    from .safety import prepare_live_migration_package
    from .retriever import build_context_block, format_for_context, search
    from .persistence import PersistentMemoryManager, WorkingMemoryBuffer
    from . import self_heal
    from . import recall
    from . import briefing as briefing_mod
    from . import corrections
    from . import summarizer
    from . import classify as classify_mod
else:  # Direct source-tree import used by the isolated regression harness.
    from extractor import extract_facts_from_turn, is_injected_payload
    from migrator import migrate_vera_files
    from models import MemoryDB
    from safety import prepare_live_migration_package
    from retriever import build_context_block, format_for_context, search
    from persistence import PersistentMemoryManager, WorkingMemoryBuffer
    import self_heal
    import recall
    import briefing as briefing_mod
    import corrections
    import summarizer
    import classify as classify_mod

logger = logging.getLogger(__name__)

# Jev's category vocabulary -> the extractor's memory_type vocabulary.  Only the
# four the typed regexes produce are mapped; every other judge category lands as
# a plain "fact" so store_memory's default still applies.
_JUDGE_CATEGORY_TO_TYPE: Dict[str, str] = {
    "decisions": "decision",
    "facts": "fact",
    "preferences": "preference",
    "corrections": "correction",
}


class VERAMemoryProvider(MemoryProvider):
    """Unified memory provider for Hermes Agent."""

    def __init__(self):
        self._hermes_home: Optional[Path] = None
        self._profile_dir: Optional[Path] = None
        self._db: Optional[MemoryDB] = None
        self._session_id: str = ""
        self._turn_count: int = 0
        self._extraction_enabled: bool = True
        self._max_prefetch_chars: int = 3000
        # v5.6: reserved slice of the ~1100-char injection budget for the
        # ledger 'seen before' tier (owner-approved knob, plan Q2).
        self._recall_budget_chars: int = 300
        self._semantic_backend: Any = None
        self._persistence: Optional[PersistentMemoryManager] = None
        self._working_buffer: Optional[WorkingMemoryBuffer] = None
        self._briefing_cache: str = ""

    def _provision_identity_registry(self, entries: Any) -> None:
        """Load owner-controlled identity records supplied by local runtime config."""
        if not self._db or not isinstance(entries, list):
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            platform = str(entry.get("platform", "")).strip().lower()
            subject_id = str(entry.get("platform_subject_id", "")).strip()
            if not platform or not subject_id:
                continue
            principal_id = self._db.upsert_principal(
                platform=platform,
                platform_subject_id=subject_id,
                principal_type=str(entry.get("principal_type", "person")),
            )
            self._db.set_canonical_identity(
                principal_id,
                context_scope=str(entry.get("context_scope", "default")),
                reference_name=entry.get("reference_name"),
                direct_address=entry.get("direct_address"),
            )

    @property
    def name(self) -> str:
        return "vera-memory"

    def is_available(self) -> bool:
        """Check whether the resolved Hermes profile storage is writable."""
        try:
            # An unset environment variable must fall back to ~/.hermes. Path("")
            # resolves to the process working directory, which makes Dashboard-side
            # provider discovery incorrectly report unavailable.
            configured_home = os.environ.get("HERMES_HOME")
            hermes_home = Path(configured_home).expanduser() if configured_home else Path.home() / ".hermes"
            if not hermes_home.exists():
                hermes_home = Path.home() / ".hermes"
            # Handle profile-scoped HERMES_HOME (e.g. profiles/phil)
            hermes_str = str(hermes_home).replace("\\", "/")
            if hermes_str.endswith("/profiles") or hermes_str.count("/profiles/") >= 1:
                # Already points at or below profiles/ — use as-is
                test_dir = hermes_home
            else:
                test_dir = hermes_home / "profiles"
            return test_dir.exists() and os.access(str(test_dir), os.W_OK)
        except Exception:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Set up database. Uses agent_identity for per-agent isolation."""
        # Resolve hermes_home
        self._hermes_home = Path(kwargs.get("hermes_home", "") or os.environ.get("HERMES_HOME", ""))
        if not self._hermes_home or not self._hermes_home.exists():
            try:
                from hermes_constants import get_hermes_home
                self._hermes_home = get_hermes_home()
            except Exception:
                self._hermes_home = Path.home() / ".hermes"

        self._session_id = session_id
        self._turn_count = 0
        self._extraction_enabled = bool(kwargs.get("extraction_enabled", True))
        self._max_prefetch_chars = int(kwargs.get("max_prefetch_chars", 500))
        # FIX #3 + FIX for double-nesting: Detect if hermes_home is already profile-scoped
        agent_identity = kwargs.get("agent_identity", "") or kwargs.get("profile", "")
        self._agent_identity = agent_identity  # V5 identity fix: used by handle_tool_call fallback
        hermes_str = str(self._hermes_home).replace("\\", "/")

        # If hermes_home already ends with /profiles/<identity>, it's profile-scoped
        if agent_identity and hermes_str.endswith(f"profiles/{agent_identity}"):
            self._profile_dir = self._hermes_home
        elif agent_identity:
            self._profile_dir = self._hermes_home / "profiles" / agent_identity
        else:
            self._profile_dir = self._hermes_home / "profiles" / "default"

        # Legacy Markdown import is never a startup side effect. Refuse before
        # opening or creating a SQLite path, so an attempted startup cannot
        # mutate either legacy files or the live profile's database state.
        db_dir = self._profile_dir / "vera-memory"
        migration_marker = db_dir / ".migration_complete"
        legacy_memory_dir = self._profile_dir / "vera" / "memory"
        if not migration_marker.exists() and legacy_memory_dir.is_dir() and any(legacy_memory_dir.glob("*.md")):
            raise RuntimeError(
                "legacy VERA files require explicit offline migration with a verified backup/restoration package"
            )
        db_path = db_dir / "memory.db"
        # A provider must never migrate a live path as a side effect of startup.
        # Existing databases require an explicit opt-in plus a verified backup and
        # separately verified restoration package before any schema write.
        if db_path.exists() and MemoryDB.needs_migration_at_path(db_path):
            if not bool(kwargs.get("allow_live_migration", False)):
                raise RuntimeError("live VERA database requires explicit approved migration")
            package_dir = db_dir / "preupgrade-backup"
            package = prepare_live_migration_package(db_path, package_dir)
            if not package.get("verified"):
                raise RuntimeError("verified live migration package is required")
        self._db = MemoryDB(db_path)
        self._db.connect()

        # Initialize ONNX semantic backend for V4 (MUST be after DB connect)
        self._semantic_backend = kwargs.get("semantic_backend")
        if self._semantic_backend is None:
            try:
                from .embedding_adapter import ONNXEmbeddingAdapter
                from .semantic import SQLiteExactSemanticBackend
                adapter = ONNXEmbeddingAdapter()
                self._semantic_backend = SQLiteExactSemanticBackend(self._db, adapter)
                logger.info("VERA V4: ONNX semantic backend initialized")
            except Exception as e:
                logger.warning("VERA V4: Semantic backend unavailable: %s (lexical-only mode)", e)
                self._semantic_backend = None

        # V5: Embedding self-heal — count cards vs shelves at every startup,
        # rebuild what is missing through the existing worker path. Bounded so
        # startup stays responsive; a TimeoutError here leaves committed
        # progress intact and the gap shrinks on each restart until zero.
        if self._semantic_backend is not None:
            try:
                heal_report = self_heal.heal_embeddings(
                    self._db,
                    self._semantic_backend.adapter,
                    timeout_seconds=float(kwargs.get("self_heal_budget_seconds", 60.0)),
                    batch_size=int(kwargs.get("self_heal_batch_size", 32)),
                )
                logger.info(
                    "VERA v5: embedding self-heal ran (gap_before=%d embedded=%d failed=%d)",
                    heal_report["missing_before"], heal_report["embedded"],
                    heal_report["failed_count"],
                )
            except TimeoutError as e:
                logger.warning(
                    "VERA v5: embedding self-heal hit budget cap, resuming next "
                    "startup: %s", e,
                )
            except Exception as e:
                # Never block session start on healing, but never stay silent.
                logger.warning("VERA v5: embedding self-heal could not run: %s", e)

        # Identity mappings come from private runtime configuration, never from
        # retrieved memories, message labels, or source-controlled fixtures.
        self._provision_identity_registry(kwargs.get("identity_registry"))

        # V4.5: Initialize persistent working memory
        vera_core_dir = self._profile_dir / "vera" / "core"
        self._persistence = PersistentMemoryManager(vera_core_dir)
        self._persistence.load()
        self._working_buffer = WorkingMemoryBuffer(
            max_entries=int(kwargs.get("working_buffer_size", 10))
        )
        logger.info(
            "VERA v4.5: Persistent memory loaded (%d entries), working buffer ready",
            len(self._persistence._entries),
        )

        # v5.5: every session starts with a fresh Jev classification budget
        # so a long session cannot starve the judge (and vice versa).
        try:
            classify_mod.reset_budget()
        except Exception as e:  # pragma: no cover
            logger.debug("VERA v5.5: classify budget reset skipped: %s", e)

        # V5: Build the session-start briefing ONCE (prompt-cache safe).
        # Last ended session topic + current Active Task from persistence.md.
        try:
            # v5.5 BUG FIX: the section is named "Active Task"
            # (persistence.SECTION_HEADERS). v5 asked for "Active Project",
            # which does not exist, so this always evaluated to [] and the
            # briefing's "Active task:" line NEVER reached the agent — every
            # session opened believing nothing was in flight. Placeholder
            # bullets are filtered too, so "(none)" is never reported as state.
            _active_task_lines: List[str] = []
            if getattr(self._persistence, "get_section", None):
                _active_task_lines = [
                    line
                    for line in (self._persistence.get_section("Active Task") or [])
                    if line and not briefing_mod._is_placeholder_line(line)
                ]
            self._briefing_cache = briefing_mod.build_session_briefing(
                self._db, _active_task_lines[:1],
            )
            if self._briefing_cache:
                logger.info("VERA v5: session briefing built")
        except Exception as e:
            logger.debug("VERA v5: briefing build skipped: %s", e)
            self._briefing_cache = ""

        # V5: Give the recall layer an honest handle on persistence.md's
        # Active Task section (previously never populated by anyone).
        try:
            self._briefing_persistence = briefing_mod.BriefingPersistence(
                vera_core_dir / "persistence.md"
            )
        except Exception as e:
            logger.debug("VERA v5: briefing persistence handle failed: %s", e)
            self._briefing_persistence = None

        # Legacy Markdown import was refused above before opening this path.

        # Start session
        self._db.start_session(session_id)
        logger.info(
            "VERA Memory initialized: %s (%d memories) [agent=%s]",
            db_path, self._db.get_memory_count(), agent_identity,
        )

    def _run_migration(self, db_dir: Path, marker: Path) -> None:
        """Run first-time migration from VERA .md files."""
        logger.info("Running first-time VERA migration...")
        shared_vera = self._hermes_home if self._hermes_home else None
        report = migrate_vera_files(
            db=self._db,
            profile_dir=self._profile_dir,
            shared_vera_dir=shared_vera,
            archive=True,
        )
        db_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(
            "Migration complete: %d entries stored, %d files archived",
            report["entries_stored"], report["files_archived"],
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "vera_search",
                    "description": "Search your VERA memory for relevant facts, decisions, and context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "category": {"type": "string", "description": "Optional: decisions, facts, preferences, corrections, research, tasks"},
                            "limit": {"type": "integer", "description": "Max results (default 10)"},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vera_store",
                    "description": "Explicitly store an important fact in VERA memory. For a memory about a NAMED ENTITY, lead the first line with canonical name + aliases: 'Jev (TypeSafe AI System One decision model): ...' - this is what makes later recall work.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "The fact to store. Entity memories: first line = canonical name + parenthetical aliases"},
                            "category": {"type": "string", "description": "Category"},
                            "importance": {"type": "number", "description": "0.0-1.0"},
                        },
                        "required": ["content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vera_forget",
                    "description": "Withdraw a memory so it is no longer retrievable. Use when the owner retracts, corrects to death, or no longer wants a stored fact recalled. The row is kept for audit; only retrieval stops.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "integer", "description": "Exact memory id to withdraw (from a prior vera_store/vera_search)"},
                            "query": {"type": "string", "description": "Search text to locate the memory when the id is unknown"},
                            "reason": {"type": "string", "description": "Why it is being withdrawn; recorded in the audit trail"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "vera_context",
                    "description": "Get a summary of all relevant VERA memory for a topic.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "Topic to get context for"},
                        },
                        "required": ["topic"],
                    },
                },
            },
        ]

    def _resolve_own_principal(self) -> Optional[Dict[str, Any]]:
        """V5 fix: resolve this agent's own principal from trusted provider state."""
        if not self._db:
            return None
        try:
            ident = getattr(self, "_agent_identity", "") or "default"
            row = self._db.resolve_principal("hermes", ident)
            return dict(row) if row else None
        except Exception:
            return None

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._db:
            return "Error: VERA Memory not initialized"
        try:
            platform = str(kwargs.get("platform", "telegram"))
            sender_id = kwargs.get("sender_id") or kwargs.get("platform_subject_id")
            authenticated = bool(kwargs.get("sender_authenticated", False))
            logger.info("VERA handle_tool_call: tool=%s, platform=%s, sender_id=%s, authenticated=%s, kwargs_keys=%s", tool_name, platform, sender_id, authenticated, list(kwargs.keys()))
            # Resolve principal if sender_id is available, regardless of auth flag
            principal = self._db.resolve_principal(platform, str(sender_id)) if sender_id else None
            principal_id = int(principal["id"]) if principal else None
            # Fallback: if no sender_id or principal not found, use default agent principal
            if principal_id is None:
                # V5 identity fix: resolve the AGENT'S OWN principal, never a hardcoded name
                default_platform = "hermes"
                default_subject = getattr(self, "_agent_identity", "") or "default"
                principal = self._db.resolve_principal(default_platform, default_subject)
                if principal:
                    principal_id = int(principal["id"])
                    authenticated = True
                    logger.info("VERA handle_tool_call: using default agent principal=%s", principal_id)
            logger.info("VERA handle_tool_call: principal=%s, principal_id=%s", principal, principal_id)
            if tool_name == "vera_search":
                query = args.get("query", "")
                category = args.get("category")
                limit = args.get("limit", 10)
                results = search(
                    self._db, query, category=category, limit=limit,
                    principal_id=principal_id, authenticated=authenticated and principal_id is not None,
                    semantic_backend=self._semantic_backend,
                )
                return format_for_context(results, max_chars=2000) if results else "No matching memories found."
            elif tool_name == "vera_store":
                content = args.get("content", "")
                if not content.strip():
                    return "Error: content is required"
                if not authenticated or principal_id is None:
                    return "Error: authenticated, provisioned sender is required to store memory"
                # v5.5: THE JEV GATE. One judge call decides how this memory
                # is filed (category), how much it weighs (importance) and
                # repairs speech-to-text artifacts in the text — the same
                # classification then feeds persistence.md below, so the
                # store and the working-memory file can never disagree.
                #
                # caller_asserted=True: an explicit vera_store is somebody
                # deliberately remembering something, so the judge may
                # re-categorize/re-weight but may NOT drop it. Gating an
                # intentional store would silently lose owner intent — the
                # automatic path (sync_turn) is where skip is enforced.
                hint = args.get("category")
                verdict = classify_mod.classify(
                    content, category_hint=hint, caller_asserted=True
                )
                clean_content = verdict["cleaned"] or content.strip()
                if verdict["source"] == "jev":
                    category = verdict["category"]
                else:
                    # Judge unavailable -> regex is only a guess, so an
                    # explicit category from the caller always wins over it.
                    category = hint if hint in classify_mod.CATEGORIES else verdict["category"]
                supplied = args.get("importance")
                try:
                    importance = (
                        float(supplied)
                        if supplied is not None
                        else verdict["importance"]
                    )
                except (TypeError, ValueError):
                    importance = verdict["importance"]
                # V5: explicit stores now resolve corrections — a near-match
                # active memory is superseded instead of silently co-existing.
                result = corrections.store_with_resolution(
                    self._db,
                    content=clean_content,
                    category=category,
                    trust_tier="owner_declared",
                    source_type="authenticated_tool",
                    subject_principal_id=principal_id,
                    sender_principal_id=principal_id,
                    source_session=self._session_id,
                    importance=importance,
                )
                memory_id = result["memory_id"]
                if result["action"] == "superseded":
                    logger.info(
                        "VERA v5: vera_store superseded stale memory %s",
                        result.get("superseded_id"),
                    )
                # V4.5: Add to working buffer + check persistence
                if self._working_buffer:
                    self._working_buffer.add(
                        content,
                        args.get("category", "general"),
                        source="explicit_store",
                    )
                if self._persistence:
                    # Same Jev-classified payload the database just got, so
                    # persistence.md reflects what was actually stored rather
                    # than a second, independent guess.
                    self._persistence.add_entry({
                        "content": clean_content,
                        "category": category,
                        "importance": importance,
                    })
                    try:
                        self._persistence.write()
                    except Exception as e:
                        logger.debug("VERA v4.5: Post-store persistence write failed: %s", e)
                return (
                    f"Memory stored (id: {memory_id}, category: {category}, "
                    f"importance: {importance}, judge: {verdict['source']})."
                )
            elif tool_name == "vera_forget":
                # v5.5: withdrawal was the one missing verb — you could store
                # and search but never retract. Locate by id, or by query when
                # only the text is known, then withdraw under the same
                # authentication gate the store path uses.
                memory_id = args.get("memory_id")
                if memory_id is None:
                    query = str(args.get("query", "") or "")
                    if not query.strip():
                        return "Error: provide memory_id or query"
                    if not authenticated or principal_id is None:
                        return "Error: authenticated, provisioned sender is required to withdraw memory"
                    hits = search(
                        self._db, query, limit=5,
                        principal_id=principal_id,
                        authenticated=authenticated and principal_id is not None,
                        semantic_backend=self._semantic_backend,
                    )
                    if not hits:
                        return "No matching memory found to withdraw."
                    if len(hits) > 1:
                        listed = "\n".join(
                            f"- id {h.get('id')}: {str(h.get('content',''))[:90]}"
                            for h in hits
                        )
                        return (
                            "Multiple matches — reply with the exact memory_id to withdraw:\n"
                            + listed
                        )
                    memory_id = hits[0].get("id")
                withdrawal = self._db.withdraw_memory(
                    int(memory_id),
                    reason=str(args.get("reason", "") or ""),
                    actor=getattr(self, "_agent_identity", "") or "agent",
                    principal_id=principal_id,
                    authenticated=authenticated and principal_id is not None,
                )
                if withdrawal.get("action") != "withdrawn":
                    return f"Withdrawal not applied: {withdrawal.get('error', withdrawal.get('action'))}"
                # Drop it from working memory too, so this session stops
                # echoing a fact the owner just retracted.
                if self._working_buffer:
                    try:
                        self._working_buffer.remove(withdrawal.get("content", ""))
                    except Exception:
                        pass
                return (
                    f"Withdrawn (id: {withdrawal['memory_id']}). It stays in the "
                    f"record for audit but will no longer be retrieved."
                )
            elif tool_name == "vera_context":
                topic = args.get("topic", "")
                # V5 fix: handle_tool_call already resolved the right principal
                # (real sender, or this agent's own). Pass it through instead of
                # letting build_context_block re-guess from empty transport.
                return build_context_block(
                    self._db, topic, max_chars=self._max_prefetch_chars,
                    platform=platform, sender_id=str(sender_id) if sender_id else None,
                    sender_authenticated=authenticated,
                    resolved_principal=principal,
                ) or "No relevant context found."
            return f"Error: Unknown tool {tool_name}"
        except Exception as e:
            logger.error("VERA Memory tool error: %s", e)
            return f"Error: {e}"

    def sync_turn(self, user: str, assistant: str, *, session_id: str = "", messages: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> None:
        """Extract facts from this turn. Uses explicit session_id when provided."""
        if not self._extraction_enabled or not self._db:
            # V5.6.1 failure visibility: this bare return used to be invisible, so a
            # session could record memories_created=0 with nothing in the log to
            # explain it. Say it out loud instead.
            logger.warning(
                "VERA sync_turn SKIPPED (extraction_enabled=%s, db_open=%s) — "
                "this turn will NOT be extracted into memory",
                self._extraction_enabled, bool(self._db),
            )
            return
        # FIX #6: Use provided session_id, fall back to stored one
        active_session = session_id or self._session_id
        self._turn_count += 1
        try:
            platform = str(kwargs.get("platform", "telegram"))
            sender_id = kwargs.get("sender_id") or kwargs.get("platform_subject_id")
            principal = self._db.resolve_principal(platform, str(sender_id)) if sender_id else None
            extracted = extract_facts_from_turn(
                user,
                assistant,
                platform=platform,
                sender_id=str(sender_id) if sender_id else None,
                sender_principal_id=principal["id"] if principal else None,
                authenticated=bool(kwargs.get("sender_authenticated", False)),
            )
            # v5.5: Jev gate on the AUTOMATIC path. This is the valve where
            # content reaches the store with nobody asserting durability, so
            # it is where a judge earns its keep: drop non-durable candidates
            # and repair dictation artifacts. The extractor keeps owning
            # category/memory_type for TYPED candidates (they stay consistent
            # with each other) so a failed judge degrades to exactly v5
            # behavior instead of losing memories — classify() falls back to
            # regex, and regex never drops what the extractor already deemed
            # durable.  ``judge_only`` candidates are the opposite contract:
            # no typed signal, so they survive ONLY on an explicit Jev verdict
            # and a Jev outage drops them rather than blind-storing them.
            _candidate_count = len(extracted)
            if extracted:
                _kept = []
                _dropped = 0
                for mem in extracted:
                    _untyped = bool(mem.pop("judge_only", False))
                    _verdict = classify_mod.classify(
                        str(mem.get("content", "")),
                        category_hint=None if _untyped else mem.get("category"),
                        caller_asserted=False,
                    )
                    if not _verdict["store"]:
                        _dropped += 1
                        continue
                    if _untyped and _verdict.get("source") != "jev":
                        # No Jev verdict (timeout / session budget / spawn
                        # failure) -> classify() fell back to regex, whose
                        # default bucket stores everything. Never let the
                        # fallback create a memory the judge never saw.
                        _dropped += 1
                        continue
                    if _verdict["cleaned"]:
                        mem["content"] = _verdict["cleaned"]
                    if _untyped:
                        mem["category"] = _verdict["category"]
                        mem["memory_type"] = _JUDGE_CATEGORY_TO_TYPE.get(
                            _verdict["category"], "fact"
                        )
                        mem["importance"] = classify_mod.importance_for(
                            _verdict["category"]
                        )
                    _kept.append(mem)
                if _dropped:
                    logger.info(
                        "VERA auto-extract: judge dropped %d of %d candidate(s) this turn",
                        _dropped, _candidate_count,
                    )
                extracted = _kept

            stored = 0
            deduped = 0
            new_ids = []
            if extracted:
                for mem in extracted:
                    mem["source_session"] = active_session
                    result = self._db.store_extracted_memory(mem)
                    if result["action"] != "duplicate":
                        stored += 1
                        # store_extracted_memory returns "memory_id" (not "id"),
                        # so the incremental embed below never fired on the
                        # automatic path and extracted rows stayed at
                        # embedding_status='missing'.
                        if result.get("memory_id"):
                            new_ids.append(int(result["memory_id"]))
                        # V4.5: Add to working buffer for within-session continuity
                        if self._working_buffer:
                            self._working_buffer.add(
                                mem.get("content", ""),
                                mem.get("category", "general"),
                                source="extraction",
                            )
                        # V4.5: Check if this should persist across sessions
                        if self._persistence:
                            self._persistence.add_entry(mem)
                    else:
                        # v5.6.2: a "duplicate" means store_memory
                        # matched an existing row — near-duplicate or exact
                        # hash — and wrote NOTHING. Counting it as a store is
                        # what made this line claim stored=1 for sessions whose
                        # row count was 0.
                        deduped += 1
            # A scripted envelope (hourly fleet ping / query file) is rejected
            # by DESIGN, so 0 candidates there is a pass, not a fault. This
            # WARNING exists to catch real extraction outages; spending it on
            # payloads we meant to drop trains the reader to ignore it
            _injected = bool(_candidate_count == 0 and is_injected_payload(user))
            if _injected:
                # Telemetry must never be able to abort the turn: everything
                # here shares sync_turn's single try/except, so a failure in a
                # bookkeeping call would cost the caller its log line and any
                # remaining stores. Optional on purpose — the store interface
                # sync_turn depends on is resolve_principal + store_extracted.
                try:
                    _record = getattr(self._db, "record_session_outcome", None)
                    if callable(_record):
                        _record(active_session, rejected_payload=1)
                except Exception as e:
                    logger.debug("VERA session-outcome telemetry skipped: %s", e)
            # Unconditional per-turn telemetry: this is the line that would
            # have shown the outage instead of a bare memories_created=0.
            logger.info(
                "VERA auto-extraction turn=%d session=%s candidates=%d "
                "survived_judge=%d stored=%d deduped=%d",
                self._turn_count, active_session, _candidate_count,
                len(extracted), stored, deduped,
            )
            if stored == 0 and _candidate_count == 0:
                if _injected:
                    logger.info(
                        "VERA auto-extraction turn=%d session=%s: scripted "
                        "payload (%d chars) rejected pre-judge — 0 candidates "
                        "is expected, not an extraction failure",
                        self._turn_count, active_session, len(user or ""),
                    )
                else:
                    logger.warning(
                        "VERA auto-extraction produced 0 candidates this turn "
                        "(session=%s, %d chars of user text) — extractor pre-filter "
                        "rejected everything; check INJECTED_PREFIXES / "
                        "MAX_INPUT_CHARS / typed patterns",
                        active_session, len(user or ""),
                    )
            if stored > 0:
                # V4.5: Write persistence if updated
                if self._persistence:
                    try:
                        self._persistence.write()
                    except Exception as e:
                        logger.warning("VERA v4.5: Persistence write failed: %s", e)
                # V4: Generate embeddings for new memories
                if self._semantic_backend and new_ids:
                    try:
                        self._incremental_embed(new_ids)
                    except Exception as e:
                        logger.debug("VERA V4: Incremental embed failed: %s", e)
        except Exception as e:
            logger.warning(
                "VERA Memory extraction error: session=%s turn=%d: %s",
                session_id or self._session_id, self._turn_count, e, exc_info=True,
            )

    def _incremental_embed(self, memory_ids: list) -> None:
        """Generate embeddings for specific memories (incremental)."""
        if not self._semantic_backend or not memory_ids:
            return
        try:
            from .semantic import SQLiteVectorStore
            adapter = self._semantic_backend.adapter
            store = SQLiteVectorStore(self._db, adapter.spec)
            
            # Load memories to embed
            rows = self._db.get_active_memories_by_ids(
                memory_ids, principal_id=None, authenticated=True
            )
            if not rows:
                return
            
            # Generate embeddings
            texts = [row.get("content", "") for row in rows]
            vectors = adapter.encode(texts, timeout_seconds=10.0)
            
            # Store vectors
            upsert_data = [
                (int(row["id"]), row.get("content_hash", ""), vector)
                for row, vector in zip(rows, vectors)
            ]
            store.upsert(upsert_data)
            logger.debug("VERA V4: Embedded %d new memories", len(upsert_data))
        except Exception as e:
            # V5: incremental embed failures are NEVER silent — they were the
            # root cause of the progressive embedding gap (33.7% -> 59%).
            logger.warning(
                "VERA v5: incremental embed FAILED for %d memories: %s",
                len(memory_ids), e,
            )

    def prefetch(self, query: str, *, session_id: str = "", **kwargs: Any) -> str:
        """V4.5: Tiered injection — identity + relevant buffer + relevant persistence + search.

        Never dumps everything. Only injects what's relevant to the current query.
        Total budget: ~1100 chars (~275 tokens).
        """
        if not self._db:
            return ""
        parts = []
        try:
            # Tier 1: Identity block — always injected, tiny (~100 chars)
            parts.append(self._get_identity_block())

            # Tier 2a: Working buffer — relevant recent entries (~300 chars)
            if self._working_buffer:
                buffer_ctx = self._working_buffer.get_relevant(query, max_chars=300)
                if buffer_ctx:
                    parts.append(buffer_ctx)

            # Tier 2b: Persistence.md — relevant cross-session entries (~400 chars)
            if self._persistence:
                persist_ctx = self._persistence.get_relevant(query, max_chars=400)
                if persist_ctx:
                    parts.append(persist_ctx)

            # Tier 3: SQLite search — standard FTS5 + semantic (~300 chars)
            # V5 fix: same principal hole as vera_context — when the runtime
            # supplies no sender_id, resolve this agent's own principal from
            # trusted provider state instead of failing closed.
            _prefetch_principal = None
            _kw_sender = kwargs.get("sender_id") or kwargs.get("platform_subject_id")
            if bool(kwargs.get("sender_authenticated", False)) and _kw_sender:
                try:
                    _row = self._db.resolve_principal(str(kwargs.get("platform", "telegram")), str(_kw_sender))
                    _prefetch_principal = dict(_row) if _row else None
                except Exception:
                    _prefetch_principal = None
            if _prefetch_principal is None:
                _prefetch_principal = self._resolve_own_principal()
            search_ctx = build_context_block(
                self._db,
                query,
                max_chars=300,
                platform=str(kwargs.get("platform", "telegram")),
                sender_id=kwargs.get("sender_id") or kwargs.get("platform_subject_id"),
                display_name=kwargs.get("display_name"),
                sender_authenticated=bool(kwargs.get("sender_authenticated", False)),
                context_scope=str(kwargs.get("context_scope", "default")),
                semantic_backend=self._semantic_backend,
                resolved_principal=_prefetch_principal,
            )
            if search_ctx:
                parts.append(search_ctx)

            # V5 Tier 3.5: proactive recall — when the message signals work
            # start/continuation, pull project context WITHOUT waiting for an
            # explicit vera_search. Labeled "[proactive]", bounded ~400 chars.
            try:
                proactive_ctx = recall.expand_prefetch(
                    self._db,
                    query,
                    semantic_backend=self._semantic_backend,
                    authenticated=bool(kwargs.get("sender_authenticated", False)),
                    agent_identity=getattr(self, "_agent_identity", ""),
                )
                if proactive_ctx:
                    parts.append(proactive_ctx)
                    # V5: work-start also owns persistence.md's Active Task,
                    # so the NEXT session's briefing knows what was in flight.
                    bpm = getattr(self, "_briefing_persistence", None)
                    if bpm is not None:
                        try:
                            bpm.set_active_task(query.strip()[:280])
                        except Exception as e:
                            logger.debug("VERA v5: active-task write skipped: %s", e)
            except Exception as e:
                logger.debug("VERA v5: proactive recall skipped: %s", e)

            # V5.6 Tier 4: ledger recall trigger - "seen before". Fires when
            # the message names a term VERA already holds (the 'who is Jev?'
            # defect: stores were fine, nothing connected the spoken word to
            # them). Own budget slice, fail-open: a ledger problem never costs
            # the other tiers their output.
            try:
                from .ledger import recall_trigger
            except ImportError:  # direct source-tree harness
                from ledger import recall_trigger  # type: ignore
            try:
                seen_ctx = recall_trigger(
                    query, self._db._get_conn(),
                    budget=self._recall_budget_chars,
                )
                if seen_ctx:
                    parts.append(seen_ctx)
            except Exception as e:
                logger.debug("VERA v5.6 recall trigger skipped: %s", e)

            return "\n".join(parts)
        except Exception as e:
            logger.debug("VERA Memory prefetch failed: %s", e)
            return ""

    def _get_identity_block(self) -> str:
        """Tiny identity block — always injected via prefetch."""
        count = 0
        if self._db:
            try:
                count = self._db.get_memory_count()
            except Exception:
                pass
        buf_size = len(self._working_buffer) if self._working_buffer else 0
        persist_size = len(self._persistence._entries) if self._persistence else 0
        return (
            f"## VERA v5.5 — {count} memories, "
            f"{buf_size} working, {persist_size} persistent"
        )

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._db:
            # V5.6.1 failure visibility: without this, an uninitialized provider
            # silently ends the session with memories_created left at 0.
            logger.warning(
                "VERA on_session_end SKIPPED (db not open) — session row will keep "
                "memories_created=0 and no summary"
            )
            return
        try:
            # V5: REAL extractive summary replaces the "Last topic" stub.
            # The summarizer runs on user messages only, skips injected blobs,
            # and feeds the session row that the next session's briefing reads.
            summary = None
            try:
                s = summarizer.summarize_session(messages or [])
                if s["summary"]:
                    summary = f"Session summary: {s['summary']}"
            except Exception as e:
                logger.debug("VERA v5: summarize failed, falling back: %s", e)
            if summary is None and messages:
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        content = str(msg.get("content", ""))[:200]
                        if content.startswith("["):
                            break  # cron/injected blob — store nothing
                        summary = f"Session ended. Last topic: {content}"
                        break
            self._db.end_session(self._session_id, summary=summary, turn_count=self._turn_count)
            # V4.5: Final persistence curation + clear working buffer
            if self._persistence:
                try:
                    self._persistence.final_curation()
                    self._persistence.write()
                    logger.info("VERA v4.5: Session-end persistence curation complete")
                except Exception as e:
                    logger.debug("VERA v4.5: Session-end curation failed: %s", e)
            if self._working_buffer:
                self._working_buffer.clear()
        except Exception as e:
            logger.warning("VERA Memory session end error: %s", e)

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mirror built-in memory writes to SQLite (backward compat)."""
        if not self._db or not content:
            return
        try:
            self._db.store_memory(
                content=content,
                category="user" if target == "user" else "general",
                source_session=self._session_id,
                source="hermes-memory-tool",
                importance=0.6,
            )
        except Exception as e:
            logger.warning("VERA Memory mirror error: %s", e)

    def system_prompt_block(self) -> str:
        """V5: Minimal system prompt + one-shot session briefing.

        The briefing is built ONCE in initialize() and stays byte-stable for
        the whole conversation (prompt-cache safety). Prefetch still does the
        per-message work.
        """
        lines = [
            "\n## VERA Memory (v5.5)\n"
            "Memory system active. `vera_search` to recall, `vera_store` to save,\n"
            "`vera_forget` to withdraw a fact the owner no longer wants recalled.\n"
            "Facts extracted automatically; a Jev judge classifies what is kept.\n"
            "Working memory tracks recent context. Before starting named work,\n"
            "search VERA for that project's context first.\n"
        ]
        if self._briefing_cache:
            lines.append(self._briefing_cache + "\n")
        return "".join(lines)

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return []

    def shutdown(self) -> None:
        if self._db:
            self._db.close()
            self._db = None


def register(ctx) -> None:
    """Register the VERA Memory provider."""
    provider = VERAMemoryProvider()
    ctx.register_memory_provider(provider)
