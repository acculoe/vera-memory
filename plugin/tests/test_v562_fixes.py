#!/usr/bin/env python3
"""V5.6.2 behavior tests — scripted-payload guard + honest extraction telemetry.

Synthetic data only: every database here is a throwaway temp file, no real
corpus, no credentials, no network. Run either way:

    python3 tests/test_v562_fixes.py     # standalone script
    pytest tests/                        # collected as ordinary test functions

Jev (or whatever judge you configured) is forced off via CLASSIFY_ENABLED so
the regex fallback is deterministic; that is also the state a judge-less
install runs in.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import tempfile
import types
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]

# --- load the package exactly as an install would, without needing a gateway --
try:  # a real Hermes runtime already provides this
    from agent.memory_provider import MemoryProvider  # noqa: F401
except Exception:  # standalone run: stub the single runtime import
    _agent = types.ModuleType("agent")
    _mp = types.ModuleType("agent.memory_provider")

    class MemoryProvider:  # minimal base, only needed for isinstance/signature
        pass

    _mp.MemoryProvider = MemoryProvider
    _agent.memory_provider = _mp
    sys.modules["agent"] = _agent
    sys.modules["agent.memory_provider"] = _mp

_spec = importlib.util.spec_from_file_location(
    "vera_memory", PLUGIN_DIR / "__init__.py",
    submodule_search_locations=[str(PLUGIN_DIR)],
)
vera_memory = importlib.util.module_from_spec(_spec)
sys.modules["vera_memory"] = vera_memory
_spec.loader.exec_module(vera_memory)

import vera_memory.classify as classify_mod  # noqa: E402
from vera_memory.extractor import (  # noqa: E402
    extract_facts_from_turn,
    classify_memory_type,
    is_injected_payload,
)
from vera_memory.models import MemoryDB  # noqa: E402


# ---------------------------------------------------------------- helpers ----
def fresh_db() -> tuple[MemoryDB, Path]:
    """A brand-new database with all migrations applied."""
    d = Path(tempfile.mkdtemp(prefix="vera562-test-"))
    db = MemoryDB(d / "memory.db")
    db.connect()
    return db, d


def fresh_provider():
    """A provider wired to a throwaway HERMES_HOME with a provisioned agent."""
    home = Path(tempfile.mkdtemp(prefix="vera562-home-"))
    p = vera_memory.VERAMemoryProvider()
    p.initialize(session_id="test-session", hermes_home=str(home), agent_identity="testagent")
    p._db.upsert_principal(platform="hermes", platform_subject_id="testagent",
                           principal_type="agent")
    return p, home


def log_records(fn):
    """Run fn() and return the log records it emitted (INFO and above)."""
    captured: list[logging.LogRecord] = []

    class _Cap(logging.Handler):
        def emit(self, record):
            captured.append(record)

    loggers = [logging.getLogger(n) for n in ("", "vera_memory", "vera_memory.models")]
    handlers = []
    for lg in loggers:
        h = _Cap()
        lg.addHandler(h)
        handlers.append((lg, h))
    old = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.INFO)
    try:
        fn()
    finally:
        for lg, h in handlers:
            lg.removeHandler(h)
        for lg, lvl in zip(loggers, old):
            lg.setLevel(lvl)
    return captured


def messages(records, min_level=logging.INFO):
    return [r.getMessage() for r in records if r.levelno >= min_level]


# ------------------------------------------------- 1. scripted payloads ------
DIGEST = (
    "Status ping from the hourly digest. Answer in exactly this format:\n"
    "Working on — one or two sentences. Based on your freshest vera record, "
    "name the concrete project/task you are on. Do not answer from kanban board state."
)
ENVELOPE = "[Some Bot|4820] Review the conversation above and summarize it."
SPEECH = "I decided that the ledger database will be PostgreSQL from now on."


def test_payload_guard_flags_scripted_messages():
    assert is_injected_payload(DIGEST) is True
    assert is_injected_payload(ENVELOPE) is True
    assert is_injected_payload(SPEECH) is False


def test_payload_guard_blocks_extraction_before_judge():
    assert extract_facts_from_turn(DIGEST) == []
    assert extract_facts_from_turn(ENVELOPE) == []


def test_ordinary_speech_still_extracts():
    got = extract_facts_from_turn(SPEECH)
    assert got, "normal owner speech must still produce candidates"
    assert got[0]["category"] in {"decisions", "facts", "preferences", "corrections"}


def test_untyped_speech_becomes_judge_only_candidate():
    """No typed signal -> the candidate must be flagged, not silently dropped."""
    untyped = "The delivery window is narrow and the weather stays warm this week."
    assert classify_memory_type(untyped) is None
    got = extract_facts_from_turn(untyped)
    assert got and got[0].get("judge_only") is True, got


# --------------------------------------------- 2. honest dedupe accounting ----
def test_exact_hash_dedupe_reports_inserted_false():
    db, d = fresh_db()
    try:
        first, second = {}, {}
        a = db.store_memory(store_outcome=first, content="Dedupe probe row",
                            category="facts", subject_principal_id=1)
        b = db.store_memory(store_outcome=second, content="Dedupe probe row",
                            category="facts", subject_principal_id=1)
        assert first.get("inserted") is True, first
        assert second.get("inserted") is False, second
        assert second.get("id") == a == b, (a, b, second)
        n = db._get_conn().execute(
            "SELECT COUNT(*) c FROM memories WHERE content LIKE 'Dedupe probe row%'"
        ).fetchone()["c"]
        assert n == 1, f"expected 1 row, found {n}"
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_store_extracted_memory_reports_duplicate_action():
    db, d = fresh_db()
    try:
        db.store_memory(content="Extractor dedupe probe", category="facts",
                        subject_principal_id=1)
        res = db.store_extracted_memory({"content": "Extractor dedupe probe",
                                         "category": "facts"})
        assert res.get("action") == "duplicate", res
        assert isinstance(res.get("memory_id"), int), res
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------ 3. end-session triage -------
def _end(db, sid):
    return log_records(lambda: db.end_session(sid, summary="s", turn_count=1))


def test_all_dedupe_session_reports_info_not_warning():
    db, d = fresh_db()
    try:
        db.start_session("s-first")
        db.store_extracted_memory({"content": "Triage probe row", "category": "facts",
                                   "source_session": "s-first"})
        db.start_session("s-second")
        db.store_extracted_memory({"content": "Triage probe row", "category": "facts",
                                   "source_session": "s-second"})
        recs = _end(db, "s-second")
        infos, warns = messages(recs), messages(recs, logging.WARNING)
        assert any("matched" in m and "existing memories" in m for m in infos), infos
        assert not any("stored ZERO memories" in m for m in warns), warns
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_genuinely_empty_session_still_warns():
    db, d = fresh_db()
    try:
        db.start_session("s-empty")
        recs = _end(db, "s-empty")
        assert any("stored ZERO memories" in m for m in messages(recs, logging.WARNING)), \
            messages(recs)
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_scripted_payload_session_reports_info_not_warning():
    db, d = fresh_db()
    try:
        db.start_session("s-payload")
        db.record_session_outcome("s-payload", rejected_payload=1)
        recs = _end(db, "s-payload")
        infos, warns = messages(recs), messages(recs, logging.WARNING)
        assert any("scripted payload" in m for m in infos), infos
        assert not any("stored ZERO memories" in m for m in warns), warns
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


# -------------------------------------------- 4. judge_only never blind-store --
def test_judge_only_candidate_not_stored_without_a_judge():
    """Judge unavailable -> an untyped candidate is dropped, never blind-stored;
    a typed candidate still stores (the v5 fail-open contract)."""
    p, home = fresh_provider()
    old = classify_mod.CLASSIFY_ENABLED
    classify_mod.CLASSIFY_ENABLED = False
    try:
        conn = p._db._get_conn()
        before = conn.execute("SELECT COUNT(*) c FROM memories").fetchone()["c"]
        p.sync_turn("The delivery window is narrow and the weather stays warm.",
                    "", session_id="s-judge", platform="hermes",
                    sender_id="testagent", sender_authenticated=True)
        mid = conn.execute("SELECT COUNT(*) c FROM memories").fetchone()["c"]
        assert mid == before, f"untyped candidate was blind-stored ({before} -> {mid})"

        p.sync_turn(SPEECH, "", session_id="s-judge", platform="hermes",
                    sender_id="testagent", sender_authenticated=True)
        after = conn.execute("SELECT COUNT(*) c FROM memories").fetchone()["c"]
        assert after == mid + 1, f"typed candidate did not store ({mid} -> {after})"
    finally:
        classify_mod.CLASSIFY_ENABLED = old


# ------------------------------------------- 5. automatic-path embedding ids ---
def test_automatic_path_feeds_real_ids_to_embed():
    """store_extracted_memory returns memory_id; the auto path used to read `id`,
    so automatically extracted rows never got embedded on store."""
    p, home = fresh_provider()
    captured: list = []
    p._semantic_backend = object()      # pretend a semantic backend exists
    p._incremental_embed = lambda ids: captured.extend(ids)
    p.sync_turn("I approved the vendor invoice for the hosting bill today.", "",
                session_id="s-embed", platform="hermes", sender_id="testagent",
                sender_authenticated=True)
    assert captured, "no ids reached _incremental_embed"
    assert all(isinstance(i, int) for i in captured), captured


# ------------------------------------------------------- 6. skip visibility ----
def test_skipped_hooks_warn_instead_of_returning_silently():
    p = vera_memory.VERAMemoryProvider()   # never initialized
    recs = log_records(lambda: p.sync_turn("anything", "", session_id="s-skip"))
    assert any("SKIPPED" in m for m in messages(recs, logging.WARNING)), messages(recs)

    p2 = vera_memory.VERAMemoryProvider()
    recs = log_records(lambda: p2.on_session_end([]))
    assert any("SKIPPED" in m for m in messages(recs, logging.WARNING)), messages(recs)


# --------------------------------------------------------------- runner -------
ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = []
    for fn in ALL:
        try:
            fn()
            print("  PASS", fn.__name__)
        except AssertionError as exc:
            print("  FAIL", fn.__name__, "->", exc)
            failed.append(fn.__name__)
        except Exception as exc:  # noqa: BLE001
            print("  ERROR", fn.__name__, "->", type(exc).__name__, exc)
            failed.append(fn.__name__)
    print(f"\n{len(ALL) - len(failed)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)
