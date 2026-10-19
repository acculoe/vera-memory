"""VERA v5.6.1: subject-free exact-hash dedupe guard — TEMP DB, never the live one.

The bug this locks down: before 5.6.1, store_memory's exact-hash dedupe also
matched subject_principal_id, so the same content stored under two subjects
(owner vs agent) created TWO active rows. The v5.5 audit found 722 such pairs
competing for the same recall slots.
"""
import pathlib as _p
import shutil, sys, tempfile
sys.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))

from models import MemoryDB

PASSED, FAILED = [], []

def check(name, fn):
    try:
        fn(); PASSED.append(name); print('  PASS', name)
    except Exception as e:
        FAILED.append(name); print('  FAIL', name, '->', type(e).__name__, e)

def fresh():
    d = tempfile.mkdtemp(prefix="vera561-")
    db = MemoryDB(_p.Path(d) / "memory.db")
    db.connect()  # creates schema + runs migrations on the temp DB
    return db, d

def active_count(db, content):
    return db._get_conn().execute(
        "SELECT COUNT(*) FROM memories WHERE content=? AND status='active'",
        (content,)).fetchone()[0]

CONTENT = "v5.6.1 dedupe guard probe: identical content must not duplicate across subjects"

def t_cross_subject_dedupes():
    db, d = fresh()
    try:
        a = db.store_memory(CONTENT, subject_principal_id=1, trust_tier="owner_declared")
        b = db.store_memory(CONTENT, subject_principal_id=49, trust_tier="observed_chat")
        assert a == b, (a, b)
        assert active_count(db, CONTENT) == 1, active_count(db, CONTENT)
    finally:
        shutil.rmtree(d, ignore_errors=True)

def t_same_subject_dedupes():
    db, d = fresh()
    try:
        a = db.store_memory(CONTENT, subject_principal_id=1)
        b = db.store_memory(CONTENT, subject_principal_id=1)
        assert a == b, (a, b)
        assert active_count(db, CONTENT) == 1
    finally:
        shutil.rmtree(d, ignore_errors=True)

def t_null_subject_dedupes_against_bound():
    db, d = fresh()
    try:
        a = db.store_memory(CONTENT, subject_principal_id=None)
        b = db.store_memory(CONTENT, subject_principal_id=1)
        c = db.store_memory(CONTENT, subject_principal_id=None)
        assert a == b == c, (a, b, c)
        assert active_count(db, CONTENT) == 1
    finally:
        shutil.rmtree(d, ignore_errors=True)

def t_same_subject_trust_upgrade_preserved():
    db, d = fresh()
    try:
        a = db.store_memory(CONTENT, subject_principal_id=1,
                            trust_tier="imported_history", importance=0.3)
        b = db.store_memory(CONTENT, subject_principal_id=1,
                            trust_tier="owner_declared", importance=0.9)
        assert a == b
        row = db._get_conn().execute(
            "SELECT trust_tier, importance FROM memories WHERE id=?", (a,)).fetchone()
        assert row["trust_tier"] == "owner_declared", dict(row)
        assert row["importance"] == 0.9, dict(row)
    finally:
        shutil.rmtree(d, ignore_errors=True)

def t_cross_subject_no_trust_theft():
    """A cross-subject duplicate returns the row but must NOT rewrite it."""
    db, d = fresh()
    try:
        a = db.store_memory(CONTENT, subject_principal_id=49,
                            trust_tier="observed_chat", importance=0.4)
        b = db.store_memory(CONTENT, subject_principal_id=1,
                            trust_tier="owner_declared", importance=0.95)
        assert a == b, (a, b)
        row = db._get_conn().execute(
            "SELECT trust_tier, importance, subject_principal_id FROM memories WHERE id=?",
            (a,)).fetchone()
        assert row["trust_tier"] == "observed_chat", dict(row)
        assert row["importance"] == 0.4, dict(row)
        assert row["subject_principal_id"] == 49, dict(row)
    finally:
        shutil.rmtree(d, ignore_errors=True)

def t_deduplicate_false_still_opt_out():
    db, d = fresh()
    try:
        a = db.store_memory(CONTENT, subject_principal_id=1)
        b = db.store_memory(CONTENT, subject_principal_id=49, deduplicate=False)
        assert a != b, (a, b)
        assert active_count(db, CONTENT) == 2
    finally:
        shutil.rmtree(d, ignore_errors=True)

def t_extractor_path_dedupes():
    """store_extracted_memory falls through to store_memory — one active row."""
    db, d = fresh()
    try:
        r1 = db.store_extracted_memory(
            {"content": CONTENT, "category": "facts",
             "subject_principal_id": 1, "source": "conversation"})
        r2 = db.store_extracted_memory(
            {"content": CONTENT, "category": "facts",
             "subject_principal_id": 49, "source": "conversation"})
        assert r1["memory_id"] == r2["memory_id"], (r1, r2)
        assert active_count(db, CONTENT) == 1, active_count(db, CONTENT)
    finally:
        shutil.rmtree(d, ignore_errors=True)

def t_prefers_own_subject_row_when_both_exist():
    """Legacy mess: both subject rows already active — a new same-subject
    store must return ITS row, not the other subject's (ORDER BY same DESC)."""
    db, d = fresh()
    try:
        conn = db._get_conn()
        a = db.store_memory(CONTENT, subject_principal_id=1, deduplicate=False)
        b = db.store_memory(CONTENT, subject_principal_id=49, deduplicate=False)
        c = db.store_memory(CONTENT, subject_principal_id=49)
        assert c == b, (a, b, c)
        d2 = db.store_memory(CONTENT, subject_principal_id=1)
        assert d2 == a, (a, b, c, d2)
    finally:
        shutil.rmtree(d, ignore_errors=True)

# Run only as a script: under pytest these are import-time side effects, and
# sys.exit() during collection aborts the whole run.
if __name__ == "__main__":
    for t in [t_cross_subject_dedupes, t_same_subject_dedupes,
              t_null_subject_dedupes_against_bound, t_same_subject_trust_upgrade_preserved,
              t_cross_subject_no_trust_theft, t_deduplicate_false_still_opt_out,
              t_extractor_path_dedupes, t_prefers_own_subject_row_when_both_exist]:
        check(t.__name__, t)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    sys.exit(1 if FAILED else 0)
