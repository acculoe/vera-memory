"""VERA v5.6 - term ledger: data spine for message-triggered recall.

Closes the v5.5 core defect behind "who is Jev?": stores were fine, but
nothing in the pipeline noticed when an owner message named a term VERA
already holds. Three consumers share this ONE extractor (DRY):

- migration 11 backfill (one pass over the whole corpus)
- store_memory / withdraw cascade (live maintenance, same transaction)
- recall_trigger() (the prefetch tier that says "seen before")

Design constraints (mirrors recall.py):
- No LLM calls, no network, no threads. Pure regex + SQL, all local.
- Stdlib-only and dependency-free: models.py imports THIS module, so it
  must never import models/retriever/recall (that cycle ends in models).
  STOPWORDS is copied from recall.py for exactly that reason.
- Fail-open at call sites: a ledger problem never blocks a store or a
  response. Purge (removal) is the one non-fail-open call - withdrawn
  means gone from every index.

Entity-grade admission (the store-side quality half of the fix): only
capitalized spans, ALLCAPS acronyms, digit-bearing tokens, and
quoted/backticked spans get ledgered, minus sentence-starter STOP words -
with leading STOP words stripped from phrases ("The LLC" -> "LLC"), so
sentence starts never become ledger keys.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Superset of recall.STOPWORDS + sentence-starters. Copied, not imported
# (see module docstring: dependency-free layer).
STOP = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "now", "lets", "let", "us", "we", "i", "you", "it",
    "its", "this", "that", "these", "those", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "doing", "would", "could", "ought", "im", "dont", "didnt", "whats",
    "hey", "hi", "hello", "thanks", "thank", "ok", "okay", "yeah", "yes",
    "please", "work", "working", "start", "starting", "continue",
    "continuing", "today", "tonight", "tomorrow", "still", "also",
    # interrogatives + common sentence-starters / past verbs
    "what", "when", "why", "how", "which", "who", "whom", "where",
    "make", "made", "add", "added", "update", "updated", "edit", "edited",
    "wrote", "write", "read", "signed", "said", "asked", "told",
    "checked", "need", "needs", "want", "want", "got", "get", "project",
    "session", "memory", "note", "new", "first", "next", "final", "sure",
    "right", "well", "actually", "maybe", "something", "everything",
    "anything", "one", "two",
})

# Quoted/backticked span | Capitalized multiword phrase | ALLCAPS acronym |
# Capitalized single token (3+ chars: entity pattern is [A-Z][a-z]{1,}).
_TERM_RE = re.compile(
    r"""(?P<quoted>[`'\u201c\u2018](?P<qin>[A-Za-z0-9_][\w.\-]{1,40})[`'\u201d\u2019])
      | (?P<phrase>\b[A-Z][\w.\-]+(?:[ \-][A-Z][\w.\-]+){1,3}\b)
      | (?P<acronym>\b[A-Z][A-Z0-9]{1,7}\b)
      | (?P<entity>\b[A-Z][a-z][\w.\-]{1,}\b)""",
    re.VERBOSE,
)

_TOKEN_RE = re.compile(r"[\w][\w.\-']*")


# Query side: function-word trim ONLY (wh-words, pronouns, auxiliaries).
# Candidates get intersected against term_stats before anything fires, so
# over-inclusion is harmless; over-trimming would drop the owner's own
# words ("jev get" must survive even though "get" is a store-side
# sentence-starter). Two lists, two jobs - do not merge them.
QUERY_STOP = frozenset({
    "the", "a", "an", "or", "and", "but", "of", "to", "in", "on", "at",
    "is", "are", "was", "were", "it", "this", "that", "these", "those",
    "not", "no", "nor", "do", "does", "did", "doing", "for", "with",
    "you", "we", "i", "they", "them", "he", "she", "his", "her", "my",
    "what", "when", "why", "how", "which", "who", "whom", "where",
    "if", "then", "so", "too", "very", "just", "about", "into", "from",
    # adverbs/pronouns/verbs that are incidentally English in a question
    # ("tell me about quuxzzz never stored anywhere" must stay quiet):
    "never", "always", "tell", "stored", "anywhere", "me", "us", "our",
    # ... plus question-scaffolding the harness caught ranking ahead of
    # the real referent ("quick question - did jev..." led with "question"):
    "question", "answer", "example", "thing", "stuff", "remember",
    # evidence-driven #2 (harness): common English verbs that are ledgered
    # as sentence-starters but never genuine referents in a question.
    # Deliberately NOT here: "get" (test_ledger requires the "jev get" bigram).
    "run", "said", "use", "see", "look", "go", "come", "think", "know",
    "check", "let", "take",
    "mine", "even", "still", "already", "really", "quite", "enough",
    "another", "other", "much", "many", "every", "fine", "thanks",
})


def make_norm(s: str) -> str:
    """Casefold + whitespace-collapse - the ledger key transform."""
    return " ".join((s or "").casefold().split())


def _admit(display: str) -> Optional[Tuple[str, str]]:
    """Strip leading STOP words from a span; None if nothing entity-grade left."""
    words = display.strip().split()
    while words and words[0].casefold() in STOP:
        words.pop(0)
    if not words:
        return None
    norm = make_norm(" ".join(words))
    if len(norm) < 2:
        return None
    return norm, " ".join(words)


def extract_terms(content: str) -> List[Tuple[str, str]]:
    """[(norm, display)] - entity-grade spans, deduped, order preserved."""
    out: List[Tuple[str, str]] = []
    seen = set()
    if not content:
        return out
    for m in _TERM_RE.finditer(content):
        raw = m.group("qin") or m.group("phrase") or m.group("acronym") or m.group("entity") or ""
        admitted = _admit(raw)
        if admitted is None:
            continue
        norm, disp = admitted
        if norm in seen:
            continue
        seen.add(norm)
        out.append((norm, disp))
    return out


def extract_query_terms(query: str) -> List[str]:
    """Case-INSENSITIVE n-grams from an owner message: unigrams + bigrams.

    Store side admits only capitalized spans; an owner often types "jev"
    lowercase, so the query side is deliberately greedy - candidates are
    cheap because only exact intersection with term_stats can fire.
    Bigrams with any STOP constituent are dropped (real phrases live in
    the ledger only if they were entity-grade at store time).
    """
    raw = [t.casefold() for t in _TOKEN_RE.findall(query or "")]
    out: List[str] = []
    for t in raw:
        if len(t) >= 2 and t not in QUERY_STOP and t not in out:
            out.append(t)
    for a, b in zip(raw, raw[1:]):
        if a in QUERY_STOP or b in QUERY_STOP:
            continue
        bg = a + " " + b
        if bg not in out:
            out.append(bg)
    return out


def context_for(content: str, term: str, width: int = 40) -> str:
    """±width chars around the first (case-insensitive) match, collapsed."""
    text = content or ""
    if not term:
        return " ".join(text.split())[: 2 * width + 20]
    m = re.search(re.escape(term), text, re.IGNORECASE)
    if m:
        start = max(0, m.start() - width)
        end = min(len(text), m.end() + width)
        return " ".join(text[start:end].split())
    return " ".join(text.split())[: 2 * width + 20]


def record_terms(conn, memory_id: int, content: str, now: str) -> int:
    """Ledger rows for one memory. Caller owns the transaction (store path)."""
    rows = [
        (norm, disp, int(memory_id), context_for(content, disp), now)
        for norm, disp in extract_terms(content)
    ]
    if not rows:
        return 0
    cur = conn.executemany(
        "INSERT OR IGNORE INTO term_ledger"
        " (norm, term, memory_id, context, recorded_at) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    return int(cur.rowcount or 0)


def purge_terms(conn, memory_id: int) -> int:
    """Ledger rows for a withdrawn/deleted memory. NOT fail-open: withdrawn
    must mean gone from every index (one fact, one home)."""
    cur = conn.execute("DELETE FROM term_ledger WHERE memory_id = ?", (int(memory_id),))
    return int(cur.rowcount or 0)


def recall_trigger(query: str, conn, *, max_terms: int = 3,
                   budget: int = 300) -> str:
    """Ledger-driven "seen before" block for the prefetch tier.

    The answer to the 'who is Jev?' defect: when an owner message names a
    term the ledger already holds, say so - mention count, truthful
    first-seen date (memory created_at), and the +-40 char context
    captured at store time. Two indexed lookups, no LLM, no network,
    fail-open: never blocks or delays a response.

    Fire rule (owner-approved): multi-word norms fire on first mention;
    single tokens need mentions >= 2 among ACTIVE memories. The
    retrievability filter mirrors the FTS trigger (status='active' AND
    retrieval_enabled=1) so superseded copies never inflate a count.
    """
    try:
        if conn is None:
            return ""
        cands = extract_query_terms(query)
        if not cands:
            return ""
        placeholders = ','.join('?' for _ in cands)
        rows = conn.execute(
            "SELECT t.norm, COUNT(*) AS mentions, MIN(t.recorded_at) AS first_seen"
            " FROM term_ledger t JOIN memories m ON m.id = t.memory_id"
            " WHERE t.norm IN (" + placeholders + ")"
            "   AND m.status = 'active' AND m.retrieval_enabled = 1"
            " GROUP BY t.norm",
            tuple(cands),
        ).fetchall()
        hits = {}
        for norm, mentions, first_seen in rows:
            mentions = int(mentions)
            if " " in norm or mentions >= 2:
                hits[norm] = (mentions, first_seen)
        if not hits:
            return ""
          # Rank by QUERY POSITION, not mention count: a generic high-count
        # word ("never", 89 mentions) must not crowd out the actual referent
        # ("jev", 12) that the owner actually led with. Count still drives
        # the display line and the single-token fire threshold.
        ranked = [(c, hits[c]) for c in cands if c in hits][:max_terms]
        header = "## Seen before"
        lines = []
        used = 0
        for norm, (mentions, first_seen) in ranked:
            best = conn.execute(
                "SELECT t.context FROM term_ledger t"
                " JOIN memories m ON m.id = t.memory_id"
                " WHERE t.norm = ? AND m.status = 'active'"
                "   AND m.retrieval_enabled = 1"
                " ORDER BY t.recorded_at DESC LIMIT 1",
                (norm,),
            ).fetchone()
            if not best or not best[0]:
                continue
            snippet = " ".join(str(best[0]).split())
            if len(snippet) > 140:
                snippet = snippet[:137] + '...'
            noun = "memory" if mentions == 1 else "memories"
            line = '- **%s** - %d %s since %s: "%s"' % (
                norm, mentions, noun, (first_seen or '')[:10], snippet)
            cost = len(line) + 1
            if used + cost > budget - len(header) - 1:
                break
            lines.append(line)
            used += cost
        if not lines:
            return ""
        return header + "\n" + "\n".join(lines)
    except Exception as e:
        logger.warning("recall trigger skipped: %s", e)
        return ""
