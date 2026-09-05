"""Safe, relevance-gated retrieval for VERA Memory."""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Sequence

if __package__:
    from .identity import NormalizedTransport, normalize_transport
    from .models import MemoryDB, TRUST_WEIGHTS, normalize_content
else:  # Direct source-tree import used by the isolated regression harness.
    from identity import NormalizedTransport, normalize_transport
    from models import MemoryDB, TRUST_WEIGHTS, normalize_content

MAX_QUERY_CHARS = 2000
MAX_QUERY_TOKENS = 24
DEFAULT_TIMEOUT_MS = 150
NEUTRAL_TOKENS = {
    "hi", "hello", "hey", "yo", "sup", "morning", "afternoon", "evening",
    "good", "everyone", "everybody", "there", "bro", "lol",
}
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize_query(query: str) -> List[str]:
    """Return bounded Unicode word/number tokens safe for FTS quoting."""
    normalized = unicodedata.normalize("NFKC", query or "")[:MAX_QUERY_CHARS]
    tokens: List[str] = []
    seen = set()
    for raw in _TOKEN_RE.findall(normalized):
        token = raw.casefold()
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= MAX_QUERY_TOKENS:
            break
    return tokens


def _clean_fts_query(query: str) -> str:
    """Build an FTS expression from quoted tokens, never raw syntax."""
    tokens = tokenize_query(query)
    return " OR ".join(f'"{token}"' for token in tokens)


def _is_neutral_opener(tokens: Sequence[str]) -> bool:
    return bool(tokens) and len(tokens) <= 5 and all(token in NEUTRAL_TOKENS for token in tokens)


def _apply_layer_policy(
    rows: List[Dict[str, Any]], query_tokens: Sequence[str], limit: int
) -> List[Dict[str, Any]]:
    """Prefer concise L2/L3 summaries for broad bootstrap queries."""
    bounded = max(1, min(int(limit), 50))
    token_set = set(query_tokens)
    broad = bool(
        token_set & {"overview", "profile", "context", "summarize", "summary"}
        or "know" in token_set
        or {"about", "me"}.issubset(token_set)
    )
    if not broad:
        return rows[:bounded]
    summaries = [row for row in rows if row.get("memory_layer") in {"l2", "l3"}]
    if not summaries:
        return rows[:bounded]
    summaries.sort(
        key=lambda row: (
            1 if row.get("memory_layer") == "l3" else 0,
            row.get("fusion_score", row.get("relevance_score", 0.0)),
        ),
        reverse=True,
    )
    atoms = [row for row in rows if row.get("memory_layer") not in {"l2", "l3"}]
    return (summaries[:3] + atoms[:2])[:bounded]


def _score_result(result: Dict[str, Any], query_tokens: Sequence[str]) -> float:
    import math
    from datetime import datetime, timezone

    memory_tokens = set(tokenize_query(result.get("content", "")))
    query_set = set(query_tokens)
    overlap = len(query_set & memory_tokens)
    lexical = overlap / max(1, len(query_set))
    importance = max(0.0, min(float(result.get("importance", 0.5)), 1.0))
    trust = TRUST_WEIGHTS.get(result.get("trust_tier", "observed_chat"), 0.25)
    layer_bonus = {"l1": 0.04, "l2": 0.03, "l3": 0.05}.get(result.get("memory_layer"), 0.0)

    # V4: Temporal decay — recent memories score higher
    decay = 0.0
    created = result.get("created_at", "")
    if created:
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - created_dt).total_seconds() / 86400
            decay = math.exp(-0.01 * age_days) * 0.10  # Max 10% boost for brand-new
        except (ValueError, TypeError):
            pass

    # V4: Access count boost — frequently accessed = still relevant
    access_count = int(result.get("access_count", 0))
    access_boost = min(math.log(1 + access_count) * 0.03, 0.08)  # Max 8% boost

    score = lexical * 0.62 + importance * 0.13 + trust * 0.20 + layer_bonus + decay + access_boost
    return min(score, 1.0)


def search(
    db: MemoryDB,
    query: str,
    category: Optional[str] = None,
    limit: int = 10,
    *,
    min_score: float = 0.40,
    allowed_ids: Optional[Sequence[int]] = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    semantic_backend: Optional[Any] = None,
    principal_id: Optional[int] = None,
    authenticated: bool = False,
) -> List[Dict[str, Any]]:
    """Return relevant active memories or an empty list.

    SQLite authorizes every lexical and semantic candidate. Semantic failure,
    timeout, or absence degrades to lexical-only retrieval.
    """
    transport = normalize_transport(query)
    tokens = tokenize_query(transport.content)
    if not tokens or _is_neutral_opener(tokens):
        return []
    fts_query = " OR ".join(f'"{token}"' for token in tokens)
    started = time.perf_counter()
    candidates = db.search_memories(
        fts_query,
        category=category,
        limit=max(limit * 5, 25),
        allowed_ids=allowed_ids,
        principal_id=principal_id,
        authenticated=authenticated,
    )
    lexical: List[Dict[str, Any]] = []
    seen_hashes = set()
    for candidate in candidates:
        if (time.perf_counter() - started) * 1000 > max(1, timeout_ms):
            break
        memory_tokens = set(tokenize_query(candidate.get("content", "")))
        overlap = len(set(tokens) & memory_tokens)
        if overlap == 0:
            continue
        ratio = overlap / len(set(tokens))
        # Governance weights only sort an already relevant candidate.  For a
        # multi-term query, one incidental FTS OR hit is never sufficient.
        if len(set(tokens)) >= 2 and overlap < 2:
            continue
        score = _score_result(candidate, tokens)
        if score < min_score:
            continue
        digest = normalize_content(candidate.get("content", ""))
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        lexical.append({**candidate, "relevance_score": round(score, 6)})
    lexical.sort(
        key=lambda item: (
            item.get("relevance_score", 0.0),
            TRUST_WEIGHTS.get(item.get("trust_tier", "observed_chat"), 0.0),
            item.get("importance", 0.0),
        ),
        reverse=True,
    )

    if semantic_backend is None:
        return _apply_layer_policy(lexical, tokens, limit)

    # Apply SQLite lifecycle/category/profile filters before semantic search.
    authorized_rows = db.get_active_retrieval_rows(
        principal_id=principal_id, authenticated=authenticated
    )
    authorized_ids = {
        int(row["id"])
        for row in authorized_rows
        if (category is None or row["category"] == category)
    }
    if allowed_ids is not None:
        authorized_ids &= {int(value) for value in allowed_ids}
    remaining_seconds = (max(1, timeout_ms) / 1000.0) - (time.perf_counter() - started)
    if not authorized_ids or remaining_seconds <= 0:
        return _apply_layer_policy(lexical, tokens, limit)
    try:
        semantic_hits = semantic_backend.search(
            transport.content,
            allowed_ids=sorted(authorized_ids),
            limit=max(limit * 5, 25),
            timeout_seconds=remaining_seconds,
        )
    except Exception:
        return _apply_layer_policy(lexical, tokens, limit)

    # Validate the entire untrusted backend response inside the fallback
    # boundary. Any malformed item preserves lexical-only output.
    try:
        if not isinstance(semantic_hits, list):
            raise ValueError("semantic backend returned non-list hits")
        validated_hits: List[Dict[str, Any]] = []
        seen_semantic_ids = set()
        for hit in semantic_hits:
            if not isinstance(hit, dict) or type(hit.get("id")) is not int:
                raise ValueError("semantic backend returned invalid hit")
            memory_id = hit["id"]
            score = float(hit.get("score"))
            if memory_id not in authorized_ids or memory_id in seen_semantic_ids:
                continue
            if not (-1.0 <= score <= 1.0):
                raise ValueError("semantic backend returned invalid score")
            seen_semantic_ids.add(memory_id)
            validated_hits.append({"id": memory_id, "score": score})
        semantic_rows = {
            int(row["id"]): row
            for row in db.get_active_memories_by_ids(
                [hit["id"] for hit in validated_hits],
                principal_id=principal_id,
                authenticated=authenticated,
            )
        }
        semantic_ranked: List[Dict[str, Any]] = []
        for hit in validated_hits:
            row = semantic_rows.get(hit["id"])
            if row:
                semantic_ranked.append({**row, "semantic_score": hit["score"]})
    except Exception:
        return _apply_layer_policy(lexical, tokens, limit)

    # Reciprocal-rank fusion keeps lexical and semantic scales independent.
    fused: Dict[int, Dict[str, Any]] = {}
    rrf: Dict[int, float] = {}
    for rank, row in enumerate(lexical, start=1):
        memory_id = int(row["id"])
        fused[memory_id] = dict(row)
        rrf[memory_id] = rrf.get(memory_id, 0.0) + 1.0 / (60.0 + rank)
    for rank, row in enumerate(semantic_ranked, start=1):
        memory_id = int(row["id"])
        fused[memory_id] = {**fused.get(memory_id, {}), **row}
        rrf[memory_id] = rrf.get(memory_id, 0.0) + 1.0 / (60.0 + rank)

    output: List[Dict[str, Any]] = []
    content_seen = set()
    for memory_id, row in fused.items():
        digest = normalize_content(row.get("content", ""))
        if not digest or digest in content_seen:
            continue
        content_seen.add(digest)
        trust = TRUST_WEIGHTS.get(row.get("trust_tier", "observed_chat"), 0.0)
        importance = max(0.0, min(float(row.get("importance", 0.5)), 1.0))
        fusion_score = rrf[memory_id] * 40.0 + trust * 0.20 + importance * 0.10
        output.append({**row, "fusion_score": round(fusion_score, 6)})
    output.sort(
        key=lambda item: (
            item.get("fusion_score", 0.0),
            TRUST_WEIGHTS.get(item.get("trust_tier", "observed_chat"), 0.0),
            item.get("importance", 0.0),
        ),
        reverse=True,
    )
    return _apply_layer_policy(output, tokens, limit)


def format_for_context(memories: List[Dict[str, Any]], max_chars: int = 3000) -> str:
    if not memories:
        return ""
    lines: List[str] = []
    used = 0
    for memory in memories:
        content = " ".join(str(memory.get("content", "")).split())
        if not content:
            continue
        category = memory.get("category", "general")
        created = str(memory.get("created_at", ""))[:10]
        trust = memory.get("trust_tier", "unverified")
        # Retrieved text is evidence, never executable prompt authority.
        entry = f"- [{category}; evidence/{trust}] > {content}"
        if created:
            entry += f" ({created})"
        # V5 fix: an oversized TOP entry used to `break` here, returning an
        # empty block whenever the best hit exceeded the whole budget.
        # Salvage by truncating into remaining space; skip only when nothing
        # meaningful fits, so later shorter entries can still fill the block.
        remaining = max_chars - used
        if len(entry) <= remaining:
            lines.append(entry)
            used += len(entry)
            continue
        if remaining >= 80:
            lines.append(entry[: remaining - 1].rstrip() + "…")
            used += remaining
            break  # budget fully consumed
        continue  # too little space left; try a shorter entry
    return "\n".join(lines)


def _identity_context(
    db: MemoryDB,
    transport: NormalizedTransport,
    context_scope: str,
    resolved_principal: Optional[Dict[str, Any]] = None,
) -> str:
    # V5 fix: use caller-resolved principal when provided; re-resolving from
    # raw transport fields fails closed without a sender_id.
    principal = (
        resolved_principal
        if resolved_principal is not None
        else db.resolve_principal(transport.platform, transport.platform_subject_id)
    )
    if not principal:
        return ""
    identity = db.resolve_identity(principal["id"], context_scope=context_scope)
    if not identity:
        return ""
    fields = []
    if identity.get("reference_name"):
        fields.append(f"Canonical reference: {identity['reference_name']}")
    if identity.get("direct_address"):
        fields.append(f"Direct address: {identity['direct_address']}")
    if not fields:
        return ""
    # The display label and raw platform id are intentionally omitted.
    return "## Verified Identity\n" + "\n".join(f"- {field}" for field in fields)


def build_context_block(
    db: MemoryDB,
    query: str,
    max_chars: int = 500,
    *,
    platform: str = "telegram",
    sender_id: Optional[str] = None,
    display_name: Optional[str] = None,
    sender_authenticated: bool = False,
    context_scope: str = "default",
    semantic_backend: Optional[Any] = None,
    resolved_principal: Optional[Dict[str, Any]] = None,
) -> str:
    transport = normalize_transport(
        query,
        platform=platform,
        sender_id=sender_id,
        display_name=display_name,
        authenticated=sender_authenticated,
    )
    # V5 fix: prefer the caller-resolved principal (from trusted provider
    # state). Re-resolving from raw transport fields fails closed when the
    # runtime supplies no sender_id (agent-initiated calls), which made
    # vera_context abstain on every topic.
    if resolved_principal is not None:
        principal = resolved_principal
    else:
        principal = (
            db.resolve_principal(transport.platform, transport.platform_subject_id)
            if transport.authenticated else None
        )
    identity_block = _identity_context(db, transport, context_scope, resolved_principal=principal) if principal else ""

    # V4: Always inject profile + recent context, even for neutral openers
    tokens = tokenize_query(transport.content)
    is_neutral = _is_neutral_opener(tokens)

    if is_neutral:
        # Neutral opener: inject profile + recent sessions only
        memories = []
        # Get recent session summaries (L2/L3)
        try:
            recent = db.search_memories(
                '"session summary" OR "profile"',
                category=None,
                limit=5,
                principal_id=principal["id"] if principal else None,
                authenticated=bool(principal),
            )
            memories = [r for r in recent if r.get("memory_layer") in {"l2", "l3"}][:3]
        except Exception:
            pass
    else:
        # Normal message: full semantic search
        memories = search(
            db,
            transport.content,
            limit=10,
            semantic_backend=semantic_backend,
            principal_id=principal["id"] if principal else None,
            authenticated=bool(principal),
        )

    memory_text = format_for_context(memories, max_chars=max_chars)
    sections: List[str] = []
    if identity_block:
        sections.append(identity_block)
    if memory_text:
        sections.append(
            f"## VERA Memory ({db.get_memory_count()} total memories)\n{memory_text}"
        )
    if not sections:
        return ""
    return "\n" + "\n\n".join(sections) + "\n"
