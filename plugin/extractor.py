"""Typed, atomic, user-grounded extraction for VERA Memory."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

if __package__:
    from .identity import normalize_transport
else:
    from identity import normalize_transport

EXTRACTOR_VERSION = "atomic-user-v2"
MAX_INPUT_CHARS = 4000
MAX_MEMORY_CHARS = 500

DECISION_PATTERNS = [
    r"\b(?:i|we)\s+(?:decided?|agreed?|chose?|confirmed?|approved?)\b",
    r"\b(?:going forward|from now on)\b",
    r"\bthe\s+(?:plan|decision|agreement|rule|policy)\s+(?:is|will be)\b",
]
FACT_PATTERNS = [
    r"\b(?:my|the)\s+(?:name|role|job|domain|default|address|number|timezone|pronouns|company|location|birthday|language)\s+(?:is|should be)\b",
    r"\b(?:we|i)\s+(?:use|have|require)\b",
    r"\bremember\s+(?:that|this)\b",
]
CORRECTION_PATTERNS = [
    r"\b(?:that(?:'s| is)|it(?:'s| is))\s+(?:wrong|incorrect|not right)\b",
    r"\b(?:actually|correction|i meant|the correct|the right|the actual)\b",
]
PREFERENCE_PATTERNS = [
    r"\bi\s+(?:prefer|like|need|want|don(?:'t|’t) like|hate)\b",
    r"\bmy\s+(?:favorite|preferred|default)\b",
    r"\b(?:always|never)\s+(?:use|call|address|send|explain|do)\b",
]

SENSITIVE_PATTERNS = [
    (r"(?i)\b(?:password|passwd|pwd)\s*(?::|=|\bis\b|\bwas\b|\bto be\b)\s*\S+", "[REDACTED_PASSWORD]"),
    (r"(?i)\b(?:api[_-]?key|apikey)\s*(?::|=|\bis\b|\bwas\b|\bto be\b)\s*\S+", "[REDACTED_API_KEY]"),
    (r"(?i)\b(?:token|secret|auth)\s*(?::|=|\bis\b|\bwas\b|\bto be\b)\s*\S+", "[REDACTED_TOKEN]"),
    (r"(?i)\b(?:token|secret|credential)\s+[A-Za-z0-9._~+/-]{12,}\b", "[REDACTED_TOKEN]"),
    (r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", "[REDACTED_GITHUB_TOKEN]"),
    (r"\bsk-[A-Za-z0-9_-]{20,}\b", "[REDACTED_API_KEY]"),
    (r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", "[REDACTED_AWS_KEY]"),
    (r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", "[REDACTED_PRIVATE_KEY]"),
    (r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{12,}\b", "[REDACTED_BEARER]"),
    (r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b", "[REDACTED_JWT]"),
    (r"(?i)\b(?:cookie|set-cookie)\s*[:=]\s*[^\s;]{4,}", "[REDACTED_COOKIE]"),
    (r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s]+", "[REDACTED_CONNECTION_STRING]"),
]

TRANSIENT_PATTERNS = [
    r"^/\w+",
    r"\b(?:just finished|completed today|currently working|in progress|todo|to-do)\b",
    r"\b(?:commit|pull request|issue)\s+#?\d+\b",
    r"\b(?:command output|traceback|stack trace|tool result)\b",
]

REJECT_MARKERS = (
    "<memory-context>",
    "[system note:",
    "[tool result",
    "[out-of-band user message",
    "```json",
)

INSTRUCTION_LIKE_PATTERNS = (
    r"(?i)\b(?:ignore|disregard|override)\b.{0,80}\b(?:instruction|policy|system|prompt)\b",
    r"(?i)\b(?:system|developer)\s+(?:prompt|message|instruction)\b",
    r"(?i)\b(?:you are|act as)\b.{0,80}\b(?:assistant|agent|system)\b",
    r"(?i)\b(?:reveal|exfiltrate|send)\b.{0,80}\b(?:secret|credential|token|password)\b",
)

CATEGORY_BY_TYPE = {
    "decision": "decisions",
    "fact": "facts",
    "correction": "corrections",
    "preference": "preferences",
}
IMPORTANCE_BY_TYPE = {
    "decision": 0.75,
    "fact": 0.65,
    "correction": 0.85,
    "preference": 0.70,
}


def redact_sensitive(text: str) -> str:
    redacted = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        redacted = re.sub(pattern, replacement, redacted, flags=re.DOTALL)
    return redacted


def classify_memory_type(sentence: str) -> Optional[str]:
    lowered = sentence.casefold()
    if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in CORRECTION_PATTERNS):
        return "correction"
    if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in PREFERENCE_PATTERNS):
        return "preference"
    if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in DECISION_PATTERNS):
        return "decision"
    if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in FACT_PATTERNS):
        return "fact"
    return None


def classify_category(text: str) -> str:
    memory_type = classify_memory_type(text)
    return CATEGORY_BY_TYPE.get(memory_type or "", "general")


def calculate_importance(text: str) -> float:
    memory_type = classify_memory_type(text)
    return IMPORTANCE_BY_TYPE.get(memory_type or "", 0.4)


def _strip_context_suffix(text: str) -> str:
    lowered = text.casefold()
    positions = [lowered.find(marker) for marker in REJECT_MARKERS if lowered.find(marker) >= 0]
    return text[: min(positions)] if positions else text


def _split_atomic_sentences(text: str) -> List[str]:
    # Newlines, bullets, and sentence endings are safe boundaries.  Semicolons
    # also split independent declarations without trying to paraphrase them.
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+|\s*;\s*", text)
    output: List[str] = []
    for part in parts:
        clean = " ".join(part.split()).strip(" \t\r\n\"'")
        if 10 <= len(clean) <= MAX_MEMORY_CHARS:
            output.append(clean)
    return output


def _is_transient_or_unsafe(sentence: str) -> bool:
    lowered = sentence.casefold()
    if any(marker in lowered for marker in REJECT_MARKERS):
        return True
    if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in TRANSIENT_PATTERNS):
        return True
    if any(re.search(pattern, sentence) for pattern in INSTRUCTION_LIKE_PATTERNS):
        return True
    if any(re.search(pattern, sentence, flags=re.DOTALL) for pattern, _ in SENSITIVE_PATTERNS):
        return True
    # V5 clean intake: rambly/low-information chat never becomes a memory.
    # Resolve the gate WITHOUT depending on sys.path/import mechanics:
    # relative -> absolute -> already-loaded alias -> fail closed.
    gate = None
    try:
        from .summarizer import is_fluent_declaration as _gate
        gate = _gate
    except (ImportError, ValueError):
        import sys as _sys
        for _name in ("vera_v5_summarizer", "summarizer"):
            _mod = _sys.modules.get(_name)
            if _mod is not None and hasattr(_mod, "is_fluent_declaration"):
                gate = _mod.is_fluent_declaration
                break
        if gate is None:
            try:
                from summarizer import is_fluent_declaration as _gate
                gate = _gate
            except ImportError:
                return True  # fail closed: unclassifiable intake is not stored
    if not gate(sentence):
        return True
    return False


def extract_facts_from_turn(
    user_message: str,
    assistant_message: str = "",
    *,
    platform: str = "telegram",
    sender_id: Optional[str] = None,
    sender_principal_id: Optional[int] = None,
    authenticated: bool = False,
) -> List[Dict[str, Any]]:
    """Extract durable atomic claims from the user message only.

    Assistant output is intentionally ignored: it is never evidence of what the
    owner declared.  Structured sender authentication controls trust; a legacy
    display envelope never increases authority.
    """
    del assistant_message  # Explicitly document that assistant text is ignored.
    raw = user_message if isinstance(user_message, str) else ""
    transport = normalize_transport(
        raw, platform=platform, sender_id=sender_id, authenticated=authenticated
    )
    content = _strip_context_suffix(transport.content).strip()
    if not content or len(content) > MAX_INPUT_CHARS:
        return []
    if content.casefold() in {"hi", "hello", "hey", "good morning", "thanks", "thank you"}:
        return []

    # The caller may already have resolved a trusted principal. A structured ID
    # never elevates an explicit unauthenticated assertion.
    is_authenticated = bool(authenticated and sender_principal_id)
    trust_tier = "owner_declared" if is_authenticated and sender_principal_id else "observed_chat"
    memories: List[Dict[str, Any]] = []
    seen = set()
    for sentence in _split_atomic_sentences(content):
        if _is_transient_or_unsafe(sentence):
            continue
        memory_type = classify_memory_type(sentence)
        if not memory_type:
            continue
        redacted = redact_sensitive(sentence)
        key = redacted.casefold()
        if key in seen:
            continue
        seen.add(key)
        memories.append(
            {
                "content": redacted,
                "category": CATEGORY_BY_TYPE[memory_type],
                "importance": IMPORTANCE_BY_TYPE[memory_type],
                "source": "conversation",
                "source_type": "authenticated_user" if trust_tier == "owner_declared" else "observed_chat",
                "memory_layer": "l1",
                "memory_type": memory_type,
                "subject_principal_id": sender_principal_id,
                "sender_principal_id": sender_principal_id,
                "trust_tier": trust_tier,
                # Unauthenticated observations remain auditable but cannot enter
                # automatic retrieval or prompt context.
                "retrieval_enabled": bool(is_authenticated and sender_principal_id),
                "extractor_version": EXTRACTOR_VERSION,
                "metadata": {
                    "transport": transport.identity_source,
                    "display_label_used_for_identity": False,
                },
            }
        )
    return memories
