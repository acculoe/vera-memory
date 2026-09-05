"""Deterministic sender and transport normalization for VERA Memory.

Display labels are untrusted metadata.  A sender is resolved only by the
(platform, platform_subject_id) tuple against a pre-existing principal record.
Unknown senders are never enrolled by retrieval or extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Legacy Board Room envelope supplied by the gateway.  It is intentionally
# narrow: one line, bounded label, numeric id, and only at the start.
_ENVELOPE_RE = re.compile(
    r"^\[(?P<label>[^\]\r\n|]{1,128})\|(?P<subject_id>[0-9]{1,32})\]\s*"
)


@dataclass(frozen=True)
class NormalizedTransport:
    content: str
    platform: str
    platform_subject_id: Optional[str]
    display_label: Optional[str]
    identity_source: str
    authenticated: bool
    envelope_present: bool


def normalize_transport(
    text: str,
    *,
    platform: str = "telegram",
    sender_id: Optional[str] = None,
    display_name: Optional[str] = None,
    authenticated: Optional[bool] = None,
) -> NormalizedTransport:
    """Separate message content from transport identity metadata.

    Structured metadata wins over any legacy text envelope.  The envelope is
    removed from searchable content, but its label never becomes identity.
    A legacy subject id may resolve an already-provisioned principal; callers
    must not use it to auto-create one.
    """

    raw = text if isinstance(text, str) else ""
    match = _ENVELOPE_RE.match(raw)
    content = raw[match.end() :] if match else raw
    envelope_id = match.group("subject_id") if match else None
    envelope_label = match.group("label").strip() if match else None

    structured_id = str(sender_id).strip() if sender_id is not None else ""
    if structured_id:
        subject_id: Optional[str] = structured_id
        source = "structured"
        # An identifier is routing metadata, not proof of transport
        # authentication.  Only the trusted gateway assertion is authority.
        authenticated_value = bool(authenticated) if authenticated is not None else False
    elif envelope_id:
        subject_id = envelope_id
        source = "legacy_envelope"
        authenticated_value = False
    else:
        subject_id = None
        source = "unknown"
        authenticated_value = False

    label = display_name.strip() if isinstance(display_name, str) and display_name.strip() else envelope_label
    return NormalizedTransport(
        content=content.strip(),
        platform=(platform or "unknown").strip().lower(),
        platform_subject_id=subject_id,
        display_label=label,
        identity_source=source,
        authenticated=authenticated_value,
        envelope_present=bool(match),
    )
