"""Secret scanner for quipu capture drain.

Detects credentials/secrets in content strings and prevents them from being
written to the store. Conservative: over-skip is acceptable; under-skip is not.

All patterns compiled at module level for performance.
"""

from __future__ import annotations

import math
import re

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# PEM private key block
_RE_PEM = re.compile(r"-----BEGIN\s+(?:\w+\s+)*PRIVATE KEY-----", re.IGNORECASE)

# OpenAI API key: sk- followed by 20+ alphanumeric chars
_RE_OPENAI = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")

# AWS Access Key ID
_RE_AWS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")

# AWS secret access key assignment
_RE_AWS_SECRET = re.compile(r"aws_secret_access_key\s*[=:]\s*\S{30,}", re.IGNORECASE)

# Bearer token (Authorization header style)
_RE_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}\b")

# JWT: eyJ...eyJ....  (header.payload.signature pattern)
_RE_JWT = re.compile(r"\beyJ[\w-]+\.eyJ[\w-]+\.[\w-]+\b")

# .env-style assignment with keyword gate: KEY=value (8+ char value)
# Matches KEY=VALUE assignments anywhere on a line — after 'export ', after
# leading whitespace, or at column 0. The keyword gate requires one of the
# sensitive keywords inside the variable name. No start-of-line anchor so
# "export API_KEY=value" and " SECRET_TOKEN=value" are caught.
# Keywords: KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD|API|CREDENTIAL|PRIVATE
_RE_DOTENV = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Z][A-Z0-9_]*)?(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD|API|CREDENTIAL|PRIVATE)"
    r"[A-Z0-9_]*\s*=\s*\S{8,}",
    re.IGNORECASE,
)

# Generic high-entropy hex string: 32+ hex chars
_RE_HEX32 = re.compile(r"\b[0-9a-fA-F]{32,}\b")

# Generic high-entropy base64 string: 40+ base64 chars
_RE_B64_40 = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")

# Entropy threshold (bits per character)
_ENTROPY_THRESHOLD = 3.5


def _shannon_entropy(s: str) -> float:
    """Compute Shannon entropy of a string in bits per character."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    entropy = 0.0
    for count in freq.values():
        p = count / n
        entropy -= p * math.log2(p)
    return entropy


def _has_high_entropy_hex(content: str) -> bool:
    """Return True if content contains a 32+ char hex string with entropy > threshold."""
    for m in _RE_HEX32.finditer(content):
        if _shannon_entropy(m.group()) > _ENTROPY_THRESHOLD:
            return True
    return False


def _has_high_entropy_b64(content: str) -> bool:
    """Return True if content contains a 40+ char base64 string with entropy > threshold."""
    for m in _RE_B64_40.finditer(content):
        token = m.group().rstrip("=")
        if len(token) >= 40 and _shannon_entropy(token) > _ENTROPY_THRESHOLD:
            return True
    return False


def looks_like_secret(content: str) -> bool:
    """Return True if content looks like it contains a credential/secret.

    Conservative: false positives (over-skip) are acceptable.
    Never logs or echoes the content — caller must NOT log content on True.

    Detectors (in order):
      1. PEM private key block
      2. OpenAI API key (sk-...)
      3. AWS AKIA access key ID
      4. AWS secret access key assignment
      5. Bearer token
      6. JWT (eyJ...eyJ...signature)
      7. .env-style KEY=value with secret keyword
      8. High-entropy hex32+ string
      9. High-entropy base64-40+ string
    """
    if not content:
        return False

    if _RE_PEM.search(content):
        return True

    if _RE_OPENAI.search(content):
        return True

    if _RE_AWS_KEY.search(content):
        return True

    if _RE_AWS_SECRET.search(content):
        return True

    if _RE_BEARER.search(content):
        return True

    if _RE_JWT.search(content):
        return True

    if _RE_DOTENV.search(content):
        return True

    if _has_high_entropy_hex(content):
        return True

    if _has_high_entropy_b64(content):
        return True

    return False
