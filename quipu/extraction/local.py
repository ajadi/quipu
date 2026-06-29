"""Local NLP extraction: entity and keyword extraction via stdlib regex only.

No external NLP libraries required. Deterministic, deduped, order-stable output.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Capitalized-word entity: one or more Title-case or mixed-case words in sequence.
_ENTITY_PATTERN = re.compile(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b')

# Stopwords to exclude from keywords
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "that", "this",
    "these", "those", "it", "its", "as", "if", "not", "no", "so", "yet",
    "both", "either", "neither", "each", "all", "any", "few", "more",
    "most", "other", "such", "than", "then", "just", "how", "what",
    "when", "where", "who", "which", "their", "there", "here", "i",
    "we", "you", "he", "she", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "also", "into", "up", "out", "about",
    "over", "after", "before", "between", "through", "during", "within",
    "without", "against", "across", "along", "among", "around", "behind",
    "below", "beside", "beyond", "inside", "outside", "since", "until",
    "upon", "above",
})

# Word token: alphabetic words, minimum length 3
_WORD_PATTERN = re.compile(r'\b[a-z]{3,}\b')


def _dedupe_stable(items: list[str]) -> list[str]:
    """Return items with duplicates removed, preserving first-occurrence order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_local(content: str) -> dict:
    """Extract entities and keywords from content using stdlib regex.

    Args:
        content: Input text string.

    Returns:
        dict with keys:
          - 'entities': list[str] — capitalized noun phrases, deduped, order-stable.
          - 'keywords': list[str] — significant lowercase words, deduped, order-stable.
    """
    if not content:
        return {"entities": [], "keywords": []}

    # --- Entity extraction ---
    # Capture sequences of capitalized words.
    raw_entities: list[str] = _ENTITY_PATTERN.findall(content)

    # Filter out single-word entities that are all-caps abbreviations of length <= 2
    # and very short entities — keep anything >= 2 chars.
    filtered_entities = [e for e in raw_entities if len(e) >= 2]

    entities = _dedupe_stable(filtered_entities)

    # --- Keyword extraction ---
    # Lowercase all, tokenize, remove stopwords, collect unique order-stable.
    lowered = content.lower()
    all_words = _WORD_PATTERN.findall(lowered)
    content_words = [w for w in all_words if w not in _STOPWORDS]
    keywords = _dedupe_stable(content_words)

    return {"entities": entities, "keywords": keywords}
