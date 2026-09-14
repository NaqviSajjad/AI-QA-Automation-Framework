"""Shared text normalisation.

Retrieval, the mock generator and the evaluators must all agree on what a
"word" and a "sentence" are. When they disagree - for example if one of them
treats a hard-wrapped markdown line as a sentence - metrics move for reasons
that have nothing to do with the AI under test. Keeping this in one module is
what makes the scores reproducible.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_BULLET_RE = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)

STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
        "were", "be", "been", "it", "this", "that", "with", "as", "at", "by", "from",
        "you", "your", "we", "our", "i", "my", "me", "if", "will", "can", "do", "does",
        "did", "have", "has", "had", "so", "but", "not", "there", "they", "them",
        "please", "any", "about", "would", "could", "should", "what", "why", "how",
        "when", "which", "get", "than", "then", "their", "its", "his", "her",
    }
)


# Domain word families that suffix stripping alone will not join. Customers and
# knowledge-base articles reliably use different members of each family
# ("why does my bank ask me to verify" vs "the customer sees a verification
# step"), and a lexical scorer that misses the link retrieves the wrong article.
LEMMAS: dict[str, str] = {
    "verify": "verif", "verifies": "verif", "verified": "verif",
    "verification": "verif", "verifying": "verif",
    "authenticate": "authentic", "authentication": "authentic",
    "authorise": "authoris", "authorize": "authoris",
    "authorisation": "authoris", "authorization": "authoris",
    "pay": "pay", "pays": "pay", "paid": "pay", "paying": "pay", "payment": "pay",
    "payments": "pay", "payable": "pay",
    "cancel": "cancel", "cancelled": "cancel", "canceled": "cancel",
    "cancellation": "cancel", "cancelling": "cancel",
    "subscribe": "subscript", "subscription": "subscript", "subscriptions": "subscript",
    "renew": "renew", "renewal": "renew", "renewals": "renew", "renews": "renew",
    "delete": "delet", "deleted": "delet", "deletion": "delet", "deleting": "delet",
    "refund": "refund", "refunds": "refund", "refunded": "refund", "refundable": "refund",
    "escalate": "escalat", "escalation": "escalat", "escalated": "escalat",
    "prorated": "prorat", "proration": "prorat",
}


def stem(word: str) -> str:
    """Light, predictable suffix stripping plus a small domain lemma table.

    Maps charge / charged / charges -> "charg" so that a customer writing
    "I was charged twice" reaches the knowledge-base section headed
    "Duplicate charges". A full stemmer would be heavier and less predictable.
    """
    if word in LEMMAS:
        return LEMMAS[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) - len(suffix) >= 3 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    if len(word) >= 5 and word.endswith("e"):
        word = word[:-1]
    return word


def tokenize(text: str, min_length: int = 2) -> list[str]:
    """Content words only, stemmed."""
    return [
        stem(token)
        for token in _TOKEN_RE.findall((text or "").lower())
        if token not in STOPWORDS and len(token) >= min_length
    ]


def token_set(text: str, min_length: int = 3) -> set[str]:
    return set(tokenize(text, min_length=min_length))


def normalize_whitespace(text: str) -> str:
    """Collapse hard-wrapped lines inside a paragraph into one line."""
    text = _BULLET_RE.sub("", text or "")
    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in _PARAGRAPH_RE.split(text)]
    return "\n\n".join(p for p in paragraphs if p)


def split_sentences(text: str, min_length: int = 20) -> list[str]:
    """Real sentences - line wrapping in the source markdown is not a boundary."""
    normalised = normalize_whitespace(text)
    sentences: list[str] = []
    for paragraph in normalised.split("\n\n"):
        for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", paragraph):
            candidate = piece.strip(" -•\t")
            if len(candidate) >= min_length:
                sentences.append(candidate)
    return sentences


def overlap_ratio(candidate: str, reference: str, min_length: int = 3) -> float:
    """Share of the candidate's content words present in the reference."""
    left = token_set(candidate, min_length)
    if not left:
        return 0.0
    return len(left & token_set(reference, min_length)) / len(left)
