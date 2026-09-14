"""Built-in reference evaluators.

These implement the *same metric definitions* DeepEval and RAGAS use - claim
extraction for faithfulness, question-coverage for answer relevancy, rank-aware
precision and reference-coverage recall for retrieval - but with a deterministic
lexical/embedding scorer instead of an LLM judge.

Why that trade is deliberate:

* an LLM judge costs money and returns a slightly different number every run,
  which makes it a poor *gate*;
* CI must run with no credentials;
* when the judge and the deterministic scorer disagree, that disagreement is
  itself a signal worth looking at.

The limitation is stated plainly: these scorers measure lexical grounding and
coverage, not deep semantics. For nightly runs with a real key, set
`EVALUATOR_BACKEND=deepeval` (or `ragas`) and the same call sites use the
model-based implementations instead. See docs/ai-evaluation.md.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from src.rag.embeddings import DeterministicEmbeddings, cosine_similarity
from src.utils.text import overlap_ratio, split_sentences, token_set, tokenize

_EMBEDDINGS = DeterministicEmbeddings()

# Sentences that open with these are courtesy, intent or questions - they are
# not factual assertions, so they are not scored for faithfulness.
_NON_CLAIM_OPENERS = (
    "thanks", "thank you", "i can", "i am sorry", "i'm sorry", "as a next step",
    "could you", "would you", "please", "let me know", "if you", "i want to",
    "i will follow up", "i am handing", "i do not have", "i don't have",
    "i have not been able", "i am not able", "you will get a reply",
    "your conversation reference", "i am not able to share", "i will not",
)


def _tokens(text: str) -> list[str]:
    return tokenize(text, min_length=3)


def _token_set(text: str) -> set[str]:
    return token_set(text, min_length=3)


def _sentences(text: str) -> list[str]:
    return split_sentences(text, min_length=16)


def _overlap(candidate: str, reference: str) -> float:
    """Share of the candidate's content words that appear in the reference."""
    return overlap_ratio(candidate, reference, min_length=3)


def _similarity(left: str, right: str) -> float:
    return cosine_similarity(_EMBEDDINGS.embed_query(left), _EMBEDDINGS.embed_query(right))


@lru_cache(maxsize=1)
def _corpus_idf() -> dict[str, float]:
    """Term importance measured against the support corpus.

    "trial" matters in a question; "more" and "time" do not. Weighting question
    terms by inverse document frequency stops a short question from being scored
    on its filler words.
    """
    try:
        from src.rag.loader import load_knowledge_base

        documents = load_knowledge_base()
    except Exception:  # pragma: no cover - corpus optional for pure unit use
        return {}
    total = len(documents)
    frequency: dict[str, int] = {}
    for document in documents:
        for term in set(_tokens(document.page_content)):
            frequency[term] = frequency.get(term, 0) + 1
    return {term: math.log((1 + total) / (1 + df)) + 1.0 for term, df in frequency.items()}


def _term_weight(term: str) -> float:
    idf = _corpus_idf()
    if not idf:
        return 1.0
    # An unseen term is at least as distinctive as the rarest known one.
    return idf.get(term, max(idf.values()))


def _weighted_coverage(question_terms: set[str], answer_terms: set[str]) -> float:
    if not question_terms:
        return 1.0
    total = sum(_term_weight(t) for t in question_terms)
    covered = sum(_term_weight(t) for t in question_terms & answer_terms)
    return covered / total if total else 1.0


def _strip_discourse_prefix(sentence: str) -> str:
    """Drop a short leading clause such as "Looking at the records on your account,".

    The framing is the assistant's own words; the claim to be checked is what
    follows it. Scoring the framing against the evidence just adds noise.
    """
    head, sep, tail = sentence.partition(",")
    if sep and len(head.split()) <= 8 and tail.strip():
        return tail.strip()
    return sentence


def extract_claims(answer: str) -> list[str]:
    """Factual assertions only - courtesy, offers and questions are excluded."""
    claims = []
    for sentence in _sentences(answer):
        lowered = sentence.lower()
        if sentence.rstrip().endswith("?"):
            continue
        if lowered.startswith(_NON_CLAIM_OPENERS):
            continue
        if len(_tokens(sentence)) < 4:
            continue
        claims.append(sentence)
    return claims


@dataclass
class MetricScore:
    name: str
    score: float
    threshold: float
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    lower_is_better: bool = False

    @property
    def passed(self) -> bool:
        if self.lower_is_better:
            return self.score <= self.threshold
        return self.score >= self.threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.name,
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "lower_is_better": self.lower_is_better,
            "passed": self.passed,
            "reason": self.reason,
        }


# --------------------------------------------------------------------------- #
# Generation metrics
# --------------------------------------------------------------------------- #
def answer_relevancy(
    question: str, answer: str, threshold: float = 0.85, topic: str | None = None
) -> MetricScore:
    """Does the answer address *this* question, and does it stay on topic?

    Three components, all observable:
      coverage    - how much of the question's content is picked up;
      on_topic    - share of answer sentences connected to the question or topic;
      actionable  - support answers must end with a next step.

    `topic` (the classified intent, e.g. "billing") widens the on-topic test:
    "I can help investigate this billing issue" is a relevant sentence even
    though it repeats none of the customer's words.
    """
    question_terms = _token_set(question)
    answer_terms = _token_set(answer)
    if not answer.strip():
        return MetricScore("answer_relevancy", 0.0, threshold, "empty answer")

    covered = _weighted_coverage(question_terms, answer_terms)
    coverage = min(1.0, covered / 0.5)  # picking up the important half is full marks

    # A good answer does not have to echo the whole question; it has to contain a
    # sentence that answers it. `directness` rewards that, so an answer is not
    # penalised for skipping filler words like "send me a copy of".
    best_sentence = max(
        (_weighted_coverage(question_terms, _token_set(s)) for s in _sentences(answer)),
        default=0.0,
    )
    coverage = max(coverage, min(1.0, best_sentence / 0.45))

    topic_terms = set(question_terms)
    if topic:
        topic_terms |= _token_set(topic)

    # Topic drift is measured over factual sentences only. Penalising the greeting
    # and the closing next step would score answer *structure*, not relevance -
    # and every well-formed support answer has both.
    claims = extract_claims(answer)
    if claims:
        on_topic_hits = sum(
            1 for s in claims if (_token_set(s) & topic_terms) or _similarity(s, question) >= 0.15
        )
        on_topic = on_topic_hits / len(claims)
    else:
        on_topic = 1.0  # a pure clarification or refusal asserts nothing off-topic

    actionable = 1.0 if re.search(
        r"next step|could you|would you like|i can |please confirm|i will", answer, re.IGNORECASE
    ) else 0.0

    score = 0.45 * coverage + 0.40 * on_topic + 0.15 * actionable
    return MetricScore(
        "answer_relevancy",
        round(score, 4),
        threshold,
        f"coverage={coverage:.2f} on_topic={on_topic:.2f} actionable={actionable:.0f}",
        {"coverage": coverage, "on_topic": on_topic, "actionable": actionable},
    )


def faithfulness(
    answer: str, contexts: list[str], threshold: float = 0.85
) -> MetricScore:
    """Share of factual claims in the answer that the context supports.

    An answer that makes no factual claims (a pure clarification, or an honest
    "I don't know") is faithful by definition - it asserts nothing false.
    """
    claims = extract_claims(answer)
    if not claims:
        return MetricScore(
            "faithfulness", 1.0, threshold, "no factual claims asserted", {"claims": 0}
        )
    if not contexts:
        return MetricScore(
            "faithfulness", 0.0, threshold, f"{len(claims)} claims with no supporting context",
            {"claims": len(claims), "supported": 0},
        )

    supported, unsupported = 0, []
    for claim in claims:
        normalised = _strip_discourse_prefix(claim)
        best = max(
            max(_overlap(normalised, ctx), _similarity(normalised, ctx)) for ctx in contexts
        )
        if best >= 0.55:
            supported += 1
        else:
            unsupported.append(claim[:80])

    score = supported / len(claims)
    return MetricScore(
        "faithfulness",
        round(score, 4),
        threshold,
        f"{supported}/{len(claims)} claims grounded"
        + (f"; unsupported: {unsupported}" if unsupported else ""),
        {"claims": len(claims), "supported": supported, "unsupported": unsupported},
    )


def hallucination(answer: str, contexts: list[str], max_score: float = 0.20) -> MetricScore:
    """Inverse of faithfulness, with a hard override for forbidden claims."""
    from src.evaluation.deterministic import find_forbidden_claims

    forbidden = find_forbidden_claims(answer)
    if forbidden:
        return MetricScore(
            "hallucination", 1.0, max_score, f"forbidden claim: {forbidden}", lower_is_better=True
        )
    faith = faithfulness(answer, contexts)
    return MetricScore(
        "hallucination", round(1.0 - faith.score, 4), max_score, faith.reason, faith.details,
        lower_is_better=True,
    )


def correctness(answer: str, reference: str, threshold: float = 0.80) -> MetricScore:
    """Semantic correctness against a reference behaviour, not a literal string.

    Scored as recall of the reference's content words plus embedding similarity,
    so a paraphrase that conveys the same thing still passes - which is the whole
    point of evaluating an LLM instead of string-matching it.
    """
    if not answer.strip():
        return MetricScore("correctness", 0.0, threshold, "empty answer")
    reference_terms = _token_set(reference)
    if not reference_terms:
        return MetricScore("correctness", 1.0, threshold, "no reference supplied")

    recall = len(reference_terms & _token_set(answer)) / len(reference_terms)
    recall_scaled = min(1.0, recall / 0.6)
    similarity = min(1.0, _similarity(answer, reference) / 0.5)
    score = 0.65 * recall_scaled + 0.35 * similarity
    return MetricScore(
        "correctness",
        round(score, 4),
        threshold,
        f"reference_recall={recall:.2f} similarity={similarity:.2f}",
        {"recall": recall, "similarity": similarity},
    )


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #
def contextual_relevancy(question: str, contexts: list[str], threshold: float = 0.75) -> MetricScore:
    """Share of retrieved chunks that are actually about the question."""
    if not contexts:
        return MetricScore("contextual_relevancy", 0.0, threshold, "nothing retrieved")
    relevant = sum(
        1 for ctx in contexts
        if _overlap(question, ctx) >= 0.25 or _similarity(question, ctx) >= 0.20
    )
    score = relevant / len(contexts)
    return MetricScore(
        "contextual_relevancy", round(score, 4), threshold,
        f"{relevant}/{len(contexts)} chunks relevant",
    )


def contextual_precision(
    sources: list[str], expected_source: str | set[str], threshold: float = 0.80
) -> MetricScore:
    """Rank-aware precision (mean average precision over the retrieved list).

    MAP rewards getting the right article high in the ranking without demanding
    that every slot be relevant - with a six-article corpus and top_k=4, some
    neighbouring content in the tail is normal and not a defect. A metric that
    punishes normal behaviour teaches people to raise the threshold, and then it
    stops protecting anything.
    """
    relevant = {expected_source} if isinstance(expected_source, str) else set(expected_source)
    if not sources:
        return MetricScore("contextual_precision", 0.0, threshold, "nothing retrieved")

    hits, precisions = 0, []
    for index, source in enumerate(sources, start=1):
        if source in relevant:
            hits += 1
            precisions.append(hits / index)
    score = sum(precisions) / len(precisions) if precisions else 0.0
    return MetricScore(
        "contextual_precision", round(score, 4), threshold,
        f"relevant {sorted(relevant)} in ranked sources {sources}",
        {"hits": hits, "retrieved": len(sources)},
    )


def contextual_recall(
    reference_context: str, contexts: list[str], threshold: float = 0.80
) -> MetricScore:
    """Share of the reference context that the retrieved chunks actually cover."""
    reference_terms = _token_set(reference_context)
    if not reference_terms:
        return MetricScore("contextual_recall", 1.0, threshold, "no reference context supplied")
    retrieved_terms = _token_set(" ".join(contexts))
    score = len(reference_terms & retrieved_terms) / len(reference_terms)
    return MetricScore(
        "contextual_recall", round(min(1.0, score / 0.8), 4), threshold,
        f"raw_term_recall={score:.2f}",
    )


# --------------------------------------------------------------------------- #
# Agent metrics
# --------------------------------------------------------------------------- #
def tool_correctness(expected: str | None, actual: str | None, threshold: float = 0.90) -> MetricScore:
    match = (expected or None) == (actual or None)
    return MetricScore(
        "tool_correctness", 1.0 if match else 0.0, threshold,
        f"expected '{expected}', got '{actual}'",
    )


def task_completion(
    answer: str,
    expected_behavior: str,
    *,
    escalated: bool = False,
    expected_escalation: bool = False,
    threshold: float = 0.85,
) -> MetricScore:
    """Did the workflow actually achieve the support outcome?

    Completion is more than a relevant answer: the right intent, an actionable
    next step, and the right escalation decision.
    """
    if not answer.strip():
        return MetricScore("task_completion", 0.0, threshold, "no answer produced")
    behaviour = correctness(answer, expected_behavior).score
    actionable = 1.0 if re.search(
        r"next step|i can |could you|would you like|please confirm|handing this over",
        answer, re.IGNORECASE
    ) else 0.0
    escalation_ok = 1.0 if escalated == expected_escalation else 0.0
    score = 0.55 * behaviour + 0.20 * actionable + 0.25 * escalation_ok
    return MetricScore(
        "task_completion", round(score, 4), threshold,
        f"behaviour={behaviour:.2f} actionable={actionable:.0f} escalation_ok={escalation_ok:.0f}",
    )
