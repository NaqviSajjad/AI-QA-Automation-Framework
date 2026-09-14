# RAG testing

## The one question worth answering

When a RAG answer is wrong, there are only two possibilities:

```
The retriever did not find the right context.      -> RETRIEVAL
The generator had the right context and ignored it. -> GENERATION
```

"The answer was wrong" is not an actionable bug report. "The retriever returned
`subscriptions.md` for a refund question" is. Everything in this suite is arranged to
produce the second kind of sentence.

## The pipeline under test

```
knowledge-base/*.md
      -> loader        one document per article, provenance in metadata
      -> chunker       heading-aware first, size-aware second
      -> embeddings    deterministic hashed bag-of-words (or OpenAI)
      -> vector store  TF-IDF weighted, FAISS or exact numpy
      -> retriever     top-k, absolute floor, relative floor
      -> LLM           grounded generation
```

Three implementation choices that exist because of testability:

**Heading-aware chunking.** A policy statement and its conditions must stay in the same
chunk. The heading is prefixed as its own sentence so it contributes to the embedding
without later being mistaken for prose.

**IDF weighting.** Raw term counts let a chunk that happens to repeat a common word out-
rank the chunk that answers the question — "charge" appears eleven times in the
duplicate-charge section. Inverse document frequency over the hashed feature space fixes
that, and is computed from this corpus only, so it stays deterministic.

**Two score floors.** An absolute floor, and a relative one at 45% of the best hit. Fixed
top-k always returns k chunks whether or not k chunks are relevant, which pads the context
with near-misses, drags contextual relevancy down, and — worse — means "no relevant
context" can never occur, so an honest "I don't know" can never be triggered.

## Retrieval tests (`tests/rag/test_retrieval.py`)

Nothing here looks at a generated answer.

* the knowledge base loads every article, with provenance;
* chunking preserves headings and respects the configured size;
* embeddings are byte-identical across runs (reproducibility is a testability property);
* unrelated texts are not similar;
* an empty store raises rather than returning nothing quietly;
* **the expected article is retrieved** for every dataset case;
* rank-aware precision and reference recall meet their thresholds;
* results are ordered by score and ranks are consistent;
* `top_k` is respected;
* the similarity floor removes noise — asserted with a nonsense query;
* retrieval is stable across repeated calls.

Ground truth for precision is a **set** (`expected_source` + `acceptable_sources`).
A question about mid-cycle plan changes is legitimately answered from both
`subscriptions.md` and `billing.md`, and a single-label ground truth would mark correct
retrieval as a failure.

Deliberately vague first turns ("I have a payment problem") are excluded: there is no
single right article to retrieve yet — that is the point of the follow-up.

## Generation tests (`tests/rag/test_rag_quality.py`)

Scored separately, and the diagnosis is printed in the failure message:

```python
def diagnose_rag_failure(retrieval_ok, generation_ok):
    if not retrieval_ok and not generation_ok:
        return "RETRIEVAL (generation cannot be judged on the wrong context)"
    if not retrieval_ok:
        return "RETRIEVAL"
    if not generation_ok:
        return "GENERATION"
    return "OK"
```

A failing case therefore reports:

```
CS-021 failure origin: RETRIEVAL
  retrieval:  0.250 (1/4 chunks relevant)
  generation: 1.000 (3/3 claims grounded)
  sources: ['accounts.md', 'subscriptions.md', 'technical_support.md']
```

The generator was fine. Do not touch the prompt.

One test deserves highlighting: `test_generation_is_scored_against_retrieved_context_not_
the_whole_corpus`. A claim that is true elsewhere in the knowledge base but absent from
*this* answer's context is still unfaithful. That is what faithfulness means, and getting
it wrong is how a RAG evaluation quietly stops detecting anything.

## No-context behaviour

The most interesting case is when retrieval legitimately finds nothing. The correct
answer is to say so:

```
Thanks for getting in touch. I do not have enough information in our support knowledge
base to answer that reliably... I have not been able to find any documented policy or
record about XYZ-999, and I will not guess at one. I can pass this to a human support
specialist...
```

The generator detects two situations: nothing relevant retrieved, and the customer naming
a specific identifier the context never mentions. The second matters — answering "what is
the policy for XYZ-999" from the *general* refund article is exactly how a support bot
invents a policy.

The entity detector is narrow on purpose: a single token mixing letters and digits
(`XYZ-999`, `INV-77777`), or a multi-word proper noun whose components are absent from the
context. An earlier, looser version matched "returns 429" and turned every status-code
question into a refusal.

## With RAGAS

`EVALUATOR_BACKEND=ragas` (plus `.[eval]` and a key) swaps in RAGAS for the same call
sites:

| RAGAS | Diagnoses |
|---|---|
| `context_precision`, `context_recall` | the retriever |
| `faithfulness`, `answer_relevancy` | the generator |

Same split, model-judged rather than lexical. `src/evaluation/ragas_metrics.py` runs the
suite over a batch and maps the results onto the framework's `MetricScore`.
