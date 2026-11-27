"""Baselines for comparison with T-RAG. They reuse retrieve() with parts switched off,
so they use the same Chroma index and embeddings (no extra embedding work).

  plain   : similarity only (no as-of filter, no window, no decay, no monthly buckets)
  recency : similarity + time decay, still no as-of filter, no window, no buckets

The LLM sees only the retrieved articles. No tools, no computed numbers, no verifier.
The raw question is embedded (the parser's search phrase is part of T-RAG, not of the baseline).
"""
from src.config import *
from src.retrieval import retrieve, count_leaks
from src.humanizer import get_client, DEFAULT_MODEL

SWITCHES = {
    "plain":   dict(use_asof=False, use_window=False, use_decay=False, use_buckets=False),
    "recency": dict(use_asof=False, use_window=False, use_decay=True,  use_buckets=False),
}

SYSTEM = ("Answer the question using the articles provided as context. "
          "If the context is not enough, say so.")


def rag_baseline(question, params, ctx, variant="plain", k=10, use_ticker=True,
                 max_chars=800, llm=DEFAULT_MODEL):
    """Returns {'answer', 'retrieved' (DataFrame), 'n_after_as_of'}.
    use_ticker=True keeps the ticker metadata filter so the baseline is reasonable;
    set False for a fully unfiltered similarity search."""
    res = retrieve(question, params.ticker if use_ticker else None,
                   params.start, params.end, params.as_of,
                   col=ctx.col, model=ctx.model, sent=None, k_total=k, **SWITCHES[variant])
    context = "\n\n".join(f"[{i + 1}] {r.date} | {r.title}\n{r.text[:max_chars]}"
                          for i, r in enumerate(res.itertuples()))
    resp = get_client().chat.completions.create(
        model=llm, temperature=0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}])
    return {"answer": resp.choices[0].message.content.strip(), "retrieved": res,
            "n_after_as_of": count_leaks(res, params.as_of)}