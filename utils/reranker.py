"""
Cross-encoder reranking: the final stage of retrieval.

A cross-encoder reads the query and a chunk *together* and outputs a relevance
score. That is more accurate than comparing separately computed embeddings
(dense search) or counting shared words (BM25), but too slow to run over the
whole corpus, so it only re-sorts the top candidates from hybrid (RRF) search.
"""
from typing import List, Sequence, Tuple

from .model_singleton import get_cached_cross_encoder

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank(
    query: str,
    candidates: Sequence[Tuple[str, str]],
    model_name: str = DEFAULT_RERANKER,
    batch_size: int = 32,
) -> List[Tuple[str, float]]:
    """Score (chunk_id, text) candidates against the query.

    Returns [(chunk_id, score)] sorted best-first. Scores are raw cross-encoder
    logits: only their order is meaningful.
    """
    if not candidates:
        return []
    model = get_cached_cross_encoder(model_name)
    scores = model.predict(
        [(query, text) for _, text in candidates],
        batch_size=batch_size,
        show_progress_bar=False,
    )
    order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
    return [(candidates[i][0], float(scores[i])) for i in order]
