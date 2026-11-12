"""
Hybrid retriever: dense (Chroma + MiniLM embeddings) and sparse (BM25) search,
fused with Reciprocal Rank Fusion (RRF).

- Dense search is good at paraphrased / conceptual questions.
- BM25 is good at exact names, terms and titles ("Zizek", "categorical imperative").
- RRF combines the two rankings without needing their scores to be comparable.
- Optionally, a cross-encoder reranks the top fused candidates (`rerank=True`).

When the user names specific episodes, a share of the result slots is reserved
for those episodes so that similar "neighbour" episodes can't crowd them out.
"""
import math
import pathlib
import sys
from typing import Dict, Iterable, List, Optional, Tuple

from .bm25_index import CHUNKS_PATH, get_bm25_index
from .chroma_store import embed_query, query as chroma_query
from .query_parser import parse_query
from .reranker import DEFAULT_RERANKER, rerank as rerank_fn

MODES = ("hybrid", "dense", "bm25")


class Retriever:
    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        collection_name="transcripts",
        persist_directory=None,
        chunks_path=None,
        candidate_pool=50,
        rrf_k=60,
        dense_weight=1.0,
        bm25_weight=1.0,
        max_per_episode=4,
        reserved_fraction=0.6,
        reranker_model=DEFAULT_RERANKER,
        rerank_pool=30,
    ):
        self.model_name = model_name
        self.collection_name = collection_name
        self.persist_directory = pathlib.Path(persist_directory) if persist_directory else None
        self.chunks_path = pathlib.Path(chunks_path) if chunks_path else CHUNKS_PATH
        self.candidate_pool = candidate_pool
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.max_per_episode = max_per_episode
        self.reserved_fraction = reserved_fraction
        self.reranker_model = reranker_model
        self.rerank_pool = rerank_pool
        self._rerank_failed = False  # set after a reranker load/run failure; later calls skip it
        self._docs: Dict[str, Dict] = {}  # chunk_id -> chunk record, filled from both sources

    # ----------------------------------------------------------- components
    @property
    def bm25(self):
        return get_bm25_index(self.chunks_path)

    def _dense_search(self, query_embedding, n, episodes) -> List[str]:
        try:
            result = chroma_query(
                query_text="",
                k=n,
                collection_name=self.collection_name,
                model_name=self.model_name,
                persist_directory=self.persist_directory,
                episode_numbers=sorted(episodes) if episodes else None,
                query_embedding=query_embedding,
            )
        except Exception as e:  # keep keyword search usable if Chroma is broken
            print(f"Warning: dense retrieval failed ({e}); falling back to BM25 only.", file=sys.stderr)
            return []

        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or [{} for _ in documents]
        ranked = []
        for chunk_id, doc, meta in zip(ids, documents, metadatas):
            if not isinstance(doc, str):
                continue
            if chunk_id not in self._docs:
                known = self.bm25.get(chunk_id)
                meta = meta if isinstance(meta, dict) else {}
                self._docs[chunk_id] = known or {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "title": meta.get("title"),
                    "episode_number": meta.get("episode_number"),
                    "url": meta.get("url"),
                    "chunk_index": 0,
                }
            ranked.append(chunk_id)
        return ranked

    def _bm25_search(self, text, n, episodes) -> List[str]:
        hits = self.bm25.search(text, top_n=n, allowed_episodes=episodes)
        ranked = []
        for chunk_id, _score in hits:
            if chunk_id not in self._docs:
                self._docs[chunk_id] = self.bm25.get(chunk_id)
            ranked.append(chunk_id)
        return ranked

    def _fused(
        self,
        text: str,
        query_embedding,
        episodes: Optional[Iterable[int]],
        mode: str,
        rerank: bool = False,
    ) -> List[Tuple[str, float, str]]:
        """Return [(chunk_id, score, sources)] best first.

        Score is the RRF score, or the cross-encoder score for the reranked head.
        """
        episodes = set(episodes) if episodes else None
        rankings = []
        if mode in ("hybrid", "dense"):
            rankings.append(("dense", self.dense_weight, self._dense_search(query_embedding, self.candidate_pool, episodes)))
        if mode in ("hybrid", "bm25"):
            rankings.append(("bm25", self.bm25_weight, self._bm25_search(text, self.candidate_pool, episodes)))

        scores: Dict[str, float] = {}
        sources: Dict[str, set] = {}
        for name, weight, ranked in rankings:
            for rank, chunk_id in enumerate(ranked):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (self.rrf_k + rank + 1)
                sources.setdefault(chunk_id, set()).add(name)

        ordered = sorted(scores, key=lambda cid: -scores[cid])
        fused = [
            (cid, scores[cid], "both" if len(sources[cid]) > 1 else next(iter(sources[cid])))
            for cid in ordered
        ]
        if rerank and fused:
            fused = self._rerank(text, fused)
        return fused

    def _rerank(self, text, fused):
        """Final stage: re-sort the top `rerank_pool` fused candidates with the cross-encoder."""
        if self._rerank_failed:
            return fused
        head, tail = fused[: self.rerank_pool], fused[self.rerank_pool:]
        src_of = {cid: src for cid, _s, src in head}
        try:
            reranked = rerank_fn(
                text,
                [(cid, self._docs[cid]["text"]) for cid, _s, _src in head],
                model_name=self.reranker_model,
            )
        except Exception as e:  # graceful fallback, like _dense_search
            print(f"Warning: reranking failed ({e}); using RRF order.", file=sys.stderr)
            self._rerank_failed = True
            return fused
        return [(cid, score, src_of[cid] + "+rerank") for cid, score in reranked] + tail

    def _spread_chunks(self, episode: int, n: int) -> List[str]:
        """Evenly spaced chunks across an episode (for topic-less 'summarise ep N')."""
        chunks = self.bm25.episode_chunks(episode)
        if not chunks:
            return []
        n = min(n, len(chunks))
        if n == 1:
            picks = [len(chunks) // 2]
        else:
            picks = sorted({round(i * (len(chunks) - 1) / (n - 1)) for i in range(n)})
        selected = []
        for i in picks:
            self._docs.setdefault(chunks[i]["chunk_id"], chunks[i])
            selected.append(chunks[i]["chunk_id"])
        return selected

    # ----------------------------------------------------------------- API
    def retrieve(
        self,
        query,
        k=5,
        episode_numbers=None,
        required_episodes=None,
        mode="hybrid",
        rerank=False,
        group_by_episode=True,
    ):
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query (str): The user query. Explicit episode references ("ep 12") are
                stripped before searching.
            k (int): Number of chunks to return.
            episode_numbers (list[int] | None): Optional filter for the candidate pool
                (e.g. the named episodes plus similar episodes).
            required_episodes (list[int] | None): Episodes the user explicitly named;
                a share of the k slots is reserved for them.
            mode (str): "hybrid" (default), "dense" or "bm25".
            rerank (bool): Re-sort the top `rerank_pool` fused candidates of every
                search pool with a cross-encoder (final retrieval stage).
            group_by_episode (bool): Group the output by episode and transcript order
                (nicer LLM context). False returns pure rank order (used by the eval).

        Returns:
            list[dict]: chunks with text, episode_number, title, chunk_id, url, score, sources.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

        parsed = parse_query(query)
        text = parsed.text or query
        required = [int(e) for e in (required_episodes or []) if e is not None]
        if required:
            known = set(int(e) for e in self.bm25.episode_numbers.tolist())
            required = [e for e in required if e in known] or required

        query_embedding = None
        if mode in ("hybrid", "dense"):
            try:
                query_embedding = embed_query(text, model_name=self.model_name)
            except Exception as e:
                print(f"Warning: query embedding failed ({e}); using BM25 only.", file=sys.stderr)
                mode = "bm25"

        selected: List[Tuple[str, float, str]] = []
        seen = set()

        def take(candidates, limit, per_episode_cap=None, counts=None):
            counts = counts if counts is not None else {}
            added = 0
            for cid, score, src in candidates:
                if added >= limit:
                    break
                if cid in seen:
                    continue
                ep = self._docs[cid].get("episode_number")
                if per_episode_cap is not None and counts.get(ep, 0) >= per_episode_cap:
                    continue
                seen.add(cid)
                counts[ep] = counts.get(ep, 0) + 1
                selected.append((cid, score, src))
                added += 1
            return added

        if required and not parsed.has_topic:
            # e.g. "summarise episode 120": no topic to search on, so cover the episode(s).
            per_ep = max(1, math.ceil(k / len(required)))
            for ep in required:
                for cid in self._spread_chunks(ep, per_ep):
                    if cid not in seen:
                        seen.add(cid)
                        selected.append((cid, 0.0, "episode"))
        else:
            counts: Dict[int, int] = {}
            if required:
                reserved = max(len(required), round(k * self.reserved_fraction))
                per_ep = max(1, math.ceil(reserved / len(required)))
                for ep in required:
                    take(self._fused(text, query_embedding, [ep], mode, rerank), per_ep, counts=counts)
            pool = episode_numbers or None
            cap = None if required else self.max_per_episode
            take(self._fused(text, query_embedding, pool, mode, rerank), k - len(selected), cap, counts)

        if group_by_episode:
            # Group by episode (in order of each episode's best hit), then transcript order.
            episode_rank: Dict[Optional[int], int] = {}
            for i, (cid, _s, _src) in enumerate(selected):
                episode_rank.setdefault(self._docs[cid].get("episode_number"), i)
            selected.sort(
                key=lambda item: (
                    episode_rank[self._docs[item[0]].get("episode_number")],
                    self._docs[item[0]].get("chunk_index", 0),
                )
            )

        items = []
        for cid, score, src in selected:
            doc = self._docs[cid]
            items.append(
                {
                    "text": doc["text"],
                    "episode_number": doc.get("episode_number"),
                    "title": doc.get("title"),
                    "chunk_id": cid,
                    "url": doc.get("url"),
                    "score": score,
                    "sources": src,
                }
            )
        return items
