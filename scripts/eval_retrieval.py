"""
Three-stage retrieval evaluation on a labelled query set (data/eval_queries.jsonl):

    Stage 1  dense           Chroma (MiniLM) vector search only
    Stage 2  hybrid          dense + BM25, fused with Reciprocal Rank Fusion
    Stage 3  hybrid_rerank   hybrid, then a cross-encoder reranks the top fused candidates

All stages share query parsing, named-episode slot reservation and the per-episode
cap, so the only difference between them is the ranking step.

Each line of the query file: {"group": str, "query": str, "episodes": [int, ...]}
Relevance is episode-level: a chunk is relevant if it comes from a listed episode.

Metrics (computed on pure rank order, i.e. `group_by_episode=False`):
  - P@5 / P@10   share of the top-5/10 chunks from a relevant episode (context quality)
  - Hit@1/Hit@5, MRR, nDCG@10 over the episode ranking (order of first appearance)
  - "episode" group (user names episodes): share of chunks from the named episodes,
    and whether every named episode appears at all
  - latency per query (after a warm-up, so model loading isn't counted)

Usage:
    python scripts/eval_retrieval.py [--k 10] [--systems dense,hybrid,hybrid_rerank]
        [--rerank-pool 30] [--reranker MODEL] [--verbose] [--out results.json]
Extra systems: legacy (pre-hybrid baseline), bm25, dense_rerank (ablation).
"""
import argparse
import json
import math
import pathlib
import sys
import time
from collections import defaultdict

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.episode_embeddings import find_similar_episodes
from utils.path_config import CHROMA_PERSIST_DIR, DATA_DIR
from utils.query_parser import extract_episode_numbers
from utils.reranker import DEFAULT_RERANKER
from utils.retriever import Retriever

EVAL_PATH = DATA_DIR / "eval_queries.jsonl"

# system name -> (mode, rerank)
SYSTEMS = {
    "legacy": ("dense", False),
    "dense": ("dense", False),
    "bm25": ("bm25", False),
    "hybrid": ("hybrid", False),
    "dense_rerank": ("dense", True),
    "hybrid_rerank": ("hybrid", True),
}
RANK_METRICS = ["P@5", "P@10", "Hit@1", "Hit@5", "MRR", "nDCG@10"]
EPISODE_METRICS = ["share", "all_named_present"]


def load_queries(path=EVAL_PATH):
    with open(path, "r", encoding="utf-8") as infile:
        return [json.loads(line) for line in infile if line.strip()]


def episode_ranking(items):
    ranking = []
    for item in items:
        ep = item.get("episode_number")
        if ep not in ranking:
            ranking.append(ep)
    return ranking


def ndcg_at(ranking, relevant, n=10):
    dcg = sum(1.0 / math.log2(i + 2) for i, ep in enumerate(ranking[:n]) if ep in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), n)))
    return dcg / ideal if ideal else 0.0


def precision_at(items, relevant, n):
    top = items[:n]
    return sum(1 for it in top if it.get("episode_number") in relevant) / n if top else 0.0


class LegacyRetriever(Retriever):
    """Pre-hybrid behaviour: dense top-k on the raw query, no episode slot reservation."""

    def retrieve(self, query, k=5, episode_numbers=None, **_ignored):
        from utils.chroma_store import query as chroma_query

        result = chroma_query(
            query_text=query, k=k, persist_directory=self.persist_directory,
            episode_numbers=episode_numbers,
        )
        return [
            {"text": d, "episode_number": (m or {}).get("episode_number"), "sources": "dense"}
            for d, m in zip(result.get("documents") or [], result.get("metadatas") or [])
        ]


def retrieval_args(query, k):
    """Mirror scripts/run_rag_pipeline.main: k and filters for named-episode queries."""
    epnums = extract_episode_numbers(query)
    if not epnums:
        return k, None, None
    kk = min(max(k, 8 * len(epnums)), 30)
    candidates = find_similar_episodes(epnums, top_k=20, include_original=True, exclude_self=True)
    return kk, (sorted(candidates) if candidates else None), epnums


def evaluate_query(retriever, mode, rerank, q, k):
    kk, episode_filter, epnums = retrieval_args(q["query"], k)
    start = time.perf_counter()
    items = retriever.retrieve(
        q["query"], k=kk, episode_numbers=episode_filter, required_episodes=epnums,
        mode=mode, rerank=rerank, group_by_episode=False,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    relevant = set(q["episodes"])

    row = {"group": q["group"], "query": q["query"], "latency_ms": latency_ms,
           "top_chunks": [it.get("chunk_id") for it in items[:10]]}
    if q["group"] == "episode":
        row["share"] = precision_at(items, relevant, len(items)) if items else 0.0
        row["all_named_present"] = float(
            all(any(it.get("episode_number") == ep for it in items) for ep in relevant)
        )
        row["primary"] = row["share"]
        row["P@10"] = precision_at(items, relevant, 10)
    else:
        ranking = episode_ranking(items)
        rank = next((i + 1 for i, ep in enumerate(ranking) if ep in relevant), None)
        row.update({
            "rank": rank,
            "top_episodes": ranking[:5],
            "P@5": precision_at(items, relevant, 5),
            "P@10": precision_at(items, relevant, 10),
            "Hit@1": float(rank == 1),
            "Hit@5": float(rank is not None and rank <= 5),
            "MRR": 1.0 / rank if rank else 0.0,
            "nDCG@10": ndcg_at(ranking, relevant, 10),
        })
        row["primary"] = row["MRR"]
    return row


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, math.ceil(p / 100 * len(values)) - 1))
    return values[idx]


def print_table(title, metrics, systems, rows_by_system, group_filter):
    print(f"\n{title}")
    print(f"  {'system':<14}" + "".join(f"{m:>11}" for m in metrics))
    for name in systems:
        rows = [r for r in rows_by_system[name] if group_filter(r["group"])]
        print(f"  {name:<14}" + "".join(f"{mean(r[m] for r in rows):>11.3f}" for m in metrics))


def compare(prev, curr, rows_by_system, verbose):
    a_rows, b_rows = rows_by_system[prev], rows_by_system[curr]
    print(f"\n{prev} -> {curr}   (per query; primary = MRR, or share for 'episode' queries)")
    for metric in ("primary", "P@10"):
        wins = ties = losses = 0
        for a, b in zip(a_rows, b_rows):
            delta = b[metric] - a[metric]
            if abs(delta) < 1e-9:
                ties += 1
            elif delta > 0:
                wins += 1
            else:
                losses += 1
        print(f"  {metric:<8} wins {wins:>3}   ties {ties:>3}   losses {losses:>3}")
    if verbose:
        for a, b in zip(a_rows, b_rows):
            if abs(b["primary"] - a["primary"]) > 1e-9 or abs(b["P@10"] - a["P@10"]) > 1e-9:
                sign = "+" if (b["primary"], b["P@10"]) > (a["primary"], a["P@10"]) else "-"
                extra = f"rank {a.get('rank')} -> {b.get('rank')}" if "rank" in a else ""
                print(f"   {sign} [{a['group']:<10}] {a['query'][:55]:<55} "
                      f"primary {a['primary']:.2f}->{b['primary']:.2f}  "
                      f"P@10 {a['P@10']:.2f}->{b['P@10']:.2f}  {extra}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--systems", default="dense,hybrid,hybrid_rerank",
                        help=f"comma-separated, in stage order; options: {','.join(SYSTEMS)}")
    parser.add_argument("--rerank-pool", type=int, default=30)
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", type=pathlib.Path, help="write per-query results as JSON")
    args = parser.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        parser.error(f"unknown systems: {unknown}")

    queries = load_queries()
    retriever = Retriever(
        persist_directory=CHROMA_PERSIST_DIR,
        reranker_model=args.reranker,
        rerank_pool=args.rerank_pool,
    )
    legacy = LegacyRetriever(persist_directory=CHROMA_PERSIST_DIR)

    rows_by_system = {}
    for name in systems:
        mode, rerank = SYSTEMS[name]
        ret = legacy if name == "legacy" else retriever
        # Warm-up: load models / indexes so latency reflects steady-state queries.
        ret.retrieve("warm up query about ethics", k=args.k, mode=mode, rerank=rerank)
        rows_by_system[name] = [evaluate_query(ret, mode, rerank, q, args.k) for q in queries]

    groups = []
    for q in queries:
        if q["group"] not in groups:
            groups.append(q["group"])
    rank_groups = [g for g in groups if g != "episode"]

    print(f"\nRetrieval eval: k={args.k}, {len(queries)} queries, "
          f"reranker={args.reranker} (pool {args.rerank_pool})")
    for group in rank_groups:
        n = sum(1 for q in queries if q["group"] == group)
        print_table(f"{group} queries (n={n})", RANK_METRICS, systems, rows_by_system, lambda g, group=group: g == group)
    if rank_groups:
        n = sum(1 for q in queries if q["group"] in rank_groups)
        print_table(f"OVERALL, excluding 'episode' (n={n})", RANK_METRICS, systems, rows_by_system,
                    lambda g: g in rank_groups)
    if "episode" in groups:
        n = sum(1 for q in queries if q["group"] == "episode")
        print_table(f"episode queries (n={n})", EPISODE_METRICS, systems, rows_by_system, lambda g: g == "episode")

    print("\nlatency per query (ms)")
    print(f"  {'system':<14}{'mean':>11}{'p95':>11}")
    for name in systems:
        lat = [r["latency_ms"] for r in rows_by_system[name]]
        print(f"  {name:<14}{mean(lat):>11.0f}{percentile(lat, 95):>11.0f}")

    for prev, curr in zip(systems, systems[1:]):
        compare(prev, curr, rows_by_system, args.verbose)

    if args.out:
        payload = {
            "k": args.k, "reranker": args.reranker, "rerank_pool": args.rerank_pool,
            "systems": systems, "results": rows_by_system,
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote per-query results to {args.out}")


if __name__ == "__main__":
    main()
