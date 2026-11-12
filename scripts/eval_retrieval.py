"""
Evaluate retrieval quality on a small labelled query set (data/eval_queries.jsonl).

Each line: {"group": "name"|"concept"|"episode", "query": str, "episodes": [int, ...]}

For "name"/"concept" queries we rank episodes by the order in which their chunks are
retrieved and report Hit@5, Hit@10 and MRR (any listed episode counts as relevant).
For "episode" queries (the user names an episode) we follow the same flow as the
RAG pipeline and report the share of retrieved chunks that come from the named episodes.

Usage:
    python scripts/eval_retrieval.py [--k 10] [--verbose]
"""
import argparse
import json
import pathlib
import sys
from collections import defaultdict

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.episode_embeddings import find_similar_episodes
from utils.path_config import CHROMA_PERSIST_DIR, DATA_DIR
from utils.query_parser import extract_episode_numbers
from utils.retriever import Retriever

EVAL_PATH = DATA_DIR / "eval_queries.jsonl"


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


def run_system(name, retriever, mode, queries, k, legacy=False, verbose=False):
    stats = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    for q in queries:
        group, query, relevant = q["group"], q["query"], set(q["episodes"])
        epnums = extract_episode_numbers(query)
        episode_filter = None
        kk = k
        if epnums:
            kk = min(max(k, 8 * len(epnums)), 30)
            candidates = find_similar_episodes(epnums, top_k=20, include_original=True, exclude_self=True)
            episode_filter = sorted(candidates) if candidates else None

        if legacy:
            # Previous behaviour: dense only, raw query (episode refs included), no reserved slots.
            items = retriever.retrieve(query, k=kk, episode_numbers=episode_filter, mode="dense", _legacy=True)
        else:
            items = retriever.retrieve(
                query, k=kk, episode_numbers=episode_filter,
                required_episodes=epnums or None, mode=mode,
            )

        counts[group] += 1
        if group == "episode":
            share = sum(1 for it in items if it.get("episode_number") in relevant) / max(len(items), 1)
            covered = all(any(it.get("episode_number") == ep for it in items) for ep in relevant)
            stats[group]["share"] += share
            stats[group]["all_named_present"] += float(covered)
            detail = f"share={share:.2f} all_present={covered}"
        else:
            ranking = episode_ranking(items)
            rank = next((i + 1 for i, ep in enumerate(ranking) if ep in relevant), None)
            stats[group]["hit@5"] += float(rank is not None and rank <= 5)
            stats[group]["hit@10"] += float(rank is not None and rank <= 10)
            stats[group]["mrr"] += 1.0 / rank if rank else 0.0
            detail = f"rank={rank} top={ranking[:5]}"
        if verbose:
            print(f"  [{name:<7}] {group:<7} {query[:55]:<55} {detail}")

    return {g: {m: v / counts[g] for m, v in s.items()} for g, s in stats.items()}, counts


class LegacyRetriever(Retriever):
    """Reproduces the pre-hybrid retrieval (dense top-k, query embedded as typed)."""

    def retrieve(self, query, k=5, episode_numbers=None, required_episodes=None, mode="dense", _legacy=False):
        from utils.chroma_store import query as chroma_query

        result = chroma_query(
            query_text=query, k=k, persist_directory=self.persist_directory,
            episode_numbers=episode_numbers,
        )
        return [
            {"text": d, "episode_number": (m or {}).get("episode_number")}
            for d, m in zip(result.get("documents") or [], result.get("metadatas") or [])
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    queries = load_queries()
    retriever = Retriever(persist_directory=CHROMA_PERSIST_DIR)
    systems = [
        ("legacy", LegacyRetriever(persist_directory=CHROMA_PERSIST_DIR), "dense", True),
        ("dense", retriever, "dense", False),
        ("bm25", retriever, "bm25", False),
        ("hybrid", retriever, "hybrid", False),
    ]

    results = {}
    for name, ret, mode, legacy in systems:
        if args.verbose:
            print(f"== {name}")
        results[name], counts = run_system(name, ret, mode, queries, args.k, legacy, args.verbose)

    groups = [g for g in ("name", "concept", "episode") if g in counts]
    print(f"\nRetrieval eval (k={args.k}, {len(queries)} queries)")
    for group in groups:
        metrics = list(next(iter(results.values()))[group].keys())
        print(f"\n{group} queries (n={counts[group]})")
        print(f"  {'system':<8}" + "".join(f"{m:>20}" for m in metrics))
        for name in results:
            print(f"  {name:<8}" + "".join(f"{results[name][group][m]:>20.3f}" for m in metrics))


if __name__ == "__main__":
    main()
