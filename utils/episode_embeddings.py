import json
import pathlib
import sys
from typing import Dict, List, Optional
import numpy as np
    
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.chroma_store import get_collection
from utils.path_config import DATA_DIR, CHROMA_PERSIST_DIR

EPISODE_EMBEDDINGS_PATH = DATA_DIR / "episode_mean_embeddings.npy"
EPISODE_INDEX_PATH = DATA_DIR / "episode_mean_embeddings_index.json"


def _coerce_episode_number(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def load_episode_index(path: pathlib.Path = EPISODE_INDEX_PATH) -> List[int]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)
    if not isinstance(data, list):
        raise ValueError("Episode index file must contain a JSON list")
    episodes: List[int] = []
    for item in data:
        ep = _coerce_episode_number(item)
        if ep is not None:
            episodes.append(ep)
    return episodes


def save_episode_index(episodes: List[int], path: pathlib.Path = EPISODE_INDEX_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        json.dump(episodes, outfile)


def load_embedding_matrix(
    path: pathlib.Path = EPISODE_EMBEDDINGS_PATH,
    expected_rows: Optional[int] = None,
) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    matrix = np.load(path)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError(
            f"Embedding matrix rows ({matrix.shape[0]}) do not match episode index ({expected_rows})"
        )
    return matrix


def save_embedding_matrix(
    matrix: np.ndarray,
    path: pathlib.Path = EPISODE_EMBEDDINGS_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix.astype(np.float32, copy=False))


def update_episode_mean_embeddings(
    collection_name: str = "transcripts",
    persist_directory: pathlib.Path = CHROMA_PERSIST_DIR,
    batch_size: int = 1000,
) -> None:
    episodes = load_episode_index()
    existing_set = set(episodes)
    existing_matrix = load_embedding_matrix(expected_rows=len(episodes))

    collection = get_collection(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    total_count = collection.count()
    if total_count == 0:
        print("Chroma collection is empty; no embeddings to process.")
        return

    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}

    offset = 0
    while offset < total_count:
        batch = collection.get(
            include=["embeddings", "metadatas"],
            limit=batch_size,
            offset=offset,
        )
        embeddings = batch.get("embeddings")
        if embeddings is None:
            embeddings = []
        elif isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        metadatas = batch.get("metadatas")
        if metadatas is None:
            metadatas = []

        for embedding, meta in zip(embeddings, metadatas):
            if not isinstance(meta, dict):
                continue
            episode_number = _coerce_episode_number(meta.get("episode_number"))
            if episode_number is None or episode_number in existing_set:
                continue
            if embedding is None:
                continue

            vector = np.asarray(embedding, dtype=np.float32)
            if episode_number in sums:
                sums[episode_number] += vector
                counts[episode_number] += 1
            else:
                sums[episode_number] = vector.copy()
                counts[episode_number] = 1

        offset += batch_size

    if not sums:
        print("No new episodes found; embeddings matrix is already up to date.")
        return

    new_episode_numbers = sorted(sums.keys())
    new_embeddings = np.vstack(
        [sums[ep] / float(counts[ep]) for ep in new_episode_numbers]
    ).astype(np.float32, copy=False)

    if existing_matrix is None:
        combined_matrix = new_embeddings
        combined_episodes = new_episode_numbers
    else:
        combined_matrix = np.vstack([existing_matrix, new_embeddings]).astype(
            np.float32, copy=False
        )
        combined_episodes = episodes + new_episode_numbers

    save_embedding_matrix(combined_matrix)
    save_episode_index(combined_episodes)

    print(
        "Updated episode embeddings: "
        f"{len(new_episode_numbers)} new, "
        f"{len(combined_episodes)} total."
    )


def main() -> None:
    update_episode_mean_embeddings()


if __name__ == "__main__":
    main()
