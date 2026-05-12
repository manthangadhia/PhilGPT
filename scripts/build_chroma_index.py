"""Build a Chroma index from existing transcript chunks."""

import json
import sys
import pathlib
from typing import Any, Dict, List

import torch
from tqdm import tqdm

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.path_config import DATA_DIR
from utils.model_singleton import get_cached_model
from utils import chroma_store

CHROMA_PERSIST_DIR = DATA_DIR / "chroma"


def load_chunk_data() -> List[Dict[str, Any]]:
    chunks_path = DATA_DIR / "transcript_chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk file not found at {chunks_path}")

    chunks: List[Dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def main() -> None:
    chunks = load_chunk_data()
    if not chunks:
        print("No transcript chunks found; nothing to index.")
        return

    try:
        existing_urls = chroma_store.get_existing_episode_urls(persist_directory=CHROMA_PERSIST_DIR)
    except Exception as e:
        print(f"Warning: failed to read existing Chroma URLs ({e}). Proceeding with all chunks.")
        existing_urls = set()

    total_chunks = len(chunks)
    new_chunks = [chunk for chunk in chunks if chunk.get("url") not in existing_urls]
    skipped_chunks = total_chunks - len(new_chunks)

    print(f"Total chunks found: {total_chunks}")
    print(f"Skipped (already indexed): {skipped_chunks}")
    print(f"New chunks to index: {len(new_chunks)}")

    if not new_chunks:
        print("No new chunks to index.")
        return

    texts: List[str] = []
    ids: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for chunk in tqdm(new_chunks, desc="Preparing chunks", unit="chunk"):
        texts.append(chunk["text"])
        ids.append(chunk["chunk_id"])
        metadatas.append(
            {
                "episode_number": chunk.get("episode_number"),
                "title": chunk.get("title"),
                "url": chunk.get("url"),
            }
        )

    model = get_cached_model("all-MiniLM-L6-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        device=device,
    )

    chroma_store.add_chunks(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings.tolist(),
        persist_directory=CHROMA_PERSIST_DIR,
    )

    updated_urls = chroma_store.get_existing_episode_urls(persist_directory=CHROMA_PERSIST_DIR)
    print(f"Chroma collection currently tracks {len(updated_urls)} unique transcript URLs.")
    print(f"Documents inserted this run: {len(texts)}")


if __name__ == "__main__":
    main()