"""
Light wrapper around ChromaDB for adding and querying chunks.

Provides two functions used by the project:
- add_chunks(texts, metadatas=None, ids=None, ...)
- query(query_text, k=5, ...)

This file intentionally keeps a minimal dependency surface and gives
clear error messages if `chromadb` is not installed.
"""
from typing import List, Optional, Dict, Any
import pathlib

from .model_singleton import ModelSingleton
from .embedding_config import DATA_DIR, get_version_config

MAX_CHROMA_BATCH_SIZE = 5000  # keep under Chroma's internal 5461 limit


def _get_chroma_client(persist_directory: Optional[pathlib.Path] = None):
    try:
        import chromadb
        from chromadb.config import Settings
    except Exception as e:  # pragma: no cover - dependency error
        raise ImportError(
            "chromadb is required for chroma_store but is not installed. "
            "Install with `pip install chromadb` and try again."
        ) from e

    # Use PersistentClient when a persist directory is provided (newer chromadb API)
    if persist_directory is not None:
        persist_directory = pathlib.Path(persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)
        try:
            return chromadb.PersistentClient(path=str(persist_directory))
        except AttributeError:
            # Fallback for older chromadb versions that don't provide PersistentClient
            settings = dict(chroma_db_impl="duckdb+parquet", persist_directory=str(persist_directory))
            return chromadb.Client(Settings(**settings))

    # In-memory / default client
    return chromadb.Client()


def add_chunks(
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None,
    embeddings: Optional[Any] = None,
    collection_name: str = "transcripts",
    version: str = None,
    persist_directory: Optional[pathlib.Path] = None,
):
    """Add a batch of text chunks to a Chroma collection.

    Args:
        texts: List of string chunks to add.
        metadatas: Optional list of metadata dicts aligned with `texts`.
        ids: Optional list of ids for the documents.
        collection_name: Name of the Chroma collection to use/create.
        version: Embedding version key (uses embedding_config.DEFAULT_VERSION if None).
        persist_directory: Directory to persist Chroma DB (defaults to data/chroma).
        embeddings: Optional list/array of embeddings aligned with `texts`.

    Returns:
        dict: raw response from chroma collection.add (if available) or summary dict.
    """
    # Resolve version config (get_version_config accepts None and returns default)
    cfg = get_version_config(version)
    model_name = cfg.get("model_name")

    # ensure persist directory
    if persist_directory is None:
        persist_directory = DATA_DIR / "chroma"
    else:
        persist_directory = pathlib.Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    client = _get_chroma_client(persist_directory=persist_directory)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    embeddings_list = None
    if embeddings is None:
        model_singleton = ModelSingleton.get_instance(model_name)
        model = model_singleton.get_model()
        encoded = model.encode(texts, batch_size=64, show_progress_bar=False)
        embeddings_list = [emb.tolist() for emb in encoded]
    else:
        embeddings_list = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

    if len(embeddings_list) != len(texts):
        raise ValueError("Embeddings count must match texts count")

    results = []
    total = len(texts)
    for start in range(0, total, MAX_CHROMA_BATCH_SIZE):
        end = min(start + MAX_CHROMA_BATCH_SIZE, total)
        batch_kwargs = {
            "documents": texts[start:end],
            "embeddings": embeddings_list[start:end],
        }
        if metadatas is not None:
            batch_kwargs["metadatas"] = metadatas[start:end]
        if ids is not None:
            batch_kwargs["ids"] = ids[start:end]

        batch_kwargs = {k: v for k, v in batch_kwargs.items() if v is not None}
        results.append(collection.add(**batch_kwargs))

    return results[-1] if results else None


def query(
    query_text: str,
    k: int = 5,
    collection_name: str = "transcripts",
    version: Optional[str] = None,
    persist_directory: Optional[pathlib.Path] = None,
):
    """Query the Chroma collection and return nearest chunks.

    Args:
        query_text: The input query string.
        k: Number of nearest neighbors to return.
        collection_name: Name of the collection to query.
        version: Embedding version key used to select the encoder model.
        persist_directory: Directory where Chroma DB is persisted.

    Returns:
        dict: Dictionary with keys `ids`, `documents`, `metadatas`, and `distances`.
    """
    cfg = get_version_config(version)
    model_name = cfg.get("model_name")

    if persist_directory is None:
        persist_directory = DATA_DIR / "chroma"
    else:
        persist_directory = pathlib.Path(persist_directory)

    client = _get_chroma_client(persist_directory=persist_directory)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    model_singleton = ModelSingleton.get_instance(model_name)
    model = model_singleton.get_model()

    query_embedding = model.encode([query_text], show_progress_bar=False)[0]

    # Chroma expects list of embeddings when querying
    resp = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k,
        include=["ids", "documents", "metadatas", "distances"],
    )

    # The returned structure contains results per-query; we only sent one query
    result = {
        "ids": resp.get("ids", [[]])[0],
        "documents": resp.get("documents", [[]])[0],
        "metadatas": resp.get("metadatas", [[]])[0],
        "distances": resp.get("distances", [[]])[0],
    }
    return result


def get_existing_episode_numbers(
    collection_name: str = "transcripts",
    persist_directory: Optional[pathlib.Path] = None,
):
    """Return a set of existing `url` values from the collection metadatas.

    Useful to detect which transcript URLs have already been scraped/added.
    """
    if persist_directory is None:
        persist_directory = DATA_DIR / "chroma"
    else:
        persist_directory = pathlib.Path(persist_directory)

    client = _get_chroma_client(persist_directory=persist_directory)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    results = collection.get(include=["metadatas"]) or {}
    metadatas = results.get("metadatas", [])

    urls = set()
    for m in metadatas:
        if isinstance(m, dict):
            url = m.get("url")
            if url:
                urls.add(url)

    return urls
