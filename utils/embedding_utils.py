from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import pathlib

from .embedding_config import get_version_config, get_version_file_path, get_version_directory

root_dir = pathlib.Path(__file__).parent.parent
data_dir = root_dir / 'data'

def embed_chunks(model, text, version="v1_miniLM"):
    """
    Embed the text chunks using the specified model and version.
    
    Args:
        model: The SentenceTransformer model to use
        text: List of text chunks to embed
        version: Version identifier for caching (default: v1_miniLM)
    """
    # Get versioned embeddings path
    embeddings_path = get_version_file_path(version, "embeddings")
    
    # Ensure version directory exists
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    if embeddings_path.exists():
        print(f"Loading cached embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
    else:
        print(f"Generating embeddings for version {version}...")
        embeddings = model.encode(text, batch_size=64, show_progress_bar=True, 
                                device="cuda", normalize_embeddings=True)
        np.save(embeddings_path, embeddings)
        print(f"Saved embeddings to {embeddings_path}")
    return embeddings