from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

import pathlib

root_dir = pathlib.Path(__file__).parent.parent
data_dir = root_dir / 'data'

def embed_chunks(model, text):
    """
    Embed the text chunks using the specified model.
    """
    # cached embeddings path
    embeddings_path = data_dir / 'embeddings.npy'

    if embeddings_path.exists():
        print(f"Loading cached embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
    else:
        print("Generating embeddings...")
        embeddings = model.encode(text, batch_size=64, show_progress_bar=True, 
                                device="cuda", normalize_embeddings=True)
        np.save(embeddings_path, embeddings)
    return embeddings