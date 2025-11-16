# scripts/build_index.py

import sys
import pathlib

# Add the project root to Python path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.embedding_utils import embed_chunks
from utils.io_utils import load_chunks, save_index, save_metadata
from utils.model_singleton import ModelSingleton
from utils.embedding_config import get_version_directory
import faiss
from tqdm import tqdm

def main():
    # Step 1: Load data
    data = load_chunks("transcript_chunks.jsonl")
    texts = [item["text"] for item in tqdm(data)]

    # Step 2: Load model and embed
    model_singleton = ModelSingleton("all-MiniLM-L6-v2")
    model = model_singleton.get_model()
    embeddings = embed_chunks(model, texts)

    # Save embeddings into the versioned directory
    version = "v1_miniLM"
    version_dir = get_version_directory(version)
    version_dir.mkdir(parents=True, exist_ok=True)
    import numpy as _np
    _np.save(str(version_dir / "embeddings.npy"), embeddings)

    # Step 3: Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Step 4: Save
    save_index(index, "faiss_transcript_index.index", version=version)
    # Save metadata at project root (keeps a single canonical metadata file)
    save_metadata(data, "transcript_metadata.json")


if __name__ == "__main__":
    main()