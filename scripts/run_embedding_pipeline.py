"""scripts/run_embedding_pipeline.py

Small helper to (re)create embeddings and faiss index for a chosen model version.
"""
import sys
import pathlib

# allow running from repo root or env directory
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.embedding_utils import embed_chunks
from utils.io_utils import save_index, save_metadata
from utils.model_singleton import ModelSingleton
from utils.embedding_config import get_version_directory
import faiss
import numpy as np
from tqdm import tqdm
from utils.io_utils import load_chunks

def run_embedding_pipeline(version='v1_miniLM', model_name='all-MiniLM-L6-v2'):
    """Run embedding pipeline and save embeddings + faiss index into version folder.

    Args:
        version (str): Version folder under data/ to store embeddings and index.
        model_name (str): SentenceTransformer model name to use.
    """
    # Load text chunks
    data = load_chunks('transcripts/transcript_chunks.jsonl')
    texts = [item['text'] for item in tqdm(data, desc='Preparing texts')]

    # Load model
    model_singleton = ModelSingleton(model_name)
    model = model_singleton.get_model()

    # Embed
    embeddings = embed_chunks(model, texts)

    # Save embeddings into version directory
    version_dir = get_version_directory(version)
    version_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(version_dir / 'embeddings.npy'), embeddings)
    print(f"Saved embeddings to {version_dir / 'embeddings.npy'}")

    # Build FAISS index (inner product) and save
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    save_index(index, 'faiss_transcript_index.index', version=version)
    print(f"Saved FAISS index to version {version}")

    # Save metadata in root data folder for now
    save_metadata(data, 'transcript_metadata.json')

    return {'version': version, 'ntotal': index.ntotal}

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--version', default='v1_miniLM')
    p.add_argument('--model', default='all-MiniLM-L6-v2')
    args = p.parse_args()
    print(run_embedding_pipeline(version=args.version, model_name=args.model))
