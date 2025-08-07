# scripts/build_index.py

from utils.embedding_utils import embed_chunks
from utils.io_utils import load_chunks, save_index, save_metadata
from utils.model_singleton import ModelSingleton
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

    # Step 3: Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Step 4: Save
    save_index(index, "faiss_transcript_index.index")
    save_metadata(data, "transcript_metadata.json")


if __name__ == "__main__":
    main()