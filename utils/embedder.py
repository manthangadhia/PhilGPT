# utils/embedder.py

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        self.model = SentenceTransformer(model_name)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode(self, texts: list[str]) -> np.ndarray:
        # Tokenize
        # Get model output
        # Mean pool / CLS token extraction
        # Return as NumPy array
        embeddings = self.model.encode(texts, show_progress_bar=True, device=self.device)
        return np.array(embeddings)