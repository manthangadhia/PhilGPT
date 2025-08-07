import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from .model_singleton import ModelSingleton
from .io_utils import load_index

class Retriever:
    def __init__(self, model_name='all-MiniLM-L6-v2', index_file='faiss_transcript_index.index'):
        # use modelSingleton to ensure only one instance of the model is created
        model_singleton = ModelSingleton(model_name)
        self.model = model_singleton.get_model()

        # load faiss index correctly from io_utils
        self.index = load_index(index_file)
        self.index.nprobe = 10  # Set the number of probes for better recall

    def retrieve(self, query, k=5):
        """
        Retrieve the top k most similar chunks for a given query.
        
        Args:
            query (str): The query string to search for.
            k (int): The number of top results to return.

        Returns:
            list: A list of tuples containing the chunk ID and similarity score.
        """
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        distances, indices = self.index.search(query_embedding.cpu().numpy(), k)
        
        results = []
        for i in range(k):
            if indices[0][i] != -1:  # Check if the index is valid
                results.append((indices[0][i], float(distances[0][i])))
        
        return results