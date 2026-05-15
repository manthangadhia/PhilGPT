import pathlib

from .chroma_store import query as chroma_query

class Retriever:
    def __init__(
        self,
        model_name='all-MiniLM-L6-v2',
        collection_name='transcripts',
        persist_directory=None,
    ):
        # Keep legacy args (`model_name`, `index_file`, `metadata_file`) for call-site compatibility.
        self.model_name = model_name
        self.collection_name = collection_name
        self.persist_directory = pathlib.Path(persist_directory) if persist_directory else None

    def retrieve(self, query, k=5):
        """
        Retrieve the top k most similar chunks for a given query.
        
        Args:
            query (str): The query string to search for.
            k (int): The number of top results to return.

        Returns:
            list: A list of dicts with chunk text and metadata fields.
        """

        result = chroma_query(
            query_text=query,
            k=k,
            collection_name=self.collection_name,
            model_name=self.model_name,
            persist_directory=self.persist_directory,
        )

        documents = result.get('documents') or []
        metadatas = result.get('metadatas') or []

        if not metadatas:
            metadatas = [{} for _ in documents]

        items = []
        for doc, meta in zip(documents, metadatas):
            if not isinstance(doc, str):
                continue
            meta = meta if isinstance(meta, dict) else {}
            items.append(
                {
                    "text": doc,
                    "episode_number": meta.get("episode_number"),
                    "title": meta.get("title"),
                }
            )

        return items