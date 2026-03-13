import pathlib

from .chroma_store import query as chroma_query

class Retriever:
    def __init__(
        self,
        model_name='all-MiniLM-L6-v2',
        index_file='faiss_transcript_index.index',
        metadata_file='transcript_metadata.json',
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
            list: A list of tuples containing the chunk ID and similarity score.
        """

        result = chroma_query(
            query_text=query,
            k=k,
            collection_name=self.collection_name,
            model_name=self.model_name,
            persist_directory=self.persist_directory,
        )

        documents = result.get('documents') or []
        documents = [doc for doc in documents if isinstance(doc, str)]
        return '\n'.join(documents)