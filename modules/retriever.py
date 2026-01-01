from modules.embedding_manager import EmbeddingManager
from modules.vector_store import VectorStore
from modules.config import DEFAULT_TOP_K
from typing import List


class Retriever:
    def __init__(self, embedding_manager: EmbeddingManager, vector_store: VectorStore):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> List[str]:
        query_embedding = self.embedding_manager.encode_text(query)

        distances, indices = self.vector_store.search(query_embedding, k)

        results = self.vector_store.get_texts(indices[0])
        return results
