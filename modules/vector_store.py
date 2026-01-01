import faiss
import numpy as np
from typing import List, Tuple
from modules.config import EMBEDDING_DIMENSION, DEFAULT_TOP_K


class VectorStore:
    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []

    def add_vectors(self, embeddings: np.ndarray, texts: List[str]):
        self.texts = texts
        self.index = faiss.IndexFlatL2(self.dimension)

        embeddings = embeddings.astype("float32")
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, k: int = DEFAULT_TOP_K) -> Tuple[np.ndarray, np.ndarray]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        return distances, indices

    def get_texts(self, indices: np.ndarray) -> List[str]:
        return [self.texts[idx] for idx in indices]

    def reset(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
