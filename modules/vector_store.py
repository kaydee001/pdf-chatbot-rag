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


if __name__ == "__main__":
    from modules.embedding_manager import EmbeddingManager

    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()

    texts = [
        "Python is great for machine learning",
        "Neural networks can learn complex patterns",
        "I love pizza and pasta",
        "Deep learning requires lots of data",
        "The weather is nice today"
    ]

    print("creating embeddings : ")
    embeddings = embedding_manager.encode_batch(texts)

    print("adding to vector store : ")
    vector_store.add_vectors(embeddings, texts)

    query = "What is good for AI?"
    print(f"query : '{query}'")
    query_embedding = embedding_manager.encode_text(query)

    distances, indices = vector_store.search(query_embedding, k=3)
    results = vector_store.get_texts(indices[0])

    print(f"top 3 results : ")
    for i, (text, distance) in enumerate(zip(results, distances[0]), 1):
        print(f"{i}. {text} (distance : {distance:.4f})")
