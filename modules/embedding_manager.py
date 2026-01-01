from sentence_transformers import SentenceTransformer
from modules.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION, EMBEDDING_BATCH_SIZE
import numpy as np
from typing import List, Union


class EmbeddingManager:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.dimension = EMBEDDING_DIMENSION

    def encode_text(self, text: str) -> np.ndarray:
        embedding = self.model.encode([text])[0]
        return np.array(embedding).astype("float32")

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, batch_size=EMBEDDING_BATCH_SIZE)
        return np.array(embeddings).astype("float32")
