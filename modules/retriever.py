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


if __name__ == "__main__":
    from modules.document_processor import DocumentProcessor
    from modules.chunking import TextChunker

    doc_processor = DocumentProcessor()
    chunker = TextChunker()
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    retriever = Retriever(embedding_manager, vector_store)

    print("📄 Loading PDF...")
    text = doc_processor.load_pdf("data/sample_pdfs/test.pdf")

    print("✂️ Chunking text...")
    chunks = chunker.chunk_text(text)

    print("🧠 Creating embeddings...")
    embeddings = embedding_manager.encode_batch(chunks)

    print("💾 Adding to vector store...")
    vector_store.add_vectors(embeddings, chunks)

    query = "What is attention mechanism?"
    print(f"\n🔍 Query: '{query}'")

    results = retriever.retrieve(query, k=3)

    print(f"\n✅ Top 3 relevant chunks:\n")
    for i, chunk in enumerate(results, 1):
        print(f"--- Chunk {i} ---")
        print(chunk[:200] + "...\n")
