from modules.document_processor import DocumentProcessor
from modules.chunking import TextChunker
from modules.embedding_manager import EmbeddingManager
from modules.vector_store import VectorStore
from modules.retriever import Retriever
from modules.llm import LLMService
from typing import Dict, List, Optional


class QASystem:
    def __init__(self, api_key: str):
        self.doc_processor = DocumentProcessor()
        self.chunker = TextChunker()
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore()
        self.retriever = Retriever(self.embedding_manager, self.vector_store)
        self.llm = LLMService(api_key)

    def load_document(self, pdf_path: str):
        text = self.doc_processor.load_pdf(pdf_path)

        if not text:
            raise ValueError(f"Failed to load PDF")

        chunks = self.chunker.chunk_text(text)

        embeddings = self.embedding_manager.encode_batch(chunks)

        self.vector_store.add_vectors(embeddings, chunks)

    def ask(self, question: str, chat_history: Optional[List[Dict]] = None, k: int = 3) -> Dict:
        relevant_chunks = self.retriever.retrieve(question, k=k)

        answer = self.llm.generate_answer(
            question=question, context_chunks=relevant_chunks, chat_history=chat_history)

        return {"answer": answer, "sources": relevant_chunks}
