from modules.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, SENTENCE_BOUNDARY_SEARCH_RANGE
from typing import List


class TextChunker:
    def __init__(self, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
        if overlap >= chunk_size:
            raise ValueError(f"Overlap must be smaller than chunk size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            ideal_end = min(start + self.chunk_size, text_len)
            end = self._find_sentence_boundary(text, ideal_end)

            if end <= start:
                end = ideal_end

            chunk = text[start:end]
            chunks.append(chunk)

            start = max(start + (self.chunk_size - self.overlap), end)

        return chunks

    def _find_sentence_boundary(self, text: str, position: int) -> int:
        start = max(0, position - SENTENCE_BOUNDARY_SEARCH_RANGE)
        end = min(len(text), position + SENTENCE_BOUNDARY_SEARCH_RANGE)

        segment = text[start:end]
        last_period = segment.rfind(". ")

        if last_period != -1:
            return start+last_period+2

        return position


if __name__ == "__main__":
    from modules.document_processor import DocumentProcessor

    processor = DocumentProcessor()
    text = processor.load_pdf("data/sample_pdfs/test.pdf")

    if text:
        chunker = TextChunker()
        chunks = chunker.chunk_text(text)

        print(f"✅ Original text: {len(text)} characters")
        print(f"✅ Created {len(chunks)} chunks")
        print(f"✅ First chunk: {len(chunks[0])} chars")
        print(f"✅ Last chunk: {len(chunks[-1])} chars")
        print(f"\nFirst chunk preview:\n{chunks[0][:200]}...")
