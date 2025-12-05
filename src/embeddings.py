from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorStore:
    def __init__(self, model_name = "paraphrase-MiniLM-L3-v2"):
        # loading pre trained embedding model
        self.model = SentenceTransformer(model_name)
        # embedding dimension for this model = 384
        self.dimension = 384
        # FAISS index for similarity search
        self.index = faiss.IndexFlatL2(self.dimension) # L2 = euclidian distance
        # storing text to retrive them by index
        self.texts = []

    def add_texts(self, texts):
        # storing original texts
        self.texts = texts
        # reset index (for new docs)
        self.index = faiss.IndexFlatL2(self.dimension)

        # generate embeddings
        embeddings = self.model.encode(texts, batch_size=32)
        # FAISS requirement -> convert to numpy array with float32
        embeddings = np.array(embeddings).astype("float32")

        # add embedding to FAISS index
        self.index.add(embeddings)
        print(f"stored {self.index.ntotal} chunks in vector database")

    def search(self, query, k=3):
        # converting query to embedding -> must a list for batch processing
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        # search FAISS index -> returns how far(distance) and which chunks(indices)
        distances, indices = self.index.search(query_embedding, k)

        # retriving original texts using indices
        results = []
        for idx in indices[0]: # indices[0] cause we sent only 1 query
            results.append(self.texts[idx])

        return results
    
if __name__ == "__main__":
    from src.pdf_loader import load_pdf
    from src.text_chunker import chunk_text

    pdf_path = "data/sample_pdfs/test.pdf"
    text = load_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size=1000, overlap=200)

    print(f"processing {len(chunks)} chunks")

    vector_store = VectorStore()
    vector_store.add_texts(chunks)

    print(f"chunks stored, testing search")

    query = "What is attention mechanism?"
    results = vector_store.search(query, k=3)

    print(f"query : {query}")
    print(f"top {len(results)} results : \n")
    for i, result in enumerate(results, 1):
        print(f"result {i}")
        print(result[:200] + "\n")