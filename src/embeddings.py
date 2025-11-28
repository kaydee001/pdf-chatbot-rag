from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorStore:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []

    def add_texts(self, texts):
        self.texts = texts

        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)
        print(f"stored {self.index.ntotal} chunks in vector database")

    def search(self, query, k=3):
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx in indices[0]:
            results.append(self.texts[idx])

        return results
    
if __name__ == "__main__":
    from pdf_loader import load_pdf
    from text_chunker import chunk_text

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

# texts = ["the cat sat on the mat", "a feline rested on the rug", "python is a programming language"]

# emb = model.encode(texts)
# emb = np.array(emb).astype("float32")

# print(f"emb shape: {emb.shape}")

# dimension = 384
# index = faiss.IndexFlatL2(dimension)

# index.add(emb)
# print(f"stored {index.ntotal} vectors in database")

# query = "kitten on carpet"
# query_embedding = model.encode([query])
# query_embedding = np.array(query_embedding).astype("float32")

# print(f"query embedding shape : {query_embedding.shape}")

# k = 2
# distances, indices = index.search(query_embedding, k)

# print(f"distances : {distances}")
# print(f"indices : {indices}")

# print(f"query : {query}")
# print("most similar texts : ")
# for i, idx in enumerate(indices[0]):
#     print(f"{i+1}. {texts[idx]} (distance : {distances[0][i]:.2f})")