EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L3-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 32

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
SENTENCE_BOUNDARY_SEARCH_RANGE = 100

FAISS_INDEX_TYPE = "IndexFlatL2"
DEFAULT_TOP_K = 3

LLM_MODEL_NAME = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1000

SYSTEM_PROMPT_TEMPLATE = """ You are a helpful assistant answering questions about a document.
Use the following context to answer questions : 
{context}
If the answer is not in the context, say so."""

TEMP_PDF_PATH = "temp.pdf"
