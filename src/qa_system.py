import os
from groq import Groq
from src.pdf_loader import load_pdf
from src.text_chunker import chunk_text
from src.embeddings import VectorStore
from dotenv import load_dotenv

load_dotenv()

class QASystem:
    def __init__(self, api_key):
        # initializing LLM client
        self.client = Groq(api_key=api_key)
        # initializing vector store for document chunks
        self.vector_store = VectorStore()

    def load_document(self, pdf_path):
        # extracting all texts from pdf
        text = load_pdf(pdf_path)
        # splitting into manageable chunks
        chunks = chunk_text(text, chunk_size=1500, overlap=200)
        self.vector_store.add_texts(chunks)

    def ask(self, question, k=3, chat_history=None):
        # initialize chat history; if not provided
        if chat_history is None:
            chat_history = []
        
        # step 1 : retrieve relevant chunks using semantic search 
        relevant_chunks = self.vector_store.search(question, k=k)
        # step 2 : combine chunks using context string
        context = "\n\n".join(relevant_chunks)
        # step 3 : build msg array for LLM
        messages = []        

        # system message instructs LLM how to behave
        system_message = f"""you are a helpful assistant answering questions about a doc;
        use the following context to answer questions :
        {context}
        if the answer is not the context, say so"""

        messages.append(
            {
                "role": "system", 
                "content": system_message
            }
        )

        # adding conversation history -> for context aware followups
        for msg in chat_history:
            messages.append(
                {
                    "role": msg["role"], # "user" or "assistant"
                    "content": msg["content"]
                }
            )
        # add current question
        messages.append(
            {
                "role": "user", 
                "content": question
            }
        )            
     
        # step 4 : call LLM api
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        # extract answer from response
        answer = response.choices[0].message.content

        # step 5 : return answer + sources
        return {
            "answer": answer, 
            "sources": relevant_chunks
        }

if __name__ == "__main__":
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY was not found in env variables")

    qa = QASystem(api_key)

    print("loading doc")
    qa.load_document("data/sample_pdfs/test.pdf")

    questions = [
        "What is attention mechanism?",
        "What dataset did they use?",
        "What are the main contributions of this paper?"
    ]

    for question in questions:
        print(f"question : {question}")
        answer = qa.ask(question)
        print(f"answer : {answer}\n")
        print("-"*20)