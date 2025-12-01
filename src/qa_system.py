import os
from groq import Groq
from src.pdf_loader import load_pdf
from src.text_chunker import chunk_text
from src.embeddings import VectorStore
from dotenv import load_dotenv

load_dotenv()

class QASystem:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.vector_store = VectorStore()

    def load_document(self, pdf_path):
        text = load_pdf(pdf_path)
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        self.vector_store.add_texts(chunks)
        # print(f"document loaded : {len(chunks)} chunks stored")

    def ask(self, question, k=3, chat_history=None):
        if chat_history is None:
            chat_history = []

        relevant_chunks = self.vector_store.search(question, k=k)
        context = "\n\n".join(relevant_chunks)

        messages = []        
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

        for msg in chat_history:
            messages.append(
                {
                    "role": msg["role"], 
                    "context": msg["content"]
                }
            )

        messages.append(
            {
                "role": "user", 
                "content": question
            }
        )
            
        # prompt = prompt = f"""based on the following context from a document, answer the question.
        #                     context:
        #                         {context}

        #                     question : {question}

        #                     answer : """
        
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )

        answer = response.choices[0].message.content

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