from groq import Groq
from modules.config import LLM_MODEL_NAME, SYSTEM_PROMPT_TEMPLATE
from typing import List, Optional,  Dict


class LLMService:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = LLM_MODEL_NAME

    def generate_answer(self, question: str, context_chunks: List[str], chat_history: Optional[List[Dict]] = None) -> str:
        context = "\n\n".join(context_chunks)
        system_message = SYSTEM_PROMPT_TEMPLATE.format(context=context)

        messages = [{"role": "system", "content": system_message}]

        if chat_history:
            messages.extend(chat_history)

        messages.append({"role": "user", "content": question})

        response = self.client.chat.completions.create(
            model=self.model, messages=messages)

        answer = response.choices[0].message.content
        return answer


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("❌ GROQ_API_KEY not found in .env")
        exit(1)

    llm_service = LLMService(api_key)

    context_chunks = [
        "The attention mechanism allows neural networks to focus on relevant parts of the input.",
        "Self-attention computes relationships between all positions in a sequence.",
        "Multi-head attention uses multiple attention layers in parallel."
    ]

    question = "What is attention mechanism?"

    print(f"🔍 Question: {question}\n")
    print("📝 Generating answer...\n")

    answer = llm_service.generate_answer(question, context_chunks)

    print(f"✅ Answer:\n{answer}")
