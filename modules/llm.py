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
