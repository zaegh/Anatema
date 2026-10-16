from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

from config import settings


def create_decision_maker():
    model = ChatOllama(
        model="qwen2.5:3b",
        temperature=0,
        base_url=settings.OLLAMA_BASE_URL,
        keep_alive=settings.OLLAMA_KEEP_ALIVE,
        num_predict=150,
    )
    fallback_model = ChatMistralAI(
        model="ministral-3b-latest",
        api_key=settings.MISTRAL_API_KEY,
        temperature=0,
        max_tokens=150,
    )
    return model.with_fallbacks([fallback_model])
