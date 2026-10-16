from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

from config import settings


# Main model (supervisor) configuration
def get_supervisor_model():
    fallback_model = ChatOllama(
        model="qwen2.5-coder:7b-instruct-q4_K_M",
        temperature=0.3,
        base_url=settings.OLLAMA_BASE_URL,
        keep_alive=settings.OLLAMA_KEEP_ALIVE,
        num_predict=500,
    )
    model = ChatMistralAI(
        model="ministral-8b-latest",
        api_key=settings.MISTRAL_API_KEY,
        temperature=0.2,
        max_tokens=500,
    )
    return model.with_fallbacks([fallback_model])
