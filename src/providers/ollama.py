"""Ollama LLM provider."""

from langchain_ollama import ChatOllama
from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """Ollama LLM provider."""
    
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url
    
    def get_llm(self) -> ChatOllama:
        return ChatOllama(
            model=self.model, 
            temperature=0, 
            base_url=self.base_url
        )
    
    def is_available(self) -> bool:
        return True  # Always available if ollama package is installed