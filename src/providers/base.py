"""Base LLM provider abstraction."""

from abc import ABC, abstractmethod
from langchain_core.language_models.base import BaseLanguageModel


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def get_llm(self) -> BaseLanguageModel:
        """Get configured LLM instance."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass