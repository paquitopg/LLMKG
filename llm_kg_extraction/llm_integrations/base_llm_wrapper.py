import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

class BaseLLMWrapper(ABC):
    """
    Abstract base class for Large Language Model client wrappers.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initializes the LLM wrapper.

        Args:
            model_name (str): The name of the model to be used.
            **kwargs: Additional provider-specific keyword arguments.
        """
        self.model_name = model_name
        self.additional_config = kwargs

    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.1, **kwargs) -> Optional[str]:
        """
        Generates a response based on a list of chat messages.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries,
                e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}].
            temperature (float): The sampling temperature.
            **kwargs: Additional provider-specific keyword arguments for the completion.

        Returns:
            Optional[str]: The content of the LLM's response message, or None if an error occurs.
        """
        pass

    @abstractmethod
    def generate_content(self, prompt: Union[str, List[Any]], temperature: float = 0.1, response_mime_type: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Generates content based on a prompt (text or multimodal parts).

        Args:
            prompt (Union[str, List[Any]]): The prompt string or a list of content parts
                                            (e.g., for multimodal input with Vertex AI).
            temperature (float): The sampling temperature.
            response_mime_type (Optional[str]): The desired MIME type for the response (e.g., "application/json").
            **kwargs: Additional provider-specific keyword arguments for content generation.

        Returns:
            Optional[str]: The generated text content, or None if an error occurs.
        """
        pass