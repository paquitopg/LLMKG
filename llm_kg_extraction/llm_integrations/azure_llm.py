import os
from typing import List, Dict, Optional, Any, Union
from openai import AzureOpenAI, APIError, APITimeoutError, APIConnectionError
from .base_llm_wrapper import BaseLLMWrapper

class AzureLLM(BaseLLMWrapper):
    """
    Wrapper for Azure OpenAI Large Language Models.
    """

    def __init__(self, model_name: str, deployment_name: str,
                 api_key: Optional[str] = None,
                 api_version: Optional[str] = None,
                 azure_endpoint: Optional[str] = None,
                 **kwargs):
        """
        Initializes the Azure OpenAI LLM wrapper.

        Args:
            model_name (str): The base model name (e.g., "gpt-4"). Not directly used by
                              chat.completions.create's 'model' param if deployment_name is specified,
                              but stored for consistency.
            deployment_name (str): The Azure deployment name for the model. This is used as the 'model'
                                   parameter in chat.completions.create.
            api_key (Optional[str]): Azure OpenAI API key. Defaults to AZURE_OPENAI_API_KEY env var.
            api_version (Optional[str]): Azure OpenAI API version. Defaults to AZURE_OPENAI_API_VERSION env var.
            azure_endpoint (Optional[str]): Azure OpenAI endpoint. Defaults to AZURE_OPENAI_ENDPOINT env var.
            **kwargs: Additional keyword arguments passed to BaseLLMWrapper.
        """
        super().__init__(model_name=model_name, **kwargs)
        self.deployment_name = deployment_name

        try:
            self.sdk_client = AzureOpenAI(
                api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            )
        except Exception as e:
            print(f"Error initializing AzureOpenAI client: {e}")
            self.sdk_client = None # type: ignore

    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.1, **kwargs) -> Optional[str]:
        """
        Generates a chat completion using Azure OpenAI.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries.
            temperature (float): Sampling temperature.
            **kwargs: Additional arguments for chat.completions.create (e.g., max_tokens if needed).

        Returns:
            Optional[str]: The content of the LLM's response, or None on error.
        """
        if not self.sdk_client:
            print("AzureOpenAI SDK client not initialized.")
            return None
        try:
            response = self.sdk_client.chat.completions.create(
                model=self.deployment_name,  # Azure uses deployment name as the model identifier
                messages=messages,
                temperature=temperature,
                **kwargs
            )
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content
            else:
                print("Azure OpenAI: No response choices or message content found.")
                # print(f"Full response: {response}") # For debugging
                return None
        except (APIError, APITimeoutError, APIConnectionError) as e:
            print(f"Azure OpenAI API error: {type(e).__name__} - {e}")
        except Exception as e:
            print(f"An unexpected error occurred with Azure OpenAI: {type(e).__name__} - {e}")
        return None

    def generate_content(self, prompt: Union[str, List[Any]], temperature: float = 0.1, response_mime_type: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Generates content using Azure OpenAI. For Azure, this typically maps to a chat completion.
        If a simple string prompt is given, it's formatted as a user message.

        Args:
            prompt (Union[str, List[Any]]): A prompt string or a list of chat messages.
                                            If string, it becomes the user message. If list, it's used directly.
            temperature (float): Sampling temperature.
            response_mime_type (Optional[str]): Ignored for Azure, as JSON mode is typically handled by messages/tool calls.
            **kwargs: Additional arguments for chat.completions.create.

        Returns:
            Optional[str]: The generated text content, or None on error.
        """
        if isinstance(prompt, str):
            messages = [
                {"role": "system", "content": "You are a helpful assistant designed to output precisely as instructed."}, # Generic system prompt
                {"role": "user", "content": prompt}
            ]
        elif isinstance(prompt, list) and all(isinstance(item, dict) and "role" in item and "content" in item for item in prompt):
            messages = prompt # type: ignore
        else:
            print("AzureLLM.generate_content: Invalid prompt format. Must be a string or a list of chat message dicts.")
            return None

        # If 'response_mime_type' implies JSON output, ensure the system prompt reflects this for Azure.
        # Note: True JSON mode in OpenAI requires specific model versions and potentially different request parameters.
        # This is a simplified approach.
        if response_mime_type == "application/json":
            json_system_prompt = "You are a helpful assistant designed to output JSON."
            if messages[0]["role"] == "system":
                messages[0]["content"] = json_system_prompt # Override existing system prompt
            else:
                messages.insert(0, {"role": "system", "content": json_system_prompt})


        return self.chat_completion(messages=messages, temperature=temperature, **kwargs)