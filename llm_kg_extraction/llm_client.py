from openai import AzureOpenAI
from dotenv import load_dotenv
import os

import google.generativeai as genai

class AzureOpenAIClient:
    """
    A class to interact with Azure OpenAI API.
    """

    def __init__(self, model_name: str):
        """
        Initialize the AzureOpenAIClient.
        Args:
            model_name (str): The model name suffix used for Azure environment variable lookup.
        """
        self.model_name = model_name
        self.client = self._make_client()

    def _make_client(self) -> AzureOpenAI:
        """
        Create an Azure OpenAI client.
        Returns:
            AzureOpenAI: The Azure OpenAI client.
        """
        load_dotenv()
        AZURE_OPENAI_ENDPOINT = os.getenv(f"AZURE_OPENAI_ENDPOINT_{self.model_name}")
        AZURE_OPENAI_API_KEY = os.getenv(f"AZURE_OPENAI_API_KEY_{self.model_name}")
        AZURE_OPENAI_API_VERSION = os.getenv(f"AZURE_OPENAI_API_VERSION_{self.model_name}")

        if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION]):
            missing_vars = [
                var for var, val in {
                    f"AZURE_OPENAI_ENDPOINT_{self.model_name}": AZURE_OPENAI_ENDPOINT,
                    f"AZURE_OPENAI_API_KEY_{self.model_name}": AZURE_OPENAI_API_KEY,
                    f"AZURE_OPENAI_API_VERSION_{self.model_name}": AZURE_OPENAI_API_VERSION,
                }.items() if not val
            ]
            raise ValueError(f"Missing Azure OpenAI environment variables for model type '{self.model_name}': {', '.join(missing_vars)}")

        return AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )


class VertexAIClient:
    """
    A class to interact with Google Gemini models via the google.generativeai SDK,
    configured to use Vertex AI as the backend if appropriate ADC are set.
    """

    def __init__(self, model_name: str = "gemini-1.5-pro-latest"):
        """
        Initialize the VertexAIClient with the Gemini model name.
        Args:
            model_name (str): The name of the Gemini model to use (e.g., "gemini-1.5-pro-latest").
                              This will be prefixed with "models/" if not already.
        """
        self.model_name = model_name
        load_dotenv() 

        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1") 

        if not self.project_id:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable not set. "
                "This is required for google.generativeai to correctly route to Vertex AI via ADC."
            )
        
        self.client = self._make_client()

    def _make_client(self) -> genai.GenerativeModel: 
        """
        Creates a GenerativeModel client from the google.generativeai library.
        Relies on Application Default Credentials (ADC) for authentication and project context.
        Returns:
            genai.GenerativeModel: The Google Generative AI model instance.
        Raises:
            RuntimeError: If model instantiation fails.
        """
        try:
            effective_model_name = self.model_name
            if not self.model_name.startswith("models/") and not self.model_name.startswith("projects/"):
                effective_model_name = f"models/{self.model_name}"

            print(f"Initializing google.generativeai.GenerativeModel with model name: {effective_model_name}")
            print(f" (Using ADC with GOOGLE_CLOUD_PROJECT='{self.project_id}' and location='{self.location}')")
            
            model = genai.GenerativeModel(model_name=effective_model_name)
            print(f"Successfully initialized google.generativeai.GenerativeModel for {effective_model_name}")
            return model
        
        except Exception as e:
            error_message = (
                f"Failed to initialize google.generativeai.GenerativeModel for model '{self.model_name}' "
                f"(effective name: '{effective_model_name}'). Project: '{self.project_id}'. Error: {e}"
            )
            print(error_message)
            raise RuntimeError(error_message) from e