import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from llm_kg_extraction.KG_builder import FinancialKGBuilder
from llm_kg_extraction.KG_visualizer import KnowledgeGraphVisualizer
import sys
from pathlib import Path


def main(name: str, model_name: str):
    """
    Main function to extract knowledge graph from a PDF file and visualize it.
    Args:
        name (str): The name of the project to be used for file paths.
        model_name (str): The name of the model to be used for extraction.
    """

    load_dotenv()
    pdf_path = Path(__file__).resolve().parents[3] / "pages" / name / f"Project_{name}_Teaser.pdf"

    api_key = os.getenv(f"AZURE_OPENAI_API_KEY_{model_name}")
    api_version = os.getenv(f"AZURE_OPENAI_API_VERSION_{model_name}") 
    azure_endpoint = os.getenv(f"AZURE_OPENAI_ENDPOINT_{model_name}")
    deployment_name = os.getenv(f"AZURE_DEPLOYMENT_NAME_{model_name}")

    print(f"Using model: {model_name}")
    print(f"Using deployment name: {deployment_name}")
    print(f"Using PDF path: {pdf_path}")
    print(f"Using API key: {api_key}")
    print(f"Using API version: {api_version}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run.py <project_name> <model_name>")
        sys.exit(1)
    project_name = sys.argv[1]
    model_name = sys.argv[2]
    main(project_name, model_name)