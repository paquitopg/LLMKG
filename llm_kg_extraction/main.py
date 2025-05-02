import os
from openai import AzureOpenAI
from KG_builder import FinancialKGBuilder
from KG_visualizer import KnowledgeGraphVisualizer
import sys
from pathlib import Path

from dotenv import load_dotenv

def main(name: str, model_name: str):
    """
    Main function to extract knowledge graph from a PDF file and visualize it.
    Args:
        name (str): The name of the project to be used for file paths.
        model_name (str): The name of the model to be used for extraction.
    """

    load_dotenv()
    pdf_path = Path(__file__).resolve().parents[3] / "pages" / name / f"Project_{name}_Teaser.pdf"

    deployment_name = os.getenv(f"AZURE_DEPLOYMENT_NAME_{model_name}")

    builder = FinancialKGBuilder(model_name=model_name, deployment_name=deployment_name)
    text = builder.extract_text_from_pdf(pdf_path)
    kg = builder.analyze_text_with_llm(text)
    builder.save_knowledge_graph(kg, name)

    visualizer = KnowledgeGraphVisualizer()
    visualizer.visualize(kg)

    output_dir = Path(__file__).resolve().parents[1] / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = str(output_dir / f"knowledge_graph_{name}_{model_name}.html")
    visualizer.export_interactive_html(kg, output_path=output_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run.py <project_name> <model_name>")
        sys.exit(1)
    project_name = sys.argv[1]
    model_name = sys.argv[2]
    main(project_name, model_name)
