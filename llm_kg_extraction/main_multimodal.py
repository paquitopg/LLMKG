import os
from openai import AzureOpenAI
from multimodal_kg_builder import MultimodalFinancialKGBuilder
from KG_visualizer import KnowledgeGraphVisualizer
import sys
from pathlib import Path

from dotenv import load_dotenv

def main(project_name: str, model_name: str):
    """
    Main function to extract knowledge graph from a PDF file and visualize it.
    Args:
        name (str): The name of the project to be used for file paths.
        model_name (str): The name of the model to be used for extraction.
    """

    load_dotenv()
    pdf_path = Path(__file__).resolve().parents[3] / "pages" / project_name / f"Project_{project_name}_Teaser.pdf"
    deployment_name = os.getenv(f"AZURE_DEPLOYMENT_NAME_{model_name}")
    
    builder = MultimodalFinancialKGBuilder(model_name=model_name, deployment_name=deployment_name)

    kg = builder.build_knowledge_graph_from_pdf(pdf_path)
    builder.save_knowledge_graph(kg, project_name)

    visualizer = KnowledgeGraphVisualizer()
    visualizer.visualize(kg)

    output_path = str(Path(__file__).resolve().parents[3] / "outputs" / project_name / f"multimodal_knowledge_graph_{project_name}_{model_name}.html")

    builder.visualize_knowledge_graph(kg, project_name)
    print(f"Knowledge graph visualization saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run.py <project_name> <model_name>")
        sys.exit(1)
    project_name = sys.argv[1]
    model_name = sys.argv[2]

    main(project_name, model_name)
    print(f"Running with project name: {project_name} and model name: {model_name}")
