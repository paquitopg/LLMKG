import os
from openai import AzureOpenAI
from multimodal_kg_builder_iterative import MultimodalFinancialKGBuilder
from KG_visualizer import KnowledgeGraphVisualizer
import sys
from pathlib import Path

from dotenv import load_dotenv

def main(project_name: str, model_name: str, dump: bool = False) -> None:
    """
    Main function to extract knowledge graph from a PDF file and visualize it.
    Args:
        project_name (str): The name of the project to be used for file paths.
        model_name (str): The name of the model to be used for extraction.
        dump (bool, optional): Flag to indicate if the knowledge subgraphs should be saved. Defaults to False.
    """

    load_dotenv()
    pdf_path = Path(__file__).resolve().parents[3] / "pages" / project_name / f"Project_{project_name}_Teaser.pdf"
    deployment_name = os.getenv(f"AZURE_DEPLOYMENT_NAME_{model_name}")
    
    builder = MultimodalFinancialKGBuilder(model_name=model_name, deployment_name=deployment_name)

    kg = builder.build_knowledge_graph_from_pdf(pdf_path, dump=dump)
    builder.save_knowledge_graph(kg, project_name)

    if dump:
        print("Knowledge subgraphs have been saved.")

    visualizer = KnowledgeGraphVisualizer()
    visualizer.visualize(kg)

    output_path = str(Path(__file__).resolve().parents[3] / "outputs" / project_name / f"multimodal_knowledge_graph_{project_name}_{model_name}.html")

    builder.visualize_knowledge_graph(kg, project_name)
    print(f"Knowledge graph visualization saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main_multimodal.py <project_name> <model_name> [--dump]")
        sys.exit(1)

    project_name = sys.argv[1]
    model_name = sys.argv[2]
    dump = False
    if len(sys.argv) == 4 and sys.argv[3] == "--dump":
        dump = True

    main(project_name, model_name, dump=dump)
        
