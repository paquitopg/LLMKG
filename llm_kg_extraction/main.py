import os
from openai import AzureOpenAI
from KG_builder import FinancialKGBuilder
from KG_visualizer import KnowledgeGraphVisualizer
import sys
from pathlib import Path

from dotenv import load_dotenv

def main(name: str, model_name: str, mode: str = "iterative"):
    """
    Main function to extract knowledge graph from a PDF file and visualize it.
    Args:
        name (str): The name of the project to be used for file paths.
        model_name (str): The name of the model to be used for extraction.
        mode (str): The mode of operation, either 'iterative' or 'batch'.
    """

    load_dotenv()
    pdf_path = Path(__file__).resolve().parents[3] / "pages" / name / f"Project_{name}_Teaser.pdf"

    deployment_name = os.getenv(f"AZURE_DEPLOYMENT_NAME_{model_name}")
    
    if mode == 'batch':
        from KG_builder import FinancialKGBuilder
    else : 
        from KG_builder_iterative import FinancialKGBuilder
    
    builder = FinancialKGBuilder(model_name=model_name, deployment_name=deployment_name)
    kg = builder.build_knowledge_graph_from_pdf(pdf_path)
    builder.save_knowledge_graph(kg, name)

    visualizer = KnowledgeGraphVisualizer()
    visualizer.visualize(kg)

    output_dir = Path(__file__).resolve().parents[3] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = str(output_dir / f"knowledge_graph_{name}_{model_name}_{mode}.html")
    visualizer.export_interactive_html(kg, output_path=output_path)
    print(f"Knowledge graph saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run.py <project_name> <model_name> <mode>")
        print("mode: 'iterative' or 'batch'")
        sys.exit(1)
    project_name = sys.argv[1]
    model_name = sys.argv[2]
    mode = sys.argv[3]
    main(project_name, model_name, mode)
