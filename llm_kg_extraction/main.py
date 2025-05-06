import os
from openai import AzureOpenAI
from KG_builder_iterative import FinancialKGBuilder
from KG_visualizer import KnowledgeGraphVisualizer
import sys
from pathlib import Path

from dotenv import load_dotenv

def main(name: str, model_name: str, mode: str = "iterative", dump: bool = False):
    """
    Main function to extract knowledge graph from a PDF file and visualize it.
    Args:
        name (str): The name of the project to be used for file paths.
        model_name (str): The name of the model to be used for extraction.
        mode (str): The mode of operation, either 'iterative' or 'onego'.
        dump (bool): Whether to dump the knowledge subgraphs to a file.
    """

    load_dotenv()

    deployment_name = os.getenv(f"AZURE_DEPLOYMENT_NAME_{model_name}")
    
    builder = FinancialKGBuilder(model_name=model_name, deployment_name=deployment_name, project_name=name, construction_mode=mode)

    if mode == "iterative":
        print("Building knowledge graph iteratively...")
    else:
        print("Building knowledge graph in one go...")

    kg = builder.build_knowledge_graph_from_pdf(dump=dump)
    builder.save_knowledge_graph(kg)

if __name__ == "__main__":
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print("Usage: python run.py <project_name> <model_name> <mode> [--dump]")
        print("mode: 'iterative' or 'batch'")
        sys.exit(1)
        
    project_name = sys.argv[1]
    model_name = sys.argv[2]
    mode = sys.argv[3]
    dump = False
    if len(sys.argv) == 5 and sys.argv[4] == "--dump":
        dump = True
        
    if mode not in ["iterative", "onego"]:
        print("Invalid mode. Choose 'iterative' or 'onego'.")
        sys.exit(1)

    main(project_name, model_name, mode, dump)
