import os
import sys
from pathlib import Path
import time
import json

from dotenv import load_dotenv

def main(project_name: str, model_name: str, mode: str = "iterative", modality: str = "textual", dump: bool = False):
    """
    Main function to extract knowledge graph from a PDF file and visualize it.
    Args:
        project_name (str): The name of the project to be used for file paths.
        model_name (str): The name of the model to be used for extraction.
        mode (str): The mode of operation, either 'iterative' or 'onego'.
        modality (str): The modality of the knowledge graph, either 'textual' or 'multimodal'.
        dump (bool): Whether to dump the knowledge subgraphs to a file.
    """

    load_dotenv()

    deployment_name = os.getenv(f"AZURE_DEPLOYMENT_NAME_{model_name}")
    
    if modality == "textual":
        print("Using textual modality for knowledge graph construction.")
        from KG_builder import FinancialKGBuilder
        builder = FinancialKGBuilder(model_name=model_name, deployment_name=deployment_name, project_name=project_name, construction_mode=mode)
    else:
        print("Using multimodal modality for knowledge graph construction.")
        from multimodal_KG_builder import MultimodalFinancialKGBuilder
        builder = MultimodalFinancialKGBuilder(model_name=model_name, deployment_name=deployment_name, project_name=project_name, construction_mode=mode)
        
    if mode == "iterative":
        print("Building knowledge graph iteratively...")
    else:
        print("Building knowledge graph in one go...")

    start_time = time.time()
    kg = builder.build_knowledge_graph_from_pdf(dump=dump)
    builder.save_knowledge_graph(kg)
    end_time = time.time()

    duration = end_time - start_time
    print(f"Knowledge graph construction time: {duration:.2f} seconds")

    time_output_path = Path(f"tests/{modality}_construction_time.json")
    if modality == "textual":
        key = f"knowledge_graph_{project_name}_{model_name}_{mode}"
    else:
        key = f"multimodal_knowledge_graph_{project_name}_{model_name}_{mode}"
    data = {}

    if time_output_path.exists():
        with open(time_output_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}

    data[key] = round(duration, 2)

    with open(time_output_path, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 5 and len(sys.argv) != 6:
        print("Usage: python main.py <project_name> <model_name> <mode> <modality> [--dump]")
        print("mode: 'iterative' or 'onego'")
        print("modality: 'textual' or 'multimodal'")
        print("Example: python main.py EXAMPLE gpt-4.1-mini iterative textual --dump")
        sys.exit(1)
        
    project_name = sys.argv[1]
    model_name = sys.argv[2]
    mode = sys.argv[3]
    modality = sys.argv[4]
    dump = False
    if len(sys.argv) == 6 and sys.argv[5] == "--dump":
        dump = True
        
    if mode not in ["iterative", "onego"]:
        print("Invalid mode. Choose 'iterative' or 'onego'.")
        sys.exit(1)

    main(project_name, model_name, mode, modality, dump)
