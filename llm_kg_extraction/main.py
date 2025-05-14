import os
import sys
from pathlib import Path
import time
import json
from merged_kg_builder import FinancialKGBuilder
from dotenv import load_dotenv

def main(project_name: str, model_name: str, mode: str = "iterative", extraction_mode: str = "text", max_workers: int = 4, dump: bool = False):
    """
    Main function to extract knowledge graph from a PDF file and visualize it.
    Args:
        project_name (str): The name of the project to be used for file paths.
        model_name (str): The name of the model to be used for extraction.
        mode (str): The mode of operation: 'iterative', 'onego', or 'parallel'.
        extraction_mode (str): The modality of the knowledge graph, either 'text' or 'multimodal'.
        max_workers (int): The maximum number of workers for parallel processing (only used in parallel mode).
        dump (bool): Whether to dump the knowledge subgraphs to a file.
    """

    load_dotenv()

    deployment_name = os.getenv(f"AZURE_DEPLOYMENT_NAME_{model_name}")
    
    if extraction_mode == "text":
        print("Using textual modality for knowledge graph construction.")
    else:
        print("Using multimodal modality for knowledge graph construction.")
    
    builder = FinancialKGBuilder(
        model_name=model_name,
        deployment_name=deployment_name,
        project_name=project_name,
        construction_mode=mode,
        extraction_mode=extraction_mode,
        max_workers=max_workers,
    )
        
    if mode == "iterative":
        print("Building knowledge graph iteratively...")
    elif mode == "onego":
        print("Building knowledge graph in one go...")
    elif mode == "parallel":
        print(f"Building knowledge graph in parallel with {max_workers} workers...")

    start_time = time.time()
    kg = builder.build_knowledge_graph_from_pdf(dump=dump)
    builder.save_knowledge_graph(kg)
    end_time = time.time()

    duration = end_time - start_time
    print(f"Knowledge graph construction time: {duration:.2f} seconds")

    # Save timing data
    time_output_path = Path(f"tests/{extraction_mode}_construction_time.json")
    if extraction_mode == "text":
        key = f"knowledge_graph_{project_name}_{model_name}_{mode}"
    else:
        key = f"multimodal_knowledge_graph_{project_name}_{model_name}_{mode}"
        
    # Add worker count to key if using parallel mode
    if mode == "parallel":
        key += f"_workers{max_workers}"
        
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

def print_usage():
    """Print the usage instructions for the script."""
    print("Usage: python main.py <project_name> <model_name> <mode> <extraction_mode> [--workers N] [--dump]")
    print("  <project_name>: Name of the project")
    print("  <model_name>: Name of the model (e.g., gpt-4.1-mini)")
    print("  <mode>: 'iterative', 'onego', or 'parallel'")
    print("  <extraction_mode>: 'text' or 'multimodal'")
    print("  --workers N: Number of parallel workers (optional, only for parallel mode, default=4)")
    print("  --dump: Whether to save intermediate results (optional)")
    print("\nExamples:")
    print("  python main.py EXAMPLE gpt-4.1-mini iterative text --dump")
    print("  python main.py EXAMPLE gpt-4.1-mini parallel multimodal --workers 8 --dump")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print_usage()
        sys.exit(1)
        
    project_name = sys.argv[1]
    model_name = sys.argv[2]
    mode = sys.argv[3]
    extraction_mode = sys.argv[4]
    
    # Default values
    max_workers = 4
    dump = False
    
    # Process optional arguments
    i = 5
    while i < len(sys.argv):
        if sys.argv[i] == "--dump":
            dump = True
            i += 1
        elif sys.argv[i] == "--workers" and i + 1 < len(sys.argv):
            try:
                max_workers = int(sys.argv[i + 1])
                if max_workers <= 0:
                    print("Error: Number of workers must be a positive integer.")
                    print_usage()
                    sys.exit(1)
            except ValueError:
                print(f"Error: Invalid number of workers: {sys.argv[i + 1]}")
                print_usage()
                sys.exit(1)
            i += 2
        else:
            print(f"Error: Unknown argument: {sys.argv[i]}")
            print_usage()
            sys.exit(1)
    
    # Validate mode
    if mode not in ["iterative", "onego", "parallel"]:
        print("Error: Invalid mode. Choose 'iterative', 'onego', or 'parallel'.")
        print_usage()
        sys.exit(1)
    
    # Validate extraction mode
    if extraction_mode not in ["text", "multimodal"]:
        print("Error: Invalid extraction mode. Choose 'text' or 'multimodal'.")
        print_usage()
        sys.exit(1)
    
    # Validate workers for non-parallel mode
    if mode != "parallel" and max_workers != 4:
        print("Warning: --workers parameter is only used with 'parallel' mode. Ignoring it.")

    main(project_name, model_name, mode, extraction_mode, max_workers, dump)