import os
import sys
from pathlib import Path
import time
import json
import uuid 
from merged_kg_builder3 import FinancialKGBuilder 
from dotenv import load_dotenv

try:
    from transform_json import transform_kg 
except ImportError:
    print("Error: Could not import 'transform_kg' from 'transform_json.py'.")
    print("Please ensure 'transform_json.py' is in the same directory as 'main.py' or in the PYTHONPATH.")
    sys.exit(1)


def main(project_name: str, 
         llm_provider: str, 
         model_name: str, 
         mode: str = "iterative", 
         extraction_mode: str = "text", 
         max_workers: int = 4, 
         dump: bool = False,
         transform_output: bool = False
         ):
    """
    Main function to extract knowledge graph from a PDF file and optionally transform it.
    Args:
        project_name (str): The name of the project to be used for file paths.
        llm_provider (str): The LLM provider: "azure" or "vertexai".
        model_name (str): The name/identifier of the model.
        mode (str): The mode of operation: 'iterative', 'onego', or 'parallel'.
        extraction_mode (str): The modality of the knowledge graph, either 'text' or 'multimodal'.
        max_workers (int): The maximum number of workers for parallel processing.
        dump (bool): Whether to dump the knowledge subgraphs to a file.
        transform_output (bool): Whether to transform the final KG into meta, nodes, links files.
    """

    load_dotenv() 

    print(f"Selected LLM Provider: {llm_provider.upper()}")
    print(f"Using Model: {model_name}")

    azure_deployment_name = None
    if llm_provider == "azure":
        azure_deployment_name = os.getenv(f"AZURE_DEPLOYMENT_NAME_{model_name}")
        if not azure_deployment_name:
            print(f"Warning: AZURE_DEPLOYMENT_NAME_{model_name} not found in environment for Azure provider.")
            
    elif llm_provider == "vertexai":
        if not os.getenv("GOOGLE_CLOUD_PROJECT"):
            print("Error: GOOGLE_CLOUD_PROJECT environment variable not set. Required for Vertex AI.")
            sys.exit(1)
        print(f"Vertex AI will use GOOGLE_CLOUD_PROJECT: {os.getenv('GOOGLE_CLOUD_PROJECT')}")
    else:
        print(f"Error: Invalid llm_provider '{llm_provider}'. Choose 'azure' or 'vertexai'.")
        sys.exit(1)
    
    if extraction_mode == "text":
        print("Using textual modality for knowledge graph construction.")
    else:
        print("Using multimodal modality for knowledge graph construction.")
    
    try:
        builder = FinancialKGBuilder(
            llm_provider=llm_provider,
            model_name=model_name, 
            deployment_name=azure_deployment_name,
            project_name=project_name, 
            construction_mode=mode,
            extraction_mode=extraction_mode,
            max_workers=max_workers,
        )
    except ValueError as ve:
        print(f"Error initializing FinancialKGBuilder: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during FinancialKGBuilder initialization: {e}")
        sys.exit(1)
        
    if mode == "iterative": print("Building knowledge graph iteratively...")
    elif mode == "onego": print("Building knowledge graph in one go...")
    elif mode == "parallel": print(f"Building knowledge graph in parallel with {max_workers} workers...")

    start_time = time.time()
    kg_output_file_path_str = None 
    try:
        kg = builder.build_knowledge_graph_from_pdf(dump=dump)
        if kg: 
            filename_prefix = "multimodal" if builder.extraction_mode == "multimodal" else "text"
            provider_suffix = builder.llm_provider
            construction_details = builder.construction_mode
            if builder.construction_mode == "parallel":
                construction_details += f"_{builder.max_workers}w"
            model_name_sanitized = builder.model_name.replace("/", "_") # Ensure model name is filename-safe
            
            output_filename = f"{filename_prefix}_kg_{builder.project_name}_{model_name_sanitized}_{provider_suffix}_{construction_details}.json"
            kg_output_file_path = builder.json_output_path / output_filename 
            kg_output_file_path_str = str(kg_output_file_path) 
            
            builder.save_knowledge_graph(kg) 
            print(f"Knowledge graph saved to: {kg_output_file_path_str}")

            if transform_output:
                if kg_output_file_path_str:
                    current_transform_request_id = str(uuid.uuid4())
                    current_transform_meta_title = project_name 
                    
                    print(f"Transforming knowledge graph output with Request ID: '{current_transform_request_id}' and Title: '{current_transform_meta_title}'...")
                    transformed_output_dir = builder.json_output_path / "transformed_outputs" 
                    
                    # Pass the necessary parameters for dynamic filename generation
                    transform_kg( 
                        input_file_path=kg_output_file_path_str, 
                        output_dir=str(transformed_output_dir),
                        request_id_to_use=current_transform_request_id,
                        meta_title_to_use=current_transform_meta_title,
                        extraction_mode=builder.extraction_mode, # from builder
                        model_name=model_name_sanitized,         # use sanitized version
                        llm_provider=builder.llm_provider,       # from builder
                        construction_mode=builder.construction_mode # from builder
                    )
                else:
                    print("Skipping transformation because main KG file path is not available.")
        else:
            print("Knowledge graph construction resulted in an empty graph. Nothing to save or transform.")
    except Exception as e:
        print(f"An error occurred during knowledge graph construction, saving, or transformation: {e}")
        import traceback
        traceback.print_exc() 
    end_time = time.time()

    duration = end_time - start_time
    print(f"Total processing time (including potential transformation): {duration:.2f} seconds")

    time_output_dir = Path("tests") 
    time_output_dir.mkdir(parents=True, exist_ok=True)
    time_log_filename = f"{extraction_mode}_construction_time.json" 
    time_output_path = time_output_dir / time_log_filename
    
    key_parts = [llm_provider, extraction_mode, project_name, model_name.replace("/", "_"), mode]
    if mode == "parallel": key_parts.append(f"workers{max_workers}")
    if transform_output: key_parts.append("transformed")

    key = "_".join(key_parts)
        
    data = {}
    if time_output_path.exists():
        with open(time_output_path, "r") as f:
            try: data = json.load(f)
            except json.JSONDecodeError: data = {} 
    data[key] = round(duration, 2)
    with open(time_output_path, "w") as f:
        json.dump(data, f, indent=4)

def print_usage():
    """Print the usage instructions for the script."""
    print("Usage: python main.py <project_name> <llm_provider> <model_name> <mode> <extraction_mode> [options]")
    print("  <project_name>: Name of the project (e.g., EXAMPLE_PROJECT)")
    print("  <llm_provider>: LLM provider: 'azure' or 'vertexai'")
    print("  <model_name>: Name/ID of the model.")
    print("                  For 'azure', a suffix for env vars (e.g., GPT4).")
    print("                  For 'vertexai', the Gemini model ID (e.g., gemini-1.5-pro-latest).")
    print("  <mode>: 'iterative', 'onego', or 'parallel'")
    print("  <extraction_mode>: 'text' or 'multimodal'")
    print("\nOptions:")
    print("  --workers N: Number of parallel workers (optional, only for 'parallel' mode, default=4)")
    print("  --dump: Whether to save intermediate results (optional)")
    print("  --transform: Perform transformation into meta, nodes, and links JSON files (optional)")
    print("               (Request ID will be auto-generated, Meta Title will be project_name)")
    print("               (Filenames will include extraction_mode, model_name, llm_provider, construction_mode)")
    print("\nExamples:")
    print("  Azure: python main.py EXAMPLE_PROJECT azure GPT4 iterative text --dump")
    print("  VertexAI: python main.py EXAMPLE_PROJECT vertexai gemini-1.5-pro-latest parallel multimodal --workers 8 --dump")
    print("  With Transformation: python main.py MySpecialProject vertexai gemini-1.5-flash-latest onego text --transform")

if __name__ == "__main__":
    if len(sys.argv) < 6: 
        print_usage()
        sys.exit(1)
        
    project_name_arg = sys.argv[1] 
    llm_provider_arg = sys.argv[2].lower() 
    model_name_arg = sys.argv[3]
    mode_arg = sys.argv[4].lower() 
    extraction_mode_arg = sys.argv[5].lower() 
    
    max_workers_val = 4 
    dump_val = False 
    transform_output_val = False 
    
    i = 6
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--dump":
            dump_val = True
            i += 1
        elif arg == "--workers" and i + 1 < len(sys.argv):
            try:
                max_workers_val = int(sys.argv[i + 1])
                if max_workers_val <= 0:
                    print("Error: Number of workers must be a positive integer.")
                    print_usage()
                    sys.exit(1)
            except ValueError:
                print(f"Error: Invalid number of workers: {sys.argv[i + 1]}")
                print_usage()
                sys.exit(1)
            i += 2
        elif arg == "--transform":
            transform_output_val = True
            i += 1
        else:
            print(f"Error: Unknown argument or incorrect usage: {arg}")
            print_usage()
            sys.exit(1)
    
    if llm_provider_arg not in ["azure", "vertexai"]:
        print(f"Error: Invalid llm_provider '{llm_provider_arg}'. Choose 'azure' or 'vertexai'.")
        print_usage()
        sys.exit(1)

    if mode_arg not in ["iterative", "onego", "parallel"]:
        print(f"Error: Invalid mode '{mode_arg}'. Choose 'iterative', 'onego', or 'parallel'.")
        print_usage()
        sys.exit(1)
    
    if extraction_mode_arg not in ["text", "multimodal"]:
        print(f"Error: Invalid extraction_mode '{extraction_mode_arg}'. Choose 'text' or 'multimodal'.")
        print_usage()
        sys.exit(1)
    
    user_passed_workers_arg = False
    temp_idx = 6 
    while temp_idx < len(sys.argv):
        if sys.argv[temp_idx] == "--workers":
            user_passed_workers_arg = True
            break
        if sys.argv[temp_idx] in ["--dump", "--transform"]: temp_idx += 1
        else: temp_idx += 2 
            
    if mode_arg != "parallel" and user_passed_workers_arg:
        print("Warning: --workers parameter is only used with 'parallel' mode. It will be ignored.")

    main(project_name=project_name_arg, 
         llm_provider=llm_provider_arg, 
         model_name=model_name_arg, 
         mode=mode_arg, 
         extraction_mode=extraction_mode_arg, 
         max_workers=max_workers_val, 
         dump=dump_val,
         transform_output=transform_output_val
         )
