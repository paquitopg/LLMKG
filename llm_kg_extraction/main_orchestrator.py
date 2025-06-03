import os
import sys
import argparse
from pathlib import Path
import time
import json
import uuid # For transform_request_id
from typing import Optional, Dict, Any

# Load environment variables (e.g., API keys)
from dotenv import load_dotenv
load_dotenv()

# Import new modular components
# Adjust paths as per your final project structure (e.g., if LLMKG is a package)
from llm_integrations.azure_llm import AzureLLM
from llm_integrations.vertex_llm import VertexLLM
from llm_integrations.base_llm_wrapper import BaseLLMWrapper

from ontology_management.ontology_loader import PEKGOntology

from _1_document_ingestion.pdf_parser import PDFParser

from _2_context_understanding.base_context_identifier import BaseContextIdentifier
from _2_context_understanding.financial_teaser_context import FinancialTeaserContextIdentifier
# Import other context identifiers here as they are created (e.g., ContractContextIdentifier)

from _3_knowledge_extraction.page_llm_processor import PageLLMProcessor
from _3_knowledge_extraction.kg_constructor_single_doc import KGConstructorSingleDoc

from _4_knowledge_graph_operations.page_level_merger import PageLevelMerger
# common_kg_utils will be used internally by merger and constructor

from visualization_tools.KG_visualizer import KnowledgeGraphVisualizer

# Import the transformation script if needed at this level
try:
    from transform_json import transform_kg # Assuming it's moved into the package
except ImportError:
    print("Warning: Could not import 'transform_kg'. Transformation step will be skipped if requested.")
    transform_kg = None


def get_llm_client(llm_provider: str, model_name: str, 
                   azure_deployment_name_env_suffix: Optional[str] = None # Suffix for Azure env var lookup
                  ) -> Optional[BaseLLMWrapper]:
    """Initializes and returns the appropriate LLM client."""
    if llm_provider == "azure":
        # For Azure, model_name could be "gpt-4-turbo" (conceptual name)
        # and azure_deployment_name_env_suffix is "GPT_4_TURBO" or "GPT_4_1_MINI"
        # used to find AZURE_DEPLOYMENT_NAME_GPT_4_TURBO in .env
        actual_deployment_name = os.getenv(f"AZURE_DEPLOYMENT_NAME_{azure_deployment_name_env_suffix}")
        if not actual_deployment_name:
            print(f"Error: AZURE_DEPLOYMENT_NAME_{azure_deployment_name_env_suffix} not found in environment for Azure model '{model_name}'.")
            return None
        print(f"Using Azure deployment: {actual_deployment_name} for model reference '{model_name}'")
        # The model_name passed to AzureLLM can be the conceptual name, 
        # but deployment_name is what's used for calls.
        return AzureLLM(model_name=model_name, deployment_name=actual_deployment_name)
    
    elif llm_provider == "vertexai":
        gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT")
        gcp_location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not gcp_project:
            print("Error: GOOGLE_CLOUD_PROJECT environment variable not set. Required for Vertex AI.")
            return None
        print(f"Vertex AI using Project: {gcp_project}, Location: {gcp_location}, Model: {model_name}")
        return VertexLLM(model_name=model_name, project_id=gcp_project, location=gcp_location)
    else:
        print(f"Error: Invalid llm_provider '{llm_provider}'. Choose 'azure' or 'vertexai'.")
        return None

def get_context_identifier(doc_type_hint: str, llm_client: BaseLLMWrapper, project_name: str) -> Optional[BaseContextIdentifier]:
    """Selects and initializes the appropriate context identifier."""
    if doc_type_hint == "financial_teaser":
        return FinancialTeaserContextIdentifier(llm_client=llm_client, project_name=project_name)
    # Add other document types here:
    # elif doc_type_hint == "contract":
    #     return ContractContextIdentifier(llm_client=llm_client, project_name=project_name)
    else:
        print(f"Warning: Unknown or unsupported doc_type_hint '{doc_type_hint}'. No specific context identification will be performed.")
        # Optionally return a default/dummy context identifier
        return None


def main_orchestrate(
    document_path_str: str,
    project_name: str,
    llm_provider: str,
    main_model_name: str,
    main_azure_model_env_suffix: Optional[str], # Suffix for MAIN Azure model's deployment name
    construction_mode: str,
    extraction_mode: str,
    num_workers: int = 16, # Default number of workers for parallel mode
    doc_type_hint: str = "financial_teaser",
    dump_page_kgs: bool = False,
    transform_output_flag: bool = False,
    output_base_dir_str: str = "outputs"
):
    """
    Main orchestration function for processing a single document.
    """
    print(f"--- Starting KG Construction for Document: {document_path_str} ---")
    print(f"Project: {project_name}, LLM Provider: {llm_provider.upper()}, Model: {main_model_name}")
    print(f"Construction: {construction_mode}, Extraction: {extraction_mode}, Doc Type Hint: {doc_type_hint}")

    document_path = Path(document_path_str)
    if not document_path.exists() or not document_path.is_file():
        print(f"Error: Document not found at {document_path_str}")
        return

    output_base_path = Path(output_base_dir_str)
    document_id = document_path.stem # Use filename (without ext) as document_id

    # 1. Initialize MAIN LLM Client (for PageLLMProcessor and as fallback for context)
    print(f"Initializing main LLM client: Provider={llm_provider}, Model={main_model_name}")
    page_processing_llm_client = get_llm_client(
        llm_provider, 
        main_model_name, 
        main_azure_model_env_suffix # Suffix for the *main* Azure model deployment name lookup
    )
    if not page_processing_llm_client:
        print("Error: Failed to initialize main LLM client. Exiting.")
        return

    # 2. Determine and Initialize LLM client for Context Identification (Hardcoded logic)
    context_identification_llm_client: BaseLLMWrapper # Type hint
    
    print(f"--- Determining LLM client for Context Identification (based on main provider: {llm_provider.upper()}) ---")
    if llm_provider == "vertexai":
        context_model_vertex_specific = "gemini-2.5-flash-preview-05-20" # Hardcoded
        print(f"Attempting to initialize specific context ID client: Vertex AI model '{context_model_vertex_specific}'")
        temp_context_client = get_llm_client(
            llm_provider="vertexai", # Explicitly "vertexai"
            model_name=context_model_vertex_specific
            # azure_deployment_name_env_suffix is not used by get_llm_client for vertexai
        )
        if temp_context_client:
            context_identification_llm_client = temp_context_client
            print(f"Successfully initialized specific context ID client: Vertex AI model '{context_model_vertex_specific}'")
        else:
            print(f"Warning: Failed to initialize specific context ID client for Vertex AI ('{context_model_vertex_specific}'). "
                  f"Falling back to main LLM client for context identification.")
            context_identification_llm_client = page_processing_llm_client
            
    elif llm_provider == "azure":
        # Define the conceptual name and the suffix for environment variable lookup for the context model
        context_model_azure_conceptual_name = "gpt-4.1-mini" # Hardcoded conceptual name
        # Derive/define the suffix for its AZURE_DEPLOYMENT_NAME_... environment variable
        context_model_azure_env_suffix_derived = "GPT41MINI" # Example: Or derive: context_model_azure_conceptual_name.upper().replace('-', '_').replace('.', '')
        
        print(f"Attempting to initialize specific context ID client: Azure model '{context_model_azure_conceptual_name}' "
              f"(env suffix for deployment: '{context_model_azure_env_suffix_derived}')")
        temp_context_client = get_llm_client(
            llm_provider="azure", # Explicitly "azure"
            model_name=context_model_azure_conceptual_name, 
            azure_deployment_name_env_suffix=context_model_azure_env_suffix_derived
        )
        if temp_context_client:
            context_identification_llm_client = temp_context_client
            print(f"Successfully initialized specific context ID client: Azure model '{context_model_azure_conceptual_name}'")
        else:
            print(f"Warning: Failed to initialize specific context ID client for Azure ('{context_model_azure_conceptual_name}'). "
                  f"Falling back to main LLM client for context identification.")
            context_identification_llm_client = page_processing_llm_client
    else:
        # This case should ideally be caught by CLI argument validation for the main llm_provider
        print(f"Critical Error: Main llm_provider '{llm_provider}' is unknown. Cannot set up context client. Exiting.")
        return

    # Now, context_identification_llm_client is set (either specific or fallback)
    # And page_processing_llm_client is the main client

    # 3. Initialize other components (passing the correct LLM clients)
    try:
        pdf_parser = PDFParser(pdf_path=str(document_path_str))
        
        # --- Ontology Path Logic ---
        ontology_file_path_str = str(Path(__file__).resolve().parent / "ontology_management" / "pekg_ontology_teasers.yaml")
        # Add your fallbacks for ontology path here if needed, as in your previous version:
        if not Path(ontology_file_path_str).exists():
            ontology_file_path_str = str(Path("llm_kg_extraction") / "ontology_management" / "ontologies" / "pekg_ontology.yaml")
            if not Path(ontology_file_path_str).exists():
                script_dir = Path(__file__).resolve().parent
                original_ontology_path = script_dir.parent / "ontology_management" / "ontologies" / "pekg_ontology_teasers.yaml" # Adjust if different
                # Check the original path used in the last successful run log by the user
                # Using ontology file: C:\PE\REPOS\llm_kg_extraction\llm_kg_extraction\ontology_management\ontologies\pekg_ontology_teasers.yaml
                # This means the path should likely be relative to where main_orchestrator.py is or a fixed path.
                # Let's assume a path relative to the project root (parent of llm_kg_extraction package)
                # project_root = Path(__file__).resolve().parents[1] # Assuming main_orchestrator is in llm_kg_extraction/
                # ontology_file_path_str = str(project_root / "ontology_management" / "ontologies" / "pekg_ontology_teasers.yaml")
                # For simplicity, using the last known working path structure relative to potentially the package root:
                ontology_file_path_str = str(Path("ontology_management") / "ontologies" / "pekg_ontology_teasers.yaml") # If run from llm_kg_extraction (inner)
                if not Path(ontology_file_path_str).exists():
                     # One level up if main_orchestrator is in llm_kg_extraction/llm_kg_extraction
                     ontology_file_path_str = str(Path(__file__).resolve().parent.parent / "ontology_management" / "ontologies" / "pekg_ontology_teasers.yaml")

                if not Path(ontology_file_path_str).exists():
                    print(f"Error: Ontology file not found. Please check ontology path configuration.")
                    # It seems the log showed: C:\PE\REPOS\llm_kg_extraction\llm_kg_extraction\ontology_management\ontologies\pekg_ontology_teasers.yaml
                    # If main_orchestrator.py is in C:\PE\repos\llm_kg_extraction\llm_kg_extraction\
                    # then a relative path from there is just "ontology_management/ontologies/pekg_ontology_teasers.yaml"
                    return
                else:
                    print(f"Using ontology file: {Path(ontology_file_path_str).resolve()}")
        
        ontology = PEKGOntology(ontology_path=ontology_file_path_str)

        # Pass the dedicated context_identification_llm_client
        context_identifier = get_context_identifier(doc_type_hint, context_identification_llm_client, project_name)
        if not context_identifier: 
            class DummyContextIdentifier(BaseContextIdentifier): # Define if not already global
                def identify_context(self, document_path: Optional[str] = None, document_content_parts: Optional[Dict[str, Any]] = None, doc_type_hint: Optional[str] = "financial_teaser") -> Dict[str, Any]: return {"identified_document_type": "unknown"}
            context_identifier = DummyContextIdentifier()
            print("Using dummy context identifier as specific one failed or was not found.")

        # Pass the dedicated page_processing_llm_client
        page_llm_processor = PageLLMProcessor(llm_client=page_processing_llm_client, ontology=ontology)
        page_level_merger = PageLevelMerger()
        graph_visualizer = KnowledgeGraphVisualizer()

    except Exception as e:
        print(f"Error during component initialization: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Prepare config for KGConstructorSingleDoc
    # model_name_sanitized is used for filenames in original main.py
    main_model_name_sanitized = main_model_name.replace("/", "_").replace(":", "_")

    builder_config = {
        "project_name": project_name,
        "document_id": document_id,
        "extraction_mode": extraction_mode,
        "construction_mode": construction_mode,
        "output_base_path": str(output_base_path),
        "dump_intermediate_page_kgs": dump_page_kgs,
        "doc_type_hint": doc_type_hint,
        "num_workers": num_workers,  # Only used in parallel mode
        "main_model_name_for_transform": main_model_name_sanitized,
        "llm_provider_for_transform": llm_provider,
        "ontology_file_for_transform": ontology_file_path_str
    }

    # 4. Instantiate and run KGConstructorSingleDoc
    start_time = time.time()
    final_kg = None
    try:
        kg_constructor = KGConstructorSingleDoc(
            pdf_parser=pdf_parser,
            context_identifier=context_identifier,
            page_llm_processor=page_llm_processor,
            page_level_merger=page_level_merger,
            graph_visualizer=graph_visualizer,
            config=builder_config
        )
        final_kg = kg_constructor.build_kg() # This method now also saves the KG
    except Exception as e:
        print(f"An error occurred during KG construction for document '{document_path_str}': {e}")
        import traceback
        traceback.print_exc()
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"--- KG Construction for Document '{document_path_str}' finished in {duration:.2f} seconds ---")

    # 5. Optional Transformation (if KG was successfully built and saved)
    # KGConstructorSingleDoc already saves the file. We need its path.
    # The path is: output_base_path / project_name / f"{document_id}_kg" / "full_document_kg.json"
    if final_kg and transform_output_flag and transform_kg:
        kg_output_file_path = output_base_path / project_name / f"{document_id}_kg" / "full_document_kg.json"
        if kg_output_file_path.exists():
            current_transform_request_id = str(uuid.uuid4())
            # Use document_id or project_name for meta title
            current_transform_meta_title = f"{project_name}_{document_id}" 
            
            print(f"Transforming KG for '{document_id}' with Request ID: '{current_transform_request_id}' and Title: '{current_transform_meta_title}'...")
            transformed_output_dir = kg_output_file_path.parent / "transformed_outputs" # Place it alongside the KG
            
            transform_kg( 
                input_file_path=str(kg_output_file_path), 
                output_dir=str(transformed_output_dir),
                request_id_to_use=current_transform_request_id,
                meta_title_to_use=current_transform_meta_title,
                extraction_mode=builder_config["extraction_mode"],
                model_name=builder_config["main_model_name_for_transform"],
                num_workers=builder_config["num_workers"],
                llm_provider=builder_config["llm_provider_for_transform"],
                construction_mode=builder_config["construction_mode"],
                ontology_file=builder_config["ontology_file_for_transform"]
            )
            print(f"Transformed output saved in: {transformed_output_dir}")
        else:
            print(f"Skipping transformation: KG file not found at {kg_output_file_path}")
    elif transform_output_flag and not transform_kg:
        print("Skipping transformation: transform_kg function not available.")
    elif transform_output_flag and not final_kg:
        print("Skipping transformation: KG construction failed or resulted in an empty graph.")


    # 6. Log processing time (similar to original main.py)
    time_output_dir = Path("outputs") / "processing_times" # Store times in a subfolder of outputs
    time_output_dir.mkdir(parents=True, exist_ok=True)
    time_log_filename = f"{extraction_mode}_construction_time.json" 
    time_output_path = time_output_dir / time_log_filename
    
    key_parts = [
        llm_provider, 
        extraction_mode, 
        project_name, 
        document_id, 
        main_model_name_sanitized, 
        construction_mode
    ]
    if construction_mode == "parallel": 
        key_parts.append(f"workers{num_workers}")
    if transform_output_flag: 
        key_parts.append("transformed")

    log_key = "_".join(key_parts)

    print(f"Logging processing time for key: {log_key}")

    time_data = {}
    if time_output_path.exists():
        with open(time_output_path, "r") as f:
            try: 
                time_data = json.load(f)
            except json.JSONDecodeError: 
                time_data = {} 
    time_data[log_key] = round(duration, 2)
    with open(time_output_path, "w") as f:
        json.dump(time_data, f, indent=4)
    print(f"Processing time logged to: {time_output_path}")


def print_orchestrator_usage():
    """Print the usage instructions for the main_orchestrator.py script."""
    print("Usage: python main_orchestrator.py <document_path> <project_name> <llm_provider> <model_name> <construction_mode> <extraction_mode> [options]")
    print("\nRequired Arguments:")
    print("  <document_path>: Full path to the PDF document to process.")
    print("  <project_name>: Name of the project (e.g., EXAMPLE_PROJECT) for organizing outputs.")
    print("  <llm_provider>: LLM provider: 'azure' or 'vertexai'.")
    print("  <model_name>: Name/ID of the LLM (e.g., 'gpt-4-turbo' for Azure, 'gemini-1.5-pro-latest' for Vertex).")
    print("  <construction_mode>: 'iterative', 'onego', or 'parallel'.")
    print("  <extraction_mode>: 'text' or 'multimodal'.")
    print("\nOptional Arguments:")
    print("  --doc_type_hint TYPE: Hint for document type (e.g., 'financial_teaser', 'contract'). Default: 'financial_teaser'.")
    print("  --workers N: Number of parallel workers for 'parallel' mode. Default: 16.")
    print("  --dump_page_kgs: Save intermediate KGs for each page. Default: False.")
    print("  --transform: Perform transformation into meta, nodes, and links JSON files. Default: False.")
    print("  --output_dir DIR: Base directory for all outputs. Default: './outputs'.")
    print("\nExamples:")
    print("  Azure: python main_orchestrator.py /path/to/doc.pdf MyProject azure gpt-4-turbo iterative text --dump_page_kgs")
    print("  VertexAI: python main_orchestrator.py ./docs/report.pdf ProjectAlpha vertexai gemini-1.5-pro-latest parallel multimodal --workers 8 --transform")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main orchestrator for Knowledge Graph extraction from a single document.")
    parser.add_argument("document_path", help="Full path to the PDF document.")
    parser.add_argument("project_name", help="Name of the project for organizing outputs.")
    parser.add_argument("llm_provider", choices=['azure', 'vertexai'], help="LLM provider.")
    parser.add_argument("main_model_name", help="Name/ID of the LLM.")
    parser.add_argument("construction_mode", choices=['iterative', 'onego', 'parallel'], help="KG construction mode.")
    parser.add_argument("extraction_mode", choices=['text', 'multimodal'], help="KG extraction mode.")
    
    parser.add_argument("--doc_type_hint", default="financial_teaser", help="Hint for document type (default: financial_teaser).")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4).")
    parser.add_argument("--dump_page_kgs", action="store_true", help="Save intermediate KGs for each page.")
    parser.add_argument("--transform", action="store_true", help="Perform output transformation.")
    parser.add_argument("--output_dir", default="outputs", help="Base directory for outputs (default: ./outputs).")

    if len(sys.argv) == 1: # No arguments provided
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.construction_mode != "parallel" and any(arg == "--workers" for arg in sys.argv):
        print("Warning: --workers parameter is only used with 'parallel' mode. It will be ignored if specified otherwise unless explicitly checked by user script parts.")

    main_orchestrate(
        document_path_str=args.document_path,
        project_name=args.project_name,
        llm_provider=args.llm_provider.lower(),
        main_model_name=args.main_model_name,#
        main_azure_model_env_suffix=args.main_model_name.upper().replace("-", "_").replace(".", "_") if args.llm_provider.lower() == "azure" else None,
        construction_mode=args.construction_mode.lower(),
        extraction_mode=args.extraction_mode.lower(),
        doc_type_hint=args.doc_type_hint.lower(),
        num_workers=args.workers,
        dump_page_kgs=args.dump_page_kgs,
        transform_output_flag=args.transform,
        output_base_dir_str=args.output_dir
    )