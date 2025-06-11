import os
import sys
import argparse
from pathlib import Path
import time
import json
import uuid # For transform_request_id
from typing import Optional, Dict, Any, List

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import new modular components
from llm_integrations.azure_llm import AzureLLM
from llm_integrations.vertex_llm import VertexLLM
from llm_integrations.base_llm_wrapper import BaseLLMWrapper

from ontology_management.ontology_loader import PEKGOntology
from _1_document_ingestion.pdf_parser import PDFParser
from _2_context_understanding.base_context_identifier import BaseContextIdentifier
from _2_context_understanding.financial_teaser_context import FinancialTeaserContextIdentifier
from _3_knowledge_extraction.page_llm_processor import PageLLMProcessor
from _3_knowledge_extraction.kg_constructor_single_doc import KGConstructorSingleDoc
from _4_knowledge_graph_operations.page_level_merger import PageLevelMerger
from _4_knowledge_graph_operations.inter_document_merger import InterDocumentMerger
from visualization_tools.KG_visualizer import KnowledgeGraphVisualizer

# --- MODIFICATION: Import the new document scanning function ---
from core_components.document_scanner import discover_pdf_files 

try:
    from transform_json import transform_kg
except ImportError:
    print("Warning: Could not import 'transform_kg'. Transformation step will be skipped if requested.")
    transform_kg = None

def get_llm_client_for_project(llm_provider: str, model_name: str,
                               azure_model_env_suffix: Optional[str] = None) -> Optional[BaseLLMWrapper]:
    if llm_provider == "azure":
        actual_deployment_name = os.getenv(f"AZURE_DEPLOYMENT_NAME_{azure_model_env_suffix}")
        if not actual_deployment_name:
            print(f"Error: AZURE_DEPLOYMENT_NAME_{azure_model_env_suffix} not found for Azure model '{model_name}'.")
            return None
        print(f"Using Azure deployment: {actual_deployment_name} for model reference '{model_name}'")
        return AzureLLM(model_name=model_name, deployment_name=actual_deployment_name)
    elif llm_provider == "vertexai":
        gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT")
        gcp_location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not gcp_project:
            print("Error: GOOGLE_CLOUD_PROJECT env var not set for Vertex AI.")
            return None
        print(f"Vertex AI using Project: {gcp_project}, Location: {gcp_location}, Model: {model_name}")
        return VertexLLM(model_name=model_name, project_id=gcp_project, location=gcp_location)
    else:
        print(f"Error: Invalid llm_provider '{llm_provider}'.")
        return None


def get_context_identifier_for_project(doc_type_hint: str, llm_client: BaseLLMWrapper, project_name_for_context: str) -> BaseContextIdentifier:
    if doc_type_hint == "financial_teaser":
        return FinancialTeaserContextIdentifier(llm_client=llm_client, project_name=project_name_for_context)
    else:
        print(f"Warning: Unknown doc_type_hint '{doc_type_hint}'. Using a dummy context identifier.")
        class DummyContextIdentifier(BaseContextIdentifier):
            def identify_context(self, document_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
                return {"identified_document_type": doc_type_hint or "unknown"}
        return DummyContextIdentifier()


def run_project_pipeline(
    input_folder_path_str: str,
    project_name: str,
    llm_provider: str,
    main_model_name: str,
    main_azure_model_env_suffix: Optional[str],
    construction_mode: str,
    extraction_mode: str,
    doc_type_hint: str = "financial_teaser", # Global hint for all docs in this run
    recursive_scan: bool = True,
    num_workers_per_doc: int = 4,
    dump_page_kgs_for_docs: bool = False,
    transform_final_project_kg_flag: bool = False,
    output_base_dir_str: str = "outputs"
    ):
    output_base_path = Path(output_base_dir_str)
    project_output_path = output_base_path / project_name 
    project_output_path.mkdir(parents=True, exist_ok=True) 

    page_processing_llm_client = get_llm_client_for_project(
        llm_provider, main_model_name, main_azure_model_env_suffix
    )

    if not page_processing_llm_client: return

    context_llm_client: BaseLLMWrapper
    if llm_provider == "vertexai":
        context_model = "gemini-2.5-flash-preview-05-20"
        ctx_client_temp = get_llm_client_for_project("vertexai", context_model)
        context_llm_client = ctx_client_temp if ctx_client_temp else page_processing_llm_client
    elif llm_provider == "azure":
        context_model_conceptual = "gpt-4.1-mini"
        context_model_env_suffix = "GPT41MINI"
        ctx_client_temp = get_llm_client_for_project("azure", context_model_conceptual, context_model_env_suffix)
        context_llm_client = ctx_client_temp if ctx_client_temp else page_processing_llm_client
    else:
        context_llm_client = page_processing_llm_client

    print(f"Context ID Client: {context_llm_client.model_name} (Provider: {context_llm_client.__class__.__name__})")
    print(f"Page Processing Client: {page_processing_llm_client.model_name} (Provider: {page_processing_llm_client.__class__.__name__})")

    ontology_file_path = Path("ontology_management") / "ontologies" / "pekg_ontology_teasers.yaml"
    if not ontology_file_path.exists():
        script_dir_parent = Path(__file__).resolve().parent.parent
        ontology_file_path = script_dir_parent / "llm_kg_extraction" / "ontology_management" / "pekg_ontology_teasers.yaml"
        if not ontology_file_path.exists():
            print(f"Error: Ontology file not found at expected paths. Last tried: {ontology_file_path}")
            return
    print(f"Using ontology file: {ontology_file_path.resolve()}")
    ontology = PEKGOntology(ontology_path=str(ontology_file_path))

    page_llm_processor = PageLLMProcessor(llm_client=page_processing_llm_client, ontology=ontology)
    page_level_merger = PageLevelMerger()
    graph_visualizer = KnowledgeGraphVisualizer()
    context_identifier_instance = get_context_identifier_for_project(
        doc_type_hint, context_llm_client, project_name
    )
    inter_document_merger = InterDocumentMerger()

    pdf_documents = discover_pdf_files(Path(input_folder_path_str), recursive_scan)
    if not pdf_documents:
        print("No PDF documents found to process.")
        return

    all_document_kgs_with_ids: List[Dict[str, Any]] = []
    total_start_time = time.time()

    for doc_path in pdf_documents:
        doc_id = doc_path.stem
        print(f"\n--- Processing Document: {doc_path.name} (ID: {doc_id}) ---")
        
        current_doc_pdf_parser = PDFParser(pdf_path=str(doc_path))

        builder_config = {
            "project_name": project_name, # Overall project
            "document_id": doc_id,       # Specific ID for this document's outputs
            "extraction_mode": extraction_mode,
            "construction_mode": construction_mode,
            "output_base_path": str(output_base_dir_str), # KGConstructorSingleDoc will create subfolder for this doc
            "dump_intermediate_page_kgs": dump_page_kgs_for_docs,
            "doc_type_hint": doc_type_hint, # Global hint for this run
            "max_workers_parallel": num_workers_per_doc, # Correct key for KGConstructorSingleDoc
            "main_model_name_for_transform": main_model_name.replace("/", "_").replace(":", "_"),
            "llm_provider_for_transform": llm_provider,
            "ontology_file_for_transform": str(ontology_file_path)
        }

        try: #
            kg_constructor = KGConstructorSingleDoc( 
                pdf_parser=current_doc_pdf_parser, 
                context_identifier=context_identifier_instance, 
                page_llm_processor=page_llm_processor,         
                page_level_merger=page_level_merger,           
                graph_visualizer=graph_visualizer,           
                config=builder_config #
            ) #
            single_doc_kg = kg_constructor.build_kg() # 

            if single_doc_kg and (single_doc_kg.get("entities") or single_doc_kg.get("relationships")): #
                all_document_kgs_with_ids.append({ #
                    "document_id": doc_id, 
                    "entities": single_doc_kg.get("entities", []), #
                    "relationships": single_doc_kg.get("relationships", []) #
                }) #
            else: #
                print(f"Warning: KG construction for document '{doc_id}' resulted in an empty or invalid graph.") #
        except Exception as e: #
            print(f"Error processing document {doc_path.name}: {e}") #
            import traceback #
            traceback.print_exc() #

    if not all_document_kgs_with_ids: #
        print("No document KGs were successfully generated for merging.") #
        return #

    print(f"\n--- Starting Inter-Document Merge for {len(all_document_kgs_with_ids)} documents ---") #
    final_project_kg = inter_document_merger.merge_project_kgs(all_document_kgs_with_ids) #
    
    project_kg_filename_base = f"PROJECT_{project_name}_MERGED_KG" #
    project_kg_json_file = project_output_path / f"{project_kg_filename_base}.json"
    project_kg_html_file = project_output_path / f"{project_kg_filename_base}.html" 

    try: #
        with open(project_kg_json_file, "w") as f: #
            json.dump(final_project_kg, f, indent=2) #
        print(f"Final Project KG saved to {project_kg_json_file}") #
        if final_project_kg.get("entities"): #
            graph_visualizer.export_interactive_html(final_project_kg, str(project_kg_html_file)) #
            print(f"Final Project KG visualization saved to {project_kg_html_file}") #
    except Exception as e: #
        print(f"Error saving final project KG: {e}") #

    if final_project_kg and transform_final_project_kg_flag and transform_kg: #
        if project_kg_json_file.exists(): #
            req_id = str(uuid.uuid4()) #
            meta_title = f"Project_{project_name}_Merged" #
            print(f"Transforming final project KG with Request ID: '{req_id}', Title: '{meta_title}'...") #
            transformed_proj_dir = project_output_path / "transformed_project_kg_outputs" #

            transform_kg(  #
                input_file_path=str(project_kg_json_file),  #
                output_dir=str(transformed_proj_dir), #
                request_id_to_use=req_id, #
                meta_title_to_use=meta_title, #
                extraction_mode=extraction_mode,  #
                model_name=main_model_name.replace("/", "_").replace(":", "_"), #
                num_workers=1, # 
                llm_provider=llm_provider, #
                construction_mode="project_merged", # 
                ontology_file=str(ontology_file_path) #
            ) #
            print(f"Transformed project KG output saved in: {transformed_proj_dir}") #
        else: #
            print("Skipping transformation: Merged project KG file not found.") #

    total_end_time = time.time() #
    total_duration = total_end_time - total_start_time #
    print(f"\n--- Project KG Pipeline for '{project_name}' completed in {total_duration:.2f} seconds ---") #


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrator for KG extraction from multiple documents in a project folder.") #
    parser.add_argument("input_folder_path", help="Full path to the root folder containing PDF documents.") #
    parser.add_argument("project_name", help="Name of the project for organizing outputs.") #
    parser.add_argument("llm_provider", choices=['azure', 'vertexai'], help="LLM provider for main extraction.") #
    parser.add_argument("main_model_name", help="Name/ID of the main LLM for page extraction.") #
    parser.add_argument("main_azure_model_env_suffix", nargs='?', default=None, #
                        help="Suffix for Azure main model's deployment name env var (e.g., 'GPT4TURBO'). Required if llm_provider is azure.") #
    parser.add_argument("construction_mode", choices=['iterative', 'onego', 'parallel'], help="KG construction mode for each document.") #
    parser.add_argument("extraction_mode", choices=['text', 'multimodal'], help="KG extraction mode for each document.") #
    
    parser.add_argument("--doc_type_hint", default="financial_teaser", help="Global hint for document type for all docs (default: financial_teaser).") #
    parser.add_argument("--recursive_scan", action="store_true", help="Scan for PDFs in subfolders as well.") #
    parser.add_argument("--workers_per_doc", type=int, default=4, help="Number of parallel workers for processing each document (if its mode is 'parallel'). Default: 4.") #
    parser.add_argument("--dump_page_kgs", action="store_true", help="Save intermediate KGs for each page of each document.") #
    parser.add_argument("--transform_final_kg", action="store_true", help="Perform transformation on the final merged project KG.") #
    parser.add_argument("--output_dir", default="outputs", help="Base directory for all outputs.") #

    if len(sys.argv) <= 7: # 
        parser.print_help(sys.stderr) #
        sys.exit(1) #
        
    args = parser.parse_args() #

    if args.llm_provider == 'azure' and not args.main_azure_model_env_suffix: #
        print("Error: --main_azure_model_env_suffix is required when llm_provider is 'azure' for the main model.") #
        parser.print_help(sys.stderr) #
        sys.exit(1) #

    run_project_pipeline( 
        input_folder_path_str=args.input_folder_path,
        project_name=args.project_name,
        llm_provider=args.llm_provider.lower(),
        main_model_name=args.main_model_name,
        main_azure_model_env_suffix=args.main_azure_model_env_suffix,
        construction_mode=args.construction_mode.lower(),
        extraction_mode=args.extraction_mode.lower(),
        doc_type_hint=args.doc_type_hint.lower(),
        recursive_scan=args.recursive_scan,
        num_workers_per_doc=args.workers_per_doc,
        dump_page_kgs_for_docs=args.dump_page_kgs,
        transform_final_project_kg_flag=args.transform_final_kg,
        output_base_dir_str=args.output_dir
    )