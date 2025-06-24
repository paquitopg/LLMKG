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
from _1_document_ingestion.document_classifier import DocumentClassifier

from _2_context_understanding.document_context_preparer import DocumentContextPreparer

from _3_knowledge_extraction.page_llm_processor import PageLLMProcessor
from _3_knowledge_extraction.kg_constructor_single_doc import KGConstructorSingleDoc

# Use the updated PEKG-aware mergers
from _4_knowledge_graph_operations.page_level_merger import PageLevelMerger
from _4_knowledge_graph_operations.inter_document_merger import InterDocumentMerger

# Use the upgraded visualizer
from visualization_tools.KG_visualizer import KnowledgeGraphVisualizer

from core_components.document_scanner import discover_pdf_files

# Optional diagnostic tool
try:
    from REPOS.llm_kg_extraction.llm_kg_extraction.merger_diagnostic_tool import diagnose_merger_issues
    DIAGNOSTIC_AVAILABLE = True
except ImportError:
    print("Note: Diagnostic tool not available. Install if you want merger analysis.")
    DIAGNOSTIC_AVAILABLE = False

try:
    from REPOS.llm_kg_extraction.llm_kg_extraction._5_transformations.transform_json import transform_kg
    TRANSFORM_AVAILABLE = True
except ImportError:
    print("Note: Transform functionality not available.")
    TRANSFORM_AVAILABLE = False


def get_llm_client(llm_provider: str, model_name: str, azure_model_env_suffix: Optional[str] = None) -> BaseLLMWrapper:
    """Helper function to initialize the correct LLM client."""
    if llm_provider == "azure":
        if not azure_model_env_suffix:
            raise ValueError("azure_model_env_suffix must be provided for Azure LLM.")
        if not model_name:
            model_name = "gpt-4o"  # Default to GPT-4o if not specified
        return AzureLLM(model_name=model_name, deployment_name=model_name, azure_model_env_suffix=azure_model_env_suffix)
    elif llm_provider == "vertexai":
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION")
        if not project_id or not location:
            print("Warning: GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION not set in environment variables. VertexAI client might fail.")
        if not model_name:
            model_name = os.getenv("VERTEXAI_DEFAULT_MODEL") or "gemini-1.5-pro"  # Provide fallback
        return VertexLLM(model_name=model_name, project_id=project_id, location=location)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

def setup_project_structure(project_name: str) -> Path:
    """Setup project directory structure and return project path."""
    base_output_path = Path(str("C:/PE/REPOS/outputs"))
    project_output_path = base_output_path / project_name
    project_output_path.mkdir(parents=True, exist_ok=True)
    print(f"Project output directory: {project_output_path}")
    return project_output_path

def initialize_components(llm_provider: str, model_configs: Dict[str, Any], ontology_path: Path) -> Dict[str, Any]:
    """Initialize all core components needed for the pipeline."""
    
    # Initialize LLM clients
    main_llm_client = get_llm_client(
        llm_provider, 
        model_configs["main_model_name"], 
        model_configs.get("main_azure_model_env_suffix")
    )
    classification_llm_client = get_llm_client(
        llm_provider, 
        model_configs["classification_model_name"], 
        model_configs.get("classification_azure_model_env_suffix")
    )
    summary_llm_client = get_llm_client(
        llm_provider, 
        model_configs["summary_model_name"], 
        model_configs.get("summary_azure_model_env_suffix")
    )

    # Load ontology
    if not ontology_path.exists():
        print(f"Error: Ontology file not found at {ontology_path}")
        sys.exit(1)

    print("Using ontology file:", model_configs.get("use_ontology"))
    if model_configs.get("use_ontology", True):
        ontology = PEKGOntology(ontology_path=str(ontology_path))
    else:
        print("Ontology loading skipped as per configuration.")
        ontology = None

    # Initialize processing components
    # Handle case where ontology might be None
    use_ontology_flag = model_configs.get("use_ontology", True) and ontology is not None
    
    components = {
        "main_llm_client": main_llm_client,
        "ontology": ontology,
        "document_classifier": DocumentClassifier(
            llm_client=classification_llm_client,
            categories=model_configs["predefined_categories"],
            summary_llm_client=summary_llm_client
        ),
        "document_context_preparer": DocumentContextPreparer(ontology=ontology or PEKGOntology(""), use_ontology=use_ontology_flag),
        "page_level_merger": PageLevelMerger(ontology=ontology),
        "inter_document_merger": InterDocumentMerger(ontology=ontology),
        "graph_visualizer": KnowledgeGraphVisualizer()
    }
    
    return components

def process_single_document(pdf_path: Path, document_output_path: Path, 
                          components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    MODIFIED: Process a single document with support for document-aware processing.
    """
    
    document_id = pdf_path.stem
    print(f"\nProcessing document: '{pdf_path.name}' (ID: {document_id})")
    print(f"Processing mode: {config.get('processing_mode', 'page_based')}")

    # Parse PDF
    pdf_parser = PDFParser(pdf_path)

    # Classify and get summary
    classification_result = components["document_classifier"].classify_document(pdf_parser)
    identified_doc_type = classification_result["identified_doc_type"]
    document_summary = classification_result["document_summary"]
    
    print(f"Document '{document_id}' classified as: '{identified_doc_type}'.")

    # Prepare context
    document_context_info = components["document_context_preparer"].prepare_context(
        identified_doc_type=identified_doc_type,
        summary=document_summary
    )

    # Initialize page processor
    page_llm_processor = PageLLMProcessor(
        llm_client=components["main_llm_client"],
        ontology=components["ontology"], 
        extraction_mode=config["extraction_mode"], 
        use_ontology=config.get("use_ontology", True),
    )
    
    # Initialize KG constructor with processing mode
    kg_constructor = KGConstructorSingleDoc(
        pdf_parser=pdf_parser,
        document_context_info=document_context_info,
        page_llm_processor=page_llm_processor,
        page_level_merger=components["page_level_merger"],
        graph_visualizer=components["graph_visualizer"],
        config={
            "dump_page_kgs": config["dump_page_kgs"],
            # NEW: Add semantic chunking configuration
            "chunk_size": config.get("chunk_size", 4000),
            "min_chunk_size": config.get("min_chunk_size", 500),
            "chunk_overlap": config.get("chunk_overlap", 200),
            "respect_sentence_boundaries": config.get("respect_sentence_boundaries", True),
            "detect_topic_shifts": config.get("detect_topic_shifts", True)
        },
        document_id=document_id,
        document_output_path=document_output_path,
        processing_mode=config.get("processing_mode", "page_based")  # NEW PARAMETER
    )

    # Execute KG construction
    document_kg = kg_constructor.construct_kg(
        construction_mode=config["construction_mode"], 
        max_workers=config["max_workers"]
    )
    
    # Return KG with document ID
    return {
        "document_id": document_id,
        "entities": document_kg.get("entities", []),
        "relationships": document_kg.get("relationships", [])
    }


def save_and_visualize_kg(kg_data: Dict[str, Any], output_path: Path, 
                         filename_base: str, visualizer: Any, 
                         is_multi_doc: bool = False) -> None:
    """Save KG to JSON and create visualizations. Centralized to avoid duplication."""
    
    # Save JSON
    json_file = output_path / f"{filename_base}.json"
    with open(json_file, "w", encoding='utf-8') as f:
        json.dump(kg_data, f, indent=2, ensure_ascii=False)
    print(f"KG saved to {json_file}")

    # Create visualizations if entities exist
    if kg_data.get("entities"):
        html_file = str(output_path / f"{filename_base}.html")
        
        try:
            visualizer.export_interactive_html(kg_data, html_file)
            print(f"Interactive visualization saved to {html_file}")
            
            # Create comparison view for multi-document KGs
            if is_multi_doc:
                comparison_file = str(output_path / f"{filename_base}_comparison.html")
                visualizer.export_multi_document_comparison(kg_data, comparison_file)
                print(f"Multi-document comparison saved to {comparison_file}")
                
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")
    else:
        print(f"No entities found in KG. Skipping visualization for {filename_base}.")

def run_diagnostic_analysis(extracted_page_kgs: List[Dict[str, Any]], 
                          final_document_kg: Dict[str, Any]) -> None:
    """Run diagnostic analysis if available."""
    if DIAGNOSTIC_AVAILABLE:
        try:
            print("\n" + "="*60)
            print("RUNNING MERGER DIAGNOSTIC ANALYSIS")
            print("="*60)
            diagnose_merger_issues(extracted_page_kgs, final_document_kg)
        except Exception as e:
            print(f"Diagnostic analysis failed: {e}")
    else:
        print("Diagnostic analysis not available. Install merger_diagnostic_tool for detailed analysis.")

def apply_transformation(kg_data: Dict[str, Any], project_name: str, 
                        output_path: Path, visualizer: Any) -> None:
    """Apply transformation if available."""
    if not TRANSFORM_AVAILABLE:
        print("Transformation requested, but 'transform_kg' function not available.")
        return
    
    print("\nPerforming transformation on the final project KG...")
    try:
        # Note: transform_kg requires additional parameters that are not available in this context
        # Commenting out for now to avoid missing parameter errors
        print("Transform functionality not fully configured - skipping transformation.")
        # transform_request_id = str(uuid.uuid4())
        # transformed_kg = transform_kg(kg_data, transform_request_id, project_name)
        
        # Save and visualize transformed KG
        # save_and_visualize_kg(
        #     transformed_kg, 
        #     output_path, 
        #     "transformed_project_kg", 
        #     visualizer, 
        #     is_multi_doc=True
        # )
        
    except Exception as e:
        print(f"Transformation failed: {e}")

def run_project_pipeline(
    input_folder_path_str: str,
    project_name: str,
    llm_provider: str,
    main_model_name: str,
    main_azure_model_env_suffix: Optional[str],
    classification_model_name: str,
    classification_azure_model_env_suffix: Optional[str],
    summary_model_name: str,
    summary_azure_model_env_suffix: Optional[str],
    construction_mode: str,
    extraction_mode: str,
    predefined_categories: List[str],
    max_workers: int,
    dump_page_kgs: bool,
    transform_final_kg: bool,
    use_ontology: bool,
    run_diagnostics: bool = False,
    # NEW PARAMETERS for semantic chunking
    processing_mode: str = "page_based",
    chunk_size: int = 4000,
    min_chunk_size: int = 500,
    chunk_overlap: int = 200,
    respect_sentence_boundaries: bool = True,
    detect_topic_shifts: bool = True):
    """
    MODIFIED: Orchestrates the entire pipeline with support for document-aware processing.
    """
    start_time = time.time()
    print(f"Starting PEKG extraction pipeline for project: '{project_name}'")
    print(f"Input folder: {input_folder_path_str}")
    print(f"Processing mode: {processing_mode}")

    # Setup project structure
    project_output_path = setup_project_structure(project_name)

    # Prepare model configurations
    model_configs = {
        "main_model_name": main_model_name,
        "main_azure_model_env_suffix": main_azure_model_env_suffix,
        "classification_model_name": classification_model_name,
        "classification_azure_model_env_suffix": classification_azure_model_env_suffix,
        "summary_model_name": summary_model_name,
        "summary_azure_model_env_suffix": summary_azure_model_env_suffix,
        "predefined_categories": predefined_categories,
        "use_ontology": use_ontology  # Default to using ontology in context preparation
    }

    # Initialize components
    ontology_path = Path(__file__).parent / "ontology_management" / "pekg_ontology_teasers.yaml"
    components = initialize_components(llm_provider, model_configs, ontology_path)

    # Processing configuration
    processing_config = {
        "construction_mode": construction_mode,
        "extraction_mode": extraction_mode,
        "max_workers": max_workers,
        "dump_page_kgs": dump_page_kgs,
        "processing_mode": processing_mode,
        "chunk_size": chunk_size,
        "min_chunk_size": min_chunk_size,
        "chunk_overlap": chunk_overlap,
        "respect_sentence_boundaries": respect_sentence_boundaries,
        "detect_topic_shifts": detect_topic_shifts, 
        "use_ontology": use_ontology
    }

    # Log configuration for document-aware processing
    if processing_mode == "document_aware":
        print(f"Semantic chunking configuration:")
        print(f"  - Max chunk size: {chunk_size} characters")
        print(f"  - Min chunk size: {min_chunk_size} characters")
        print(f"  - Chunk overlap: {chunk_overlap} characters")
        print(f"  - Respect sentence boundaries: {respect_sentence_boundaries}")
        print(f"  - Detect topic shifts: {detect_topic_shifts}")

    # Discover PDF files
    pdf_files = discover_pdf_files(input_folder_path_str)
    if not pdf_files:
        print(f"No PDF files found in {input_folder_path_str}. Exiting.")
        return

    print(f"Found {len(pdf_files)} PDF files to process")

    # Process each document
    all_document_kgs_with_ids: List[Dict[str, Any]] = []
    
    for pdf_path in pdf_files:
        document_id = pdf_path.stem
        document_output_path = project_output_path / document_id
        document_output_path.mkdir(parents=True, exist_ok=True)
        
        # Process single document
        document_kg_with_id = process_single_document(
            pdf_path, document_output_path, components, processing_config
        )
        all_document_kgs_with_ids.append(document_kg_with_id)

    # Rest of the function remains the same...
    # (Inter-document merging, visualization, transformation, etc.)
    
    # Merge all document KGs (Project-Level)
    print(f"\nMerging {len(all_document_kgs_with_ids)} document KGs into project-level KG...")
    final_project_kg = components["inter_document_merger"].merge_project_kgs(all_document_kgs_with_ids)

    # Save and visualize final project KG
    save_and_visualize_kg(
        final_project_kg, 
        project_output_path, 
        "full_project_kg", 
        components["graph_visualizer"], 
        is_multi_doc=True
    )

    # Apply transformation if requested
    if transform_final_kg:
        apply_transformation(
            final_project_kg, 
            project_name, 
            project_output_path, 
            components["graph_visualizer"]
        )

    # Final statistics and timing
    end_time = time.time()
    total_entities = len(final_project_kg.get("entities", []))
    total_relationships = len(final_project_kg.get("relationships", []))
    
    print(f"\n{'='*60}")
    print(f"PEKG EXTRACTION PIPELINE COMPLETED")
    print(f"{'='*60}")
    print(f"Project: {project_name}")
    print(f"Processing mode: {processing_mode}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Documents processed: {len(pdf_files)}")
    print(f"Final entities: {total_entities}")
    print(f"Final relationships: {total_relationships}")
    print(f"Output directory: {project_output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate PEKG Knowledge Graph extraction from PDF documents.")

    parser.add_argument("input_folder_path", type=str, help="Path to the folder containing PDF documents.")
    parser.add_argument("project_name", type=str, help="Name of the project. Used for output organization and ontology loading.")
    parser.add_argument("llm_provider", type=str, choices=["azure", "vertexai"], help="LLM provider to use (azure or vertexai).")
    
    parser.add_argument("--main_model_name", type=str, required=True, 
                        help="Name of the main LLM model for page-level extraction (e.g., 'gemini-1.5-pro-preview-0514' or 'gpt-4o').")
    parser.add_argument("--main_azure_model_env_suffix", type=str, 
                        help="Suffix for Azure LLM environment variables (e.g., 'GPT4O_AZURE' if env vars are AZURE_OPENAI_ENDPOINT_GPT4O_AZURE). Required for Azure.")

    parser.add_argument("--classification_model_name", type=str,
                        help="Name of the LLM model for document classification.")
    parser.add_argument("--classification_azure_model_env_suffix", type=str, 
                        help="Suffix for Azure LLM env vars for classification model. Required if llm_provider is Azure for classification.")

    parser.add_argument("--summary_model_name", type=str,
                        help="Name of the LLM model for document summarization.")
    parser.add_argument("--summary_azure_model_env_suffix", type=str, 
                        help="Suffix for Azure LLM env vars for summary model. Required if llm_provider is Azure for summarization.")
    
    parser.add_argument("--construction_mode", type=str, choices=["iterative", "parallel"], default="parallel",
                        help="KG construction mode: 'iterative' (sequential page processing) or 'parallel' (concurrent page processing).")
    parser.add_argument("--extraction_mode", type=str, choices=["text", "multimodal"], default="text",
                        help="Extraction modality: 'text' or 'multimodal'.")
    parser.add_argument("--predefined_categories", nargs='+', default=["financial_teaser", "legal_contract", "press_release", "annual_report"],
                        help="List of predefined document categories for classification. Default: financial_teaser legal_contract press_release annual_report")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of parallel workers (default: 4).")
    parser.add_argument("--dump_page_kgs", action="store_true", help="Save intermediate KGs for each page of each document.")
    parser.add_argument("--transform_final_kg", action="store_true", help="Perform transformation on the final merged project KG.")
    parser.add_argument("--run_diagnostics", action="store_true", help="Run diagnostic analysis on merging process.")

    parser.add_argument("--processing_mode", type=str, choices=["page_based", "document_aware"], 
                       default="page_based", 
                       help="Processing approach: 'page_based' (existing) or 'document_aware' (new semantic chunking)")

    parser.add_argument('--no_ontology', action='store_true', 
                        help="Disable ontology-driven extraction. The LLM will identify entities and relations freely.")
    
    parser.add_argument("--chunk_size", type=int, default=4000,
                       help="Maximum chunk size for document-aware processing (default: 4000)")
    
    parser.add_argument("--min_chunk_size", type=int, default=500,
                       help="Minimum chunk size for document-aware processing (default: 500)")
    
    parser.add_argument("--chunk_overlap", type=int, default=200,
                       help="Overlap size between chunks for document-aware processing (default: 200)")
    
    parser.add_argument("--no_sentence_boundaries", action="store_true",
                       help="Allow chunk breaks in middle of sentences (not recommended)")
    
    parser.add_argument("--no_topic_detection", action="store_true",
                       help="Disable automatic topic shift detection in chunks")


    if len(sys.argv) <= 7:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()

    if args.llm_provider == 'azure':
        if not args.main_azure_model_env_suffix:
            parser.error("--main_azure_model_env_suffix is required when llm_provider is 'azure' for the main model.")
        if not args.classification_azure_model_env_suffix:
            parser.error("--classification_azure_model_env_suffix is required when llm_provider is 'azure' for the classification model.")
        if not args.summary_azure_model_env_suffix:
            parser.error("--summary_azure_model_env_suffix is required when llm_provider is 'azure' for the summarization model.")

    use_ontology_flag = not args.no_ontology
    if use_ontology_flag:
        print("Ontology-driven extraction is enabled. The LLM will use the ontology to guide entity and relation extraction.")
    else:
        print("Ontology-driven extraction is disabled. The LLM will identify entities and relations freely.")

    run_project_pipeline(
        input_folder_path_str=args.input_folder_path,
        project_name=args.project_name,
        llm_provider=args.llm_provider.lower(),
        main_model_name=args.main_model_name,
        main_azure_model_env_suffix=args.main_azure_model_env_suffix,
        classification_model_name=args.classification_model_name,
        classification_azure_model_env_suffix=args.classification_azure_model_env_suffix,
        summary_model_name=args.summary_model_name,
        summary_azure_model_env_suffix=args.summary_azure_model_env_suffix,
        construction_mode=args.construction_mode.lower(),
        extraction_mode=args.extraction_mode.lower(),
        predefined_categories=args.predefined_categories,
        max_workers=args.max_workers,
        dump_page_kgs=args.dump_page_kgs,
        transform_final_kg=args.transform_final_kg,
        run_diagnostics=args.run_diagnostics,
        processing_mode=args.processing_mode,
        chunk_size=args.chunk_size,
        min_chunk_size=args.min_chunk_size,
        chunk_overlap=args.chunk_overlap,
        respect_sentence_boundaries=not args.no_sentence_boundaries,
        detect_topic_shifts=not args.no_topic_detection,
        use_ontology=use_ontology_flag
    )