import json
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, Optional, List
import threading

# Assuming components are structured as previously discussed
# Adjust import paths based on your final project structure
from _1_document_ingestion.pdf_parser import PDFParser
from _2_context_understanding.base_context_identifier import BaseContextIdentifier
from _3_knowledge_extraction.page_llm_processor import PageLLMProcessor

from _4_knowledge_graph_operations.page_level_merger import PageLevelMerger
from _4_knowledge_graph_operations.common_kg_utils import normalize_entity_ids, clean_knowledge_graph

from visualization_tools.KG_visualizer import KnowledgeGraphVisualizer # Assuming this path

class KGConstructorSingleDoc:
    """
    Constructs a knowledge graph from a single document by orchestrating
    context identification, page processing, and page-level KG merging.
    """

    def __init__(self,
                 pdf_parser: PDFParser,
                 context_identifier: BaseContextIdentifier,
                 page_llm_processor: PageLLMProcessor,
                 page_level_merger: PageLevelMerger,
                 graph_visualizer: KnowledgeGraphVisualizer,
                 config: Dict[str, Any]):
        """
        Initializes the KGConstructorSingleDoc.

        Args:
            pdf_parser (PDFParser): An instance of the PDF parser with the document loaded.
            context_identifier (BaseContextIdentifier): An instance of a context identifier.
            page_llm_processor (PageLLMProcessor): An instance of the page LLM processor.
            page_level_merger (PageLevelMerger): An instance of the page-level KG merger.
            graph_visualizer (KnowledgeGraphVisualizer): An instance for visualizing KGs.
            config (Dict[str, Any]): Configuration dictionary, including:
                - "project_name" (str): Name of the project.
                - "document_id" (str): A unique ID or name for the current document.
                - "extraction_mode" (str): "text" or "multimodal".
                - "construction_mode" (str): "iterative", "parallel", or "onego".
                - "output_base_path" (str): Base path for saving outputs.
                - "num_workers" (int, optional): Number of workers for parallel mode. Defaults to 16.
                - "dump_intermediate_page_kgs" (bool, optional): Whether to save KGs for each page. Defaults to False.
                - "doc_type_hint" (str, optional): Hint for document type for context identification.
        """
        self.pdf_parser = pdf_parser
        self.context_identifier = context_identifier
        self.page_llm_processor = page_llm_processor
        self.page_level_merger = page_level_merger
        self.graph_visualizer = graph_visualizer
        self.config = config

        self.project_name = config.get("project_name", "default_project")
        self.document_id = config.get("document_id", "unknown_document")
        self.extraction_mode = config.get("extraction_mode", "text")
        self.construction_mode = config.get("construction_mode", "iterative")
        self.output_base_path = Path(config.get("output_base_path", "outputs"))
        self.num_workers = config.get("num_workers", 16)
        self.dump_page_kgs = config.get("dump_intermediate_page_kgs", False)
        self.doc_type_hint = config.get("doc_type_hint")

        # Prepare output paths
        self.document_output_path = self.output_base_path / self.project_name / f"{self.document_id}_kg"
        self.document_output_path.mkdir(parents=True, exist_ok=True)
        if self.dump_page_kgs:
            self.page_kG_output_path = self.document_output_path / "page_kGs"
            self.page_kG_output_path.mkdir(parents=True, exist_ok=True)

        self.document_context_info: Dict[str, Any] = {}


    def _save_page_graph(self, graph: Dict[str, Any], page_num_0_indexed: int, llm_provider_name: str) -> None:
        """Saves and visualizes an individual page's knowledge graph."""
        if not self.dump_page_kgs or not graph or (not graph.get("entities") and not graph.get("relationships")):
            return

        page_num_display = page_num_0_indexed + 1
        graph_to_save = normalize_entity_ids(clean_knowledge_graph(graph.copy()))

        # Define a base filename for the page KG
        # Model name might be part of page_llm_processor.llm_client if needed
        model_name_sanitized = self.page_llm_processor.llm_client.model_name.replace('/', '_').replace(':', '_')
        provider_name_sanitized = llm_provider_name.lower().replace(" ", "_")

        base_filename = (f"{self.extraction_mode}_kg_pg{page_num_display}_{model_name_sanitized}_"
                         f"{provider_name_sanitized}_{self.construction_mode}")

        json_file = self.page_kG_output_path / f"{base_filename}.json"
        html_file = str(self.page_kG_output_path / f"{base_filename}.html")

        try:
            with open(json_file, "w") as f:
                json.dump(graph_to_save, f, indent=2)
        except Exception as e:
            print(f"Error saving JSON for page {page_num_display}: {e}")

        try:
            if graph_to_save.get("entities"): # Basic check for visualizability
                self.graph_visualizer.export_interactive_html(graph_to_save, html_file)
        except Exception as e:
            print(f"Could not save HTML visualization for page {page_num_display}: {type(e).__name__} - {e}")


    def _parallel_page_worker(self, page_num_0_indexed: int, page_text: str, page_image_base64: Optional[str]) -> Dict[str, Any]:
        """
        Worker function for processing a single page in parallel.
        Processes a page using PageLLMProcessor.
        """
        page_data = {"text": page_text, "image_base64": page_image_base64}
        page_num_display = str(page_num_0_indexed + 1)

        print(f"Worker processing page {page_num_display} for doc '{self.document_id}'...")
        page_kg = self.page_llm_processor.process_page(
            page_data=page_data,
            document_context_info=self.document_context_info,
            extraction_mode=self.extraction_mode,
            construction_mode="parallel", # In parallel, each page is processed independently initially
            previous_graph_context=None,
            page_num_for_logging=page_num_display
        )
        
        llm_provider_name = self.page_llm_processor.llm_client.__class__.__name__
        if self.dump_page_kgs:
            self._save_page_graph(page_kg, page_num_0_indexed, llm_provider_name)
        
        print(f"Worker completed page {page_num_display}. Entities: {len(page_kg.get('entities',[]))}, Relationships: {len(page_kg.get('relationships',[]))}")
        return {"page_num": page_num_0_indexed, "graph": page_kg}


    def _build_kg_iterative(self) -> Dict[str, List[Any]]:
        """Builds KG iteratively, page by page."""
        merged_document_kg: Dict[str, List[Any]] = {"entities": [], "relationships": []}
        num_pages = len(self.pdf_parser.doc)

        for i in range(num_pages):
            page_num_display = str(i + 1)
            print(f"Processing page {page_num_display}/{num_pages} for doc '{self.document_id}' (iterative, {self.extraction_mode})...")
            
            page_content = self.pdf_parser.extract_page_from_pdf(i) # Method to get text & image_base64
            
            if not page_content.get("text","").strip() and (self.extraction_mode == "text" or not page_content.get("image_base64")):
                print(f"Skipping page {page_num_display} due to no relevant content for extraction mode.")
                continue

            page_kg = self.page_llm_processor.process_page(
                page_data=page_content,
                document_context_info=self.document_context_info,
                extraction_mode=self.extraction_mode,
                construction_mode="iterative",
                previous_graph_context=merged_document_kg, # Pass the current merged graph
                page_num_for_logging=page_num_display
            )
            
            llm_provider_name = self.page_llm_processor.llm_client.__class__.__name__
            if self.dump_page_kgs:
                self._save_page_graph(page_kg, i, llm_provider_name)

            merged_document_kg = self.page_level_merger.merge_incrementally(merged_document_kg, page_kg)
            
            # Optional: periodic cleaning
            if (i + 1) % 5 == 0:
                print(f"Performing interim cleaning of document KG after page {page_num_display}...")
                merged_document_kg = clean_knowledge_graph(merged_document_kg)
            
            print(f"  Completed page {page_num_display}. Doc KG: {len(merged_document_kg.get('entities',[]))} entities, {len(merged_document_kg.get('relationships',[]))} relationships")
            
        return merged_document_kg

    def _build_kg_parallel(self) -> Dict[str, List[Any]]:
        """Builds KG by processing pages in parallel and then merging."""
        num_pages = len(self.pdf_parser.doc)
        print(f"Starting parallel KG construction for doc '{self.document_id}' ({num_pages} pages, {self.extraction_mode}) with {self.num_workers} workers.")

        page_inputs = []
        for i in range(num_pages):
            page_content = self.pdf_parser.extract_page_from_pdf(i) # Method to get text & image_base64
            page_inputs.append({
                "page_num_0_indexed": i,
                "page_text": page_content.get("text", ""),
                "page_image_base64": page_content.get("image_base64")
            })

        page_results: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_page_num = {
                executor.submit(
                    self._parallel_page_worker,
                    pi["page_num_0_indexed"],
                    pi["page_text"],
                    pi["page_image_base64"]
                ): pi["page_num_0_indexed"] for pi in page_inputs
            }
            for future in concurrent.futures.as_completed(future_to_page_num):
                page_num = future_to_page_num[future]
                try:
                    result = future.result()
                    page_results.append(result)
                except Exception as exc:
                    print(f"Page {page_num + 1} generated an exception: {exc}")
                    page_results.append({"page_num": page_num, "graph": {"entities": [], "relationships": []}}) # Add empty graph for failed pages

        # Sort results by page number before merging
        page_results.sort(key=lambda r: r["page_num"])
        
        page_kgs_to_merge = [res["graph"] for res in page_results]
        
        print("Merging all page KGs from parallel processing...")
        merged_document_kg = self.page_level_merger.merge_all_page_kgs(page_kgs_to_merge)
        return merged_document_kg

    def _build_kg_onego(self) -> Dict[str, List[Any]]:
        """Builds KG by processing the entire document content at once (text-only or multimodal pages merged then onego)."""
        print(f"Starting one-go KG construction for doc '{self.document_id}' ({self.extraction_mode}).")
        if self.extraction_mode == "text":
            full_text = self.pdf_parser.extract_full_text() # PDFParser needs this method
            if not full_text.strip():
                print("Warning: No text extracted from PDF for 'onego' text mode.")
                return {"entities": [], "relationships": []}
            
            # Use page_num_for_logging="all_pages_onego" or similar
            document_kg = self.page_llm_processor.process_page(
                page_data={"text": full_text, "image_base64": None},
                document_context_info=self.document_context_info,
                extraction_mode="text",
                construction_mode="onego",
                page_num_for_logging="all_pages_onego"
            )
            llm_provider_name = self.page_llm_processor.llm_client.__class__.__name__
            if self.dump_page_kgs: # Save the single resulting graph as if it's a page KG
                 self._save_page_graph(document_kg, 0, llm_provider_name) # Save as page 0
            return document_kg
        
        elif self.extraction_mode == "multimodal":
            # "One-go" for multimodal usually means processing each page multimodally
            # and then merging them without iterative context. This is similar to parallel.
            # If it means sending ALL pages+images to one LLM call, that's usually not feasible.
            # Re-interpreting "onego" for multimodal as independent page processing then merge:
            print("One-go multimodal: Processing all pages independently then merging.")
            num_pages = len(self.pdf_parser.doc)
            page_kgs_list: List[Dict[str, Any]] = []
            for i in range(num_pages):
                page_num_display = str(i + 1)
                page_content = self.pdf_parser.extract_page_from_pdf(i)
                
                if not page_content.get("text","").strip() and not page_content.get("image_base64"):
                    print(f"Skipping page {page_num_display} in one-go multimodal due to no content.")
                    continue

                print(f"Processing page {page_num_display}/{num_pages} (multimodal, one-go independent)...")
                page_kg = self.page_llm_processor.process_page(
                    page_data=page_content,
                    document_context_info=self.document_context_info,
                    extraction_mode="multimodal",
                    construction_mode="onego", # or "parallel" to signify no prev_graph_context
                    previous_graph_context=None,
                    page_num_for_logging=page_num_display
                )
                llm_provider_name = self.page_llm_processor.llm_client.__class__.__name__
                if self.dump_page_kgs:
                    self._save_page_graph(page_kg, i, llm_provider_name)
                page_kgs_list.append(page_kg)
            
            return self.page_level_merger.merge_all_page_kgs(page_kgs_list)
        
        return {"entities": [], "relationships": []}

    def build_kg(self) -> Dict[str, List[Any]]:
        """
        Main method to build the knowledge graph for the single document.
        """
        print(f"Starting KG construction for document: '{self.document_id}' (Project: '{self.project_name}')")
        print(f"Extraction Mode: {self.extraction_mode}, Construction Mode: {self.construction_mode}")

        # 1. Identify context for the document
        print("Identifying document context...")
        try:
            self.document_context_info = self.context_identifier.identify_context(
                document_path=str(self.pdf_parser.pdf_path), # Assuming pdf_parser has pdf_path attribute
                doc_type_hint=self.doc_type_hint
            )
            print(f"Document context identified: {self.document_context_info.get('identified_document_type', 'N/A')}")
            # You might want to log more details from document_context_info here
        except Exception as e:
            print(f"Error during document context identification for '{self.document_id}': {e}")
            print("Proceeding with default/empty context.")
            self.document_context_info = {"identified_document_type": self.doc_type_hint or "unknown"}


        # 2. Build KG based on construction mode
        final_document_kg: Dict[str, List[Any]]

        if self.construction_mode == "iterative":
            final_document_kg = self._build_kg_iterative()
        elif self.construction_mode == "parallel":
            final_document_kg = self._build_kg_parallel()
        elif self.construction_mode == "onego":
            final_document_kg = self._build_kg_onego()
        else:
            raise ValueError(f"Unsupported construction_mode: {self.construction_mode}")

        # 3. Final cleaning and normalization
        print("Performing final graph cleanup and normalization for the document KG...")
        final_document_kg = clean_knowledge_graph(final_document_kg)
        final_document_kg = normalize_entity_ids(final_document_kg) # Renumber IDs for this document

        # Note: The consolidation step from the original KG_builder.py that called an LLM
        # to consolidate the graph is omitted here for simplicity. It could be added back
        # as a post-processing step if desired, perhaps as part of the PageLevelMerger or here.

        print(f"KG construction for document '{self.document_id}' completed.")
        print(f"Final document KG: {len(final_document_kg.get('entities',[]))} entities, {len(final_document_kg.get('relationships',[]))} relationships.")
        
        # Save the final KG for this document
        doc_kg_json_file = self.document_output_path / "full_document_kg.json"
        doc_kg_html_file = str(self.document_output_path / "full_document_kg.html")
        try:
            with open(doc_kg_json_file, "w") as f:
                json.dump(final_document_kg, f, indent=2)
            print(f"Document KG saved to {doc_kg_json_file}")
            if final_document_kg.get("entities"):
                self.graph_visualizer.export_interactive_html(final_document_kg, doc_kg_html_file)
                print(f"Document KG visualization saved to {doc_kg_html_file}")
        except Exception as e:
            print(f"Error saving final document KG for '{self.document_id}': {e}")
            
        return final_document_kg