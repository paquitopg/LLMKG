# File: llm_kg_extraction/_3_knowledge_extraction/kg_constructor_single_doc.py

import json
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, Optional, List
import threading

# Assuming components are structured as previously discussed
# Adjust import paths based on your final project structure
from _1_document_ingestion.pdf_parser import PDFParser
from _3_knowledge_extraction.page_llm_processor import PageLLMProcessor

from _4_knowledge_graph_operations.page_level_merger import PageLevelMerger
from _4_knowledge_graph_operations.common_kg_utils import normalize_entity_ids, clean_knowledge_graph

from visualization_tools.KG_visualizer import KnowledgeGraphVisualizer # Assuming this path
from REPOS.llm_kg_extraction.llm_kg_extraction.merger_diagnostic_tool import diagnose_merger_issues

class KGConstructorSingleDoc:
    """
    Constructs a knowledge graph from a single document by orchestrating
    page processing and page-level KG merging.
    """

    def __init__(self,
                 pdf_parser: PDFParser,
                 document_context_info: Dict[str, Any], 
                 page_llm_processor: PageLLMProcessor,
                 page_level_merger: PageLevelMerger,
                 graph_visualizer: KnowledgeGraphVisualizer,
                 config: Dict[str, Any],
                 document_id: str,
                 document_output_path: Path
                 ):
        """
        Initializes the KGConstructorSingleDoc.

        Args:
            pdf_parser (PDFParser): An instance of the PDF parser with the document loaded.
            document_context_info (Dict[str, Any]): The pre-prepared document-level context
                                                    containing summary, type, and ontology.
            page_llm_processor (PageLLMProcessor): An instance of the page-level LLM processor.
            page_level_merger (PageLevelMerger): An instance of the page-level KG merger.
            graph_visualizer (KnowledgeGraphVisualizer): An instance of the KG visualizer.
            config (Dict[str, Any]): Configuration dictionary (e.g., {"dump_page_kgs": True}).
            document_id (str): Unique identifier for the current document.
            document_output_path (Path): Path to the output directory for this specific document.
        """
        self.pdf_parser = pdf_parser
        self.document_context_info = document_context_info
        self.page_llm_processor = page_llm_processor
        self.page_level_merger = page_level_merger
        self.graph_visualizer = graph_visualizer
        self.config = config
        self.document_id = document_id
        self.document_output_path = document_output_path
        self.lock = threading.Lock()

    def _process_single_page(self, page_data: Dict[str, Any], 
                           previous_graph_context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Processes a single page using the PageLLMProcessor.
        This method replaces the separate text/multimodal methods since PageLLMProcessor now handles both.
        """
        page_number = page_data.get('page_number', 'N/A')
        
        print(f"  Processing page {page_number}...")
        try:
            # Use the unified process_page method
            page_kg = self.page_llm_processor.process_page(
                page_data=page_data,
                document_context_info=self.document_context_info,
                construction_mode="iterative",  # or could be passed as parameter
                previous_graph_context=previous_graph_context
            )
            
            if page_kg and (page_kg.get("entities") or page_kg.get("relationships")):
                print(f"  Extracted KG for page {page_number}: {len(page_kg.get('entities', []))} entities, {len(page_kg.get('relationships', []))} relationships.")
                # Dump html visualization for this page if configured
                if self.config.get("dump_page_kgs", False):
                    dump_dir = self.document_output_path / "page_kgs"
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    html_dump_file = dump_dir / f"page_{page_number}_kg.html"
                    self.graph_visualizer.export_interactive_html(page_kg, str(html_dump_file))
                    print(f"  Page KG visualization saved to {html_dump_file}")
                return page_kg
            else:
                print(f"  No entities or relationships extracted for page {page_number}.")
                return None
                
        except Exception as e:
            print(f"  Error processing page {page_number}: {e}")
            return None

    def construct_kg(self, construction_mode: str, max_workers: int) -> Dict[str, Any]:
        """
        Constructs the knowledge graph for the entire document.

        Args:
            construction_mode (str): 'iterative' or 'parallel'.
            max_workers (int): Maximum number of parallel workers if 'parallel' mode is used.

        Returns:
            Dict[str, Any]: The consolidated knowledge graph for the document.
        """
        print(f"Starting KG construction for document '{self.document_id}' in '{construction_mode}' mode.")

        extracted_page_kgs: List[Dict[str, Any]] = []

        # Get pages data based on extraction mode
        if self.page_llm_processor.extraction_mode == "text":
            pages_data = self.pdf_parser.extract_all_pages_text()
        elif self.page_llm_processor.extraction_mode == "multimodal":
            pages_data = self.pdf_parser.extract_all_pages_multimodal()
        else:
            raise ValueError(f"Unsupported extraction_mode: {self.page_llm_processor.extraction_mode}")

        if not pages_data:
            print(f"No pages found for document '{self.document_id}'. Returning empty KG.")
            return {"entities": [], "relationships": []}

        if construction_mode == "iterative":
            # Process pages sequentially, building context progressively
            accumulated_kg = {"entities": [], "relationships": []}
            
            for page_data in pages_data:
                # For iterative mode, pass the accumulated KG as context for the next page
                page_kg = self._process_single_page(page_data, accumulated_kg)
                
                if page_kg:
                    extracted_page_kgs.append(page_kg)
                    # Merge this page's KG into the accumulated context for the next page
                    accumulated_kg = self.page_level_merger.merge_incrementally(accumulated_kg, page_kg)
                    
        elif construction_mode == "parallel":
            # Process pages in parallel without context sharing
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all pages for processing
                futures = {
                    executor.submit(self._process_single_page, page_data): page_data['page_number'] 
                    for page_data in pages_data
                }
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    page_num = futures[future]
                    try:
                        page_kg = future.result()
                        if page_kg:
                            extracted_page_kgs.append(page_kg)

                    except Exception as e:
                        print(f"  Error processing page {page_num} in parallel: {e}")
        else:
            raise ValueError(f"Unsupported construction_mode: {construction_mode}")

        # Dump individual page KGs if configured
        if self.config.get("dump_page_kgs", False):
            self._dump_page_kgs(extracted_page_kgs)

        # Merge all page KGs into final document KG
        print(f"Merging {len(extracted_page_kgs)} page KGs for document '{self.document_id}'...")
        
        if construction_mode == "iterative":
            # For iterative mode, we already have the final merged KG in accumulated_kg
            final_document_kg = accumulated_kg
        else:
            # For parallel mode, merge all page KGs
            final_document_kg = self.page_level_merger.merge_all_page_kgs(extracted_page_kgs)
            print("  Diagnosing merger issues...")
            #diagnose_merger_issues(extracted_page_kgs, final_document_kg)

        # Apply common KG utilities (cleaning, re-numbering)
        print("  Cleaning and normalizing document KG...")
        final_document_kg = clean_knowledge_graph(final_document_kg)
        final_document_kg = normalize_entity_ids(final_document_kg)

        print(f"KG construction for document '{self.document_id}' completed.")
        print(f"Final document KG: {len(final_document_kg.get('entities',[]))} entities, {len(final_document_kg.get('relationships',[]))} relationships.")
        
        # Save the final KG for this document
        self._save_document_kg(final_document_kg)
        
        return final_document_kg

    def _dump_page_kgs(self, page_kgs: List[Dict[str, Any]]) -> None:
        """Dump individual page KGs to files."""
        dump_dir = self.document_output_path / "page_kgs"
        dump_dir.mkdir(parents=True, exist_ok=True)
        
        for page_kg in page_kgs:
            page_num = page_kg.get("page_number", "unknown")
            dump_file = dump_dir / f"page_{page_num}_kg.json"
            with open(dump_file, "w") as f:
                json.dump(page_kg, f, indent=2)
            html_dump_file = dump_dir / f"page_{page_num}_kg.html"
            self.graph_visualizer.export_interactive_html(page_kg, str(html_dump_file))

    def _save_document_kg(self, final_document_kg: Dict[str, Any]) -> None:
        """Save the final document KG and create visualization."""
        doc_kg_json_file = self.document_output_path / "full_document_kg.json"
        doc_kg_html_file = str(self.document_output_path / "full_document_kg.html")
        
        try:
            with open(doc_kg_json_file, "w") as f:
                json.dump(final_document_kg, f, indent=2)
            print(f"Document KG saved to {doc_kg_json_file}")
            
            if final_document_kg.get("entities"):
                self.graph_visualizer.export_interactive_html(final_document_kg, doc_kg_html_file)
                print(f"Document KG visualization saved to {doc_kg_html_file}")
            else:
                print(f"No entities found in document KG for '{self.document_id}'. Skipping visualization.")
        except Exception as e:
            print(f"Error saving or visualizing document KG for '{self.document_id}': {e}")