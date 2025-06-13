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
#from REPOS.llm_kg_extraction.llm_kg_extraction.merger_diagnostic_tool import diagnose_merger_issues



from .document_aware_extraction import SemanticChunker, SemanticChunk
from typing import Union

class KGConstructorSingleDoc:
    """
    Modified version to support both page-based and document-aware processing.
    """
    
    def __init__(self,
                 pdf_parser: PDFParser,
                 document_context_info: Dict[str, Any], 
                 page_llm_processor: PageLLMProcessor,
                 page_level_merger: PageLevelMerger,
                 graph_visualizer: KnowledgeGraphVisualizer,
                 config: Dict[str, Any],
                 document_id: str,
                 document_output_path: Path,
                 processing_mode: str = "page_based"):  # NEW PARAMETER
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
        # Existing initialization
        self.pdf_parser = pdf_parser
        self.document_context_info = document_context_info
        self.page_llm_processor = page_llm_processor
        self.page_level_merger = page_level_merger
        self.graph_visualizer = graph_visualizer
        self.config = config
        self.document_id = document_id
        self.document_output_path = document_output_path
        self.lock = threading.Lock()
        
        # NEW: Processing mode and semantic chunking
        self.processing_mode = processing_mode
        
        if processing_mode == "document_aware":
            # Initialize semantic chunker
            self.semantic_chunker = SemanticChunker(
                max_chunk_size=config.get("chunk_size", 4000),
                min_chunk_size=config.get("min_chunk_size", 500),
                overlap_size=config.get("chunk_overlap", 200),
                respect_sentence_boundaries=config.get("respect_sentence_boundaries", True),
                detect_topic_shifts=config.get("detect_topic_shifts", True)
            )
    
    def construct_kg(self, construction_mode: str, max_workers: int) -> Dict[str, Any]:
        """
        Modified to support both page-based and document-aware processing.
        """
        print(f"Starting KG construction for document '{self.document_id}' in '{construction_mode}' mode")
        print(f"Processing mode: {self.processing_mode}")

        if self.processing_mode == "page_based":
            # Use existing page-based processing
            return self._construct_kg_page_based(construction_mode, max_workers)
        elif self.processing_mode == "document_aware":
            # Use new semantic chunk-based processing
            return self._construct_kg_document_aware(construction_mode, max_workers)
        else:
            raise ValueError(f"Unknown processing_mode: {self.processing_mode}")
    
    def _construct_kg_page_based(self, construction_mode: str, max_workers: int) -> Dict[str, Any]:
        """
        Existing page-based processing logic (unchanged).
        """
        # This is your existing implementation
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
            accumulated_kg = {"entities": [], "relationships": []}
            
            for page_data in pages_data:
                page_kg = self._process_single_page(page_data, accumulated_kg)
                
                if page_kg:
                    extracted_page_kgs.append(page_kg)
                    accumulated_kg = self.page_level_merger.merge_incrementally(accumulated_kg, page_kg)
                    
        elif construction_mode == "parallel":
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._process_single_page, page_data): page_data['page_number'] 
                    for page_data in pages_data
                }
                
                for future in concurrent.futures.as_completed(futures):
                    page_num = futures[future]
                    try:
                        page_kg = future.result()
                        if page_kg:
                            extracted_page_kgs.append(page_kg)
                    except Exception as e:
                        print(f"  Error processing page {page_num} in parallel: {e}")

        # Rest of existing page-based logic...
        return self._finalize_document_kg(extracted_page_kgs, construction_mode)
    
    def _construct_kg_document_aware(self, construction_mode: str, max_workers: int) -> Dict[str, Any]:
        """
        NEW: Document-aware processing using semantic chunks.
        """
        print(f"Creating semantic chunks for document '{self.document_id}'...")
        
        # Create semantic chunks
        semantic_chunks = self.semantic_chunker.create_chunks(self.pdf_parser)
        
        if not semantic_chunks:
            print(f"No semantic chunks created for document '{self.document_id}'. Returning empty KG.")
            return {"entities": [], "relationships": []}
        
        print(f"Created {len(semantic_chunks)} semantic chunks")
        
        # Display chunk statistics
        stats = self.semantic_chunker.get_chunk_statistics(semantic_chunks)
        print(f"Chunk statistics: avg size {stats['avg_chunk_size']:.0f} chars, "
              f"{stats['sections_detected']} sections detected")
        
        extracted_chunk_kgs: List[Dict[str, Any]] = []
        
        if construction_mode == "iterative":
            # Process chunks sequentially with accumulated context
            accumulated_kg = {"entities": [], "relationships": []}
            
            for chunk in semantic_chunks:
                chunk_kg = self._process_single_chunk(chunk, accumulated_kg)
                
                if chunk_kg:
                    extracted_chunk_kgs.append(chunk_kg)
                    # Merge this chunk's KG into accumulated context
                    accumulated_kg = self.page_level_merger.merge_incrementally(accumulated_kg, chunk_kg)
                    
        elif construction_mode == "parallel":
            # Process chunks in parallel (with less context sharing)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._process_single_chunk, chunk): chunk.id 
                    for chunk in semantic_chunks
                }
                
                for future in concurrent.futures.as_completed(futures):
                    chunk_id = futures[future]
                    try:
                        chunk_kg = future.result()
                        if chunk_kg:
                            extracted_chunk_kgs.append(chunk_kg)
                    except Exception as e:
                        print(f"  Error processing chunk {chunk_id} in parallel: {e}")
        
        # Dump individual chunk KGs if configured
        if self.config.get("dump_page_kgs", False):  # Reuse same config option
            self._dump_chunk_kgs(extracted_chunk_kgs)
        
        # Finalize document KG
        return self._finalize_document_kg(extracted_chunk_kgs, construction_mode)
    
    def _process_single_chunk(self, chunk: SemanticChunk, 
                             previous_graph_context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        NEW: Process a single semantic chunk using the LLM processor.
        """
        print(f"  Processing {chunk.id} (Section: {chunk.section_context})")
        
        try:
            # Convert SemanticChunk to format expected by PageLLMProcessor
            chunk_data = {
                'chunk_number': chunk.id,  # Use chunk ID instead of page number
                'text': chunk.text,
                'section_context': chunk.section_context,
                'previous_chunk_summary': chunk.previous_chunk_summary,
                'document_position': chunk.document_position
            }
            
            # Add image data if multimodal processing
            if self.page_llm_processor.extraction_mode == "multimodal":
                # For semantic chunks, we might not have page-specific images
                # This could be enhanced to extract relevant images
                chunk_data['image_base64'] = None
            
            # Use existing page processor with enhanced context
            chunk_kg = self.page_llm_processor.process_page(
                page_data=chunk_data,
                document_context_info=self._enhance_context_for_chunk(chunk),
                construction_mode="iterative",  # Always use iterative for chunks
                previous_graph_context=previous_graph_context
            )
            
            if chunk_kg and (chunk_kg.get("entities") or chunk_kg.get("relationships")):
                print(f"  Extracted KG for {chunk.id}: {len(chunk_kg.get('entities', []))} entities, "
                      f"{len(chunk_kg.get('relationships', []))} relationships.")
                
                # Add chunk metadata to results
                chunk_kg['source_chunk_id'] = chunk.id
                chunk_kg['section_context'] = chunk.section_context
                chunk_kg['document_position'] = chunk.document_position
                
                return chunk_kg
            else:
                print(f"  No entities or relationships extracted for {chunk.id}.")
                return None
                
        except Exception as e:
            print(f"  Error processing chunk {chunk.id}: {e}")
            return None
    
    def _enhance_context_for_chunk(self, chunk: SemanticChunk) -> Dict[str, Any]:
        """
        NEW: Enhance document context with chunk-specific information.
        """
        enhanced_context = self.document_context_info.copy()
        
        # Add chunk-specific context
        enhanced_context.update({
            "current_section": chunk.section_context,
            "previous_context": chunk.previous_chunk_summary,
            "document_position": chunk.document_position,
            "processing_mode": "document_aware"
        })
        
        return enhanced_context
    
    def _dump_chunk_kgs(self, chunk_kgs: List[Dict[str, Any]]) -> None:
        """
        NEW: Dump individual chunk KGs to files (similar to page KGs).
        """
        dump_dir = self.document_output_path / "chunk_kgs"
        dump_dir.mkdir(parents=True, exist_ok=True)
        
        for chunk_kg in chunk_kgs:
            chunk_id = chunk_kg.get("source_chunk_id", "unknown")
            dump_file = dump_dir / f"{chunk_id}_kg.json"
            
            try:
                with open(dump_file, "w") as f:
                    json.dump(chunk_kg, f, indent=2)
                
                # Also create visualization if entities exist
                if chunk_kg.get("entities"):
                    html_dump_file = dump_dir / f"{chunk_id}_kg.html"
                    self.graph_visualizer.export_interactive_html(chunk_kg, str(html_dump_file))
                    
            except Exception as e:
                print(f"Error dumping chunk KG for {chunk_id}: {e}")
    
    def _finalize_document_kg(self, extracted_kgs: List[Dict[str, Any]], 
                             construction_mode: str) -> Dict[str, Any]:
        """
        Finalize document KG (works for both page and chunk-based processing).
        """
        print(f"Merging {len(extracted_kgs)} KGs for document '{self.document_id}'...")
        
        if construction_mode == "iterative":
            # For iterative mode, we already have the final merged KG
            if extracted_kgs:
                final_document_kg = extracted_kgs[-1]  # Last KG has everything merged
            else:
                final_document_kg = {"entities": [], "relationships": []}
        else:
            # For parallel mode, merge all KGs
            final_document_kg = self.page_level_merger.merge_all_page_kgs(extracted_kgs)
        
        # Apply common KG utilities
        print("  Cleaning and normalizing document KG...")
        final_document_kg = clean_knowledge_graph(final_document_kg)
        final_document_kg = normalize_entity_ids(final_document_kg)
        
        print(f"KG construction for document '{self.document_id}' completed.")
        print(f"Final document KG: {len(final_document_kg.get('entities',[]))} entities, "
              f"{len(final_document_kg.get('relationships',[]))} relationships.")
        
        # Save the final KG
        self._save_document_kg(final_document_kg)
        
        return final_document_kg

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