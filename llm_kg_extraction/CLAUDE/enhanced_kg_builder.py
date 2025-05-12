import os
import json
import time
import pymupdf
import hashlib
from typing import List, Dict, Tuple, Optional, Generator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from tqdm import tqdm
import logging

from llm_client import AzureOpenAIClient
from KG_visualizer import KnowledgeGraphVisualizer
from utils.pdf_utils import PDFProcessor
from ontology.loader import PEKGOntology
from utils.kg_utils import merge_knowledge_graphs
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FinancialKGBuilder')

load_dotenv()

class LLMCache:
    """Cache for LLM responses to avoid redundant API calls."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LLM cache initialized at {cache_dir}")
        
    def _generate_key(self, prompt: str) -> str:
        """Generate a unique key for a prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[Dict]:
        """Retrieve a cached response for a prompt if it exists."""
        key = self._generate_key(prompt)
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    logger.debug(f"Cache hit for key {key}")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading from cache: {e}")
                return None
        return None
    
    def set(self, prompt: str, response: Dict) -> None:
        """Cache a response for a prompt."""
        key = self._generate_key(prompt)
        cache_file = self.cache_dir / f"{key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(response, f)
            logger.debug(f"Cached response for key {key}")
        except Exception as e:
            logger.warning(f"Error writing to cache: {e}")


class FinancialKGBuilder:
    """
    An enhanced class to build a financial knowledge graph from text using Azure OpenAI.
    Features improvements for memory management, parallelization, and caching.
    """
    
    def __init__(self, model_name, deployment_name, project_name, construction_mode, 
                 ontology_path: str = Path(__file__).resolve().parent / "ontology" / "pekg_ontology.yaml",
                 max_workers: int = 4,
                 batch_size: int = 3,
                 use_cache: bool = True):
        """
        Initialize the FinancialKGBuilder with enhanced parameters.
        
        Args:
            model_name (str): The name of the model to be used for extraction.
            deployment_name (str): The name of the deployment in Azure OpenAI.
            project_name (str): Name of the project for file organization.
            construction_mode (str): The construction mode ("iterative", "onego", or "parallel").
            ontology_path (str): Path to the ontology file.
            max_workers (int): Maximum number of workers for parallel processing.
            batch_size (int): Number of pages to process in a single batch.
            use_cache (bool): Whether to use caching for LLM responses.
        """
        self.model_name = model_name
        self.project_name = project_name
        self.llm = AzureOpenAIClient(model_name=model_name)
        self.deployment_name = deployment_name
        self.ontology = PEKGOntology(ontology_path)
        self.pdf_path = Path(__file__).resolve().parents[3] / "pages" / project_name / f"Project_{project_name}_Teaser.pdf"
        self.construction_mode = construction_mode
        self.visualizer = KnowledgeGraphVisualizer()
        self.pdf_processor = PDFProcessor(self.pdf_path)
        
        # Enhanced parameters
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.use_cache = use_cache
        
        # Setup cache
        if self.use_cache:
            cache_dir = Path(__file__).resolve().parents[3] / "cache" / project_name
            self.cache = LLMCache(cache_dir)
        
        # Setup output directories
        self.output_dir = Path(__file__).resolve().parents[3] / "outputs" / self.project_name
        self.pages_dir = self.output_dir / "pages"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized FinancialKGBuilder for project '{project_name}' using model '{model_name}'")
        logger.info(f"Construction mode: {construction_mode}, Max workers: {max_workers}, Batch size: {batch_size}")

    def build_prompt(self, text: str, previous_graph: Dict = None) -> str:
        """
        Build the prompt for the LLM based on the provided text, ontology, and previous graph.
        
        Args:
            text (str): The text to be analyzed.
            previous_graph (dict, optional): The merged subgraph from previous pages to provide context.
            
        Returns:
            str: The formatted prompt for the LLM.
        """
        ontology_desc = self.ontology.format_for_prompt()
        previous_graph_json = json.dumps(previous_graph) if previous_graph else "{}"
        
        prompt = f"""
        You are a financial information extraction expert.
        Your task is to extract an extensive and structured knowledge graph from the financial text provided.
        The knowledge graph should include entities, relationships, and attributes based on the provided ontology.
        If provided, use the previous knowledge graph context to inform your extraction.
        The ontology is as follows:

        {ontology_desc}

        Previous knowledge graph context (from previous pages):
        {previous_graph_json}

        ###FORMAT ###
        Output a JSON object like:
        {{
        "entities": [
            {{"id": "e1", "type": "pekg:Company", "name": "ABC Capital"}},
            {{"id": "e2", "type": "pekg:FundingRound", "roundAmount": 5000000, "roundDate": "2022-06-01"}}
        ],
        "relationships": [
            {{"source": "e1", "target": "e2", "type": "pekg:receivedInvestment"}}
        ]
        }}
        
        ### TEXT ###
        \"\"\"{text}\"\"\"

        ### INSTRUCTIONS ###
        - Pay particular attention to numerical values, dates, and monetary amounts.
        - If the same entity appears under slightly different names (e.g. "DECK" vs. "DESK"), assume they refer to the same entity and normalize to the most frequent or contextually correct name.
        - Use your understanding of context to correct obvious typos.
        - Resolve entity mentions that refer to the same company/person/etc., and merge them into a single entity.

        ### RESPONSE ###
        Respond with *only* valid JSON in the specified format. Do not include any commentary. Do not include Markdown syntax (no ```json). Do not include explanations.
        """
        return prompt

    def analyze_text_with_llm(self, text: str, previous_graph: Dict = None) -> Dict:
        """
        Analyze the provided text using the LLM to extract a knowledge graph.
        Uses caching if enabled to avoid redundant API calls.
        
        Args:
            text (str): The text to be analyzed.
            previous_graph (dict, optional): The merged graph from previous pages to provide context.
            
        Returns:
            dict: The extracted knowledge graph in JSON format.
        """
        prompt = self.build_prompt(text, previous_graph)
        
        # Check cache first if enabled
        if self.use_cache:
            cached_response = self.cache.get(prompt)
            if cached_response:
                logger.info("Using cached LLM response")
                return cached_response
        
        # If not in cache or cache not enabled, call the API
        try:
            start_time = time.time()
            response = self.llm.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a financial information extraction assistant. "
                     "Your task is to extract a knowledge graph from the financial text provided."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10000
            )
            elapsed = time.time() - start_time
            logger.info(f"LLM API call completed in {elapsed:.2f} seconds")
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content.lstrip("```json").rstrip("```").strip()

            try:
                result = json.loads(content)
                
                # Cache the result if caching is enabled
                if self.use_cache:
                    self.cache.set(prompt, result)
                    
                return result
            except Exception as e:
                logger.error(f"Error parsing LLM response: {e}")
                logger.debug(f"Problematic content: {content[:200]}...")
                return {}
                
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return {}

    def process_batch(self, batch: List[Tuple[int, str]], previous_graph: Dict = None) -> Dict:
        """
        Process a batch of pages and merge their knowledge graphs.
        
        Args:
            batch: List of (page_index, page_text) tuples
            previous_graph: The graph built so far
            
        Returns:
            The merged knowledge graph for this batch
        """
        batch_graph = previous_graph.copy() if previous_graph else {}
        
        for page_idx, page_text in batch:
            logger.info(f"Processing page {page_idx+1}...")
            page_graph = self.analyze_text_with_llm(page_text, batch_graph)
            batch_graph = merge_knowledge_graphs(batch_graph, page_graph)
            
            # Save individual page graph if needed
            self._save_page_graph(page_idx, page_graph)
            
        return batch_graph

    def _save_page_graph(self, page_idx: int, page_graph: Dict) -> None:
        """Save the knowledge graph for a single page."""
        if not page_graph:
            return
            
        # Filter relationships to only include those between entities on this page
        entity_ids = {entity['id'] for entity in page_graph.get("entities", [])}
        filtered_relationships = [
            rel for rel in page_graph.get("relationships", [])
            if rel["source"] in entity_ids and rel["target"] in entity_ids
        ]
        page_graph["relationships"] = filtered_relationships
        
        # Save as JSON
        output_file = self.pages_dir / f"knowledge_graph_page_{page_idx+1}_{self.model_name}_{self.construction_mode}.json"
        with open(output_file, "w") as f:
            json.dump(page_graph, f, indent=2)
        
        # Save visualization
        html_output = str(self.pages_dir / f"knowledge_graph_page_{page_idx+1}_{self.model_name}_{self.construction_mode}.html")
        self.visualizer.export_interactive_html(page_graph, html_output)

    def chunk_document(self, pages_text: List[str]) -> List[List[Tuple[int, str]]]:
        """
        Divide document pages into batches for processing.
        
        Args:
            pages_text: List of page texts
            
        Returns:
            List of batches, where each batch is a list of (page_index, page_text) tuples
        """
        batches = []
        current_batch = []
        
        for i, page_text in enumerate(pages_text):
            current_batch.append((i, page_text))
            
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = []
        
        # Add any remaining pages
        if current_batch:
            batches.append(current_batch)
            
        return batches

    def _process_single_batch_worker(self, batch_data: Tuple[List[Tuple[int, str]], Dict]) -> Dict:
        """Worker function for processing a single batch in parallel mode."""
        batch, previous_graph = batch_data
        return self.process_batch(batch, previous_graph)

    def build_knowledge_graph_iterative(self) -> Dict:
        """
        Build knowledge graph iteratively, page by page.
        
        Returns:
            The complete knowledge graph
        """
        pages_text = self.pdf_processor.extract_text_as_list()
        merged_graph = {}
        
        for i, page_text in enumerate(tqdm(pages_text, desc="Processing pages")):
            logger.info(f"Processing page {i+1} of {len(pages_text)}...")
            page_graph = self.analyze_text_with_llm(page_text, merged_graph)
            merged_graph = merge_knowledge_graphs(merged_graph, page_graph)
            self._save_page_graph(i, page_graph)
            
        return merged_graph

    def build_knowledge_graph_batch(self) -> Dict:
        """
        Build knowledge graph using batch processing.
        
        Returns:
            The complete knowledge graph
        """
        pages_text = self.pdf_processor.extract_text_as_list()
        batches = self.chunk_document(pages_text)
        merged_graph = {}
        
        for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
            logger.info(f"Processing batch {batch_idx+1} of {len(batches)}...")
            batch_graph = self.process_batch(batch, merged_graph)
            merged_graph = merge_knowledge_graphs(merged_graph, batch_graph)
            
        return merged_graph

    def build_knowledge_graph_parallel(self) -> Dict:
        """
        Build knowledge graph using parallel processing for batches.
        
        Returns:
            The complete knowledge graph
        """
        pages_text = self.pdf_processor.extract_text_as_list()
        batches = self.chunk_document(pages_text)
        merged_graph = {}
        
        # Create tasks for parallel execution
        tasks = []
        for batch in batches:
            # Create a copy of the current merged graph for this batch
            tasks.append((batch, merged_graph.copy()))
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_single_batch_worker, task) for task in tasks]
            
            # Collect and merge results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                batch_graph = future.result()
                merged_graph = merge_knowledge_graphs(merged_graph, batch_graph)
        
        return merged_graph

    def build_knowledge_graph_onego(self) -> Dict:
        """
        Build knowledge graph from the entire document at once.
        
        Returns:
            The knowledge graph
        """
        text = self.pdf_processor.extract_text()
        logger.info(f"Processing entire document at once ({len(text)} characters)...")
        return self.analyze_text_with_llm(text)

    def build_knowledge_graph_streaming(self) -> Dict:
        """
        Build knowledge graph using a streaming approach to minimize memory usage.
        
        Returns:
            The final knowledge graph
        """
        pages_text = self.pdf_processor.extract_text_as_list()
        merged_graph = {}
        
        # Process document in chunks to reduce memory usage
        chunk_size = 3  # pages per chunk
        for i in range(0, len(pages_text), chunk_size):
            chunk_text = ' '.join(pages_text[i:i+chunk_size])
            logger.info(f"Processing chunk {i//chunk_size + 1} (pages {i+1}-{min(i+chunk_size, len(pages_text))})...")
            
            # Process this chunk with the existing graph context
            chunk_graph = self.analyze_text_with_llm(chunk_text, merged_graph)
            
            # Update the merged graph
            merged_graph = merge_knowledge_graphs(merged_graph, chunk_graph)
            
            # Write intermediate results to disk
            interim_file = self.output_dir / f"interim_graph_chunk_{i//chunk_size + 1}.json"
            with open(interim_file, "w") as f:
                json.dump(merged_graph, f, indent=2)
        
        return merged_graph

    def build_knowledge_graph_from_pdf(self) -> Dict:
        """
        Build a knowledge graph from a PDF using the specified construction mode.
        
        Returns:
            dict: The final merged knowledge graph.
        """
        start_time = time.time()
        logger.info(f"Starting knowledge graph extraction using {self.construction_mode} mode")
        
        if self.construction_mode == "onego":
            graph = self.build_knowledge_graph_onego()
        elif self.construction_mode == "iterative":
            graph = self.build_knowledge_graph_iterative()
        elif self.construction_mode == "batch":
            graph = self.build_knowledge_graph_batch()
        elif self.construction_mode == "parallel":
            graph = self.build_knowledge_graph_parallel()
        elif self.construction_mode == "streaming":
            graph = self.build_knowledge_graph_streaming()
        else:
            logger.error(f"Unknown construction mode: {self.construction_mode}")
            raise ValueError(f"Unknown construction mode: {self.construction_mode}")
        
        elapsed = time.time() - start_time
        logger.info(f"Knowledge graph building completed in {elapsed:.2f} seconds")
        
        return graph
    
    def save_knowledge_graph(self, data: dict) -> None:
        """
        Save the knowledge graph data to JSON and HTML files.
        
        Args:
            data (dict): The knowledge graph data to be saved.
        """
        if not data:
            logger.warning("No knowledge graph data to save")
            return
            
        # Save JSON
        json_output_file = self.output_dir / f"knowledge_graph_{self.project_name}_{self.model_name}_{self.construction_mode}.json"
        with open(json_output_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Knowledge graph saved to {json_output_file}")
        
        # Save HTML visualization
        html_output_file = str(self.output_dir / f"knowledge_graph_{self.project_name}_{self.model_name}_{self.construction_mode}.html")
        self.visualizer.export_interactive_html(data, html_output_file)
        logger.info(f"Knowledge graph visualization saved to {html_output_file}")


class IncrementalKGBuilder(FinancialKGBuilder):
    """
    A specialized version of the KG builder that saves results incrementally
    and can resume from previous runs to avoid reprocessing.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with the same parameters as FinancialKGBuilder."""
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_latest_checkpoint(self) -> Tuple[int, Dict]:
        """
        Get the latest checkpoint if available.
        
        Returns:
            Tuple of (last processed page, checkpoint graph)
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            return 0, {}
            
        # Extract page numbers from filenames and find the highest
        checkpoint_pages = [int(cp.stem.split('_')[1]) for cp in checkpoints]
        latest_page = max(checkpoint_pages)
        
        # Load the latest checkpoint
        latest_file = self.checkpoint_dir / f"checkpoint_{latest_page}.json"
        with open(latest_file, 'r') as f:
            checkpoint_data = json.load(f)
            
        logger.info(f"Resuming from checkpoint at page {latest_page}")
        return latest_page, checkpoint_data
    
    def save_checkpoint(self, page_idx: int, graph: Dict) -> None:
        """Save a checkpoint of the current processing state."""
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{page_idx}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(graph, f, indent=2)
        logger.info(f"Saved checkpoint at page {page_idx}")
    
    def build_knowledge_graph_from_pdf(self) -> Dict:
        """
        Build a knowledge graph with checkpointing support.
        Can resume from the last checkpoint if processing was interrupted.
        
        Returns:
            The final knowledge graph
        """
        start_page, merged_graph = self.get_latest_checkpoint()
        pages_text = self.pdf_processor.extract_text_as_list()
        
        for i, page_text in enumerate(pages_text[start_page:], start=start_page):
            logger.info(f"Processing page {i+1} of {len(pages_text)}...")
            page_graph = self.analyze_text_with_llm(page_text, merged_graph)
            merged_graph = merge_knowledge_graphs(merged_graph, page_graph)
            
            # Save page results
            self._save_page_graph(i, page_graph)
            
            # Save checkpoint every 5 pages
            if (i + 1) % 5 == 0 or i == len(pages_text) - 1:
                self.save_checkpoint(i + 1, merged_graph)
        
        return merged_graph


class KGEvaluator:
    """
    A class to evaluate the quality of generated knowledge graphs.
    Can be used to compare different extraction approaches.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def calculate_metrics(self, graph: Dict) -> Dict:
        """Calculate metrics for a knowledge graph."""
        metrics = {
            "entity_count": len(graph.get("entities", [])),
            "relationship_count": len(graph.get("relationships", [])),
            "entity_types": self._count_entity_types(graph),
            "density": self._calculate_density(graph),
            "completeness": self._estimate_completeness(graph)
        }
        return metrics
    
    def _count_entity_types(self, graph: Dict) -> Dict:
        """Count entities by type."""
        type_counts = {}
        for entity in graph.get("entities", []):
            entity_type = entity.get("type", "unknown")
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        return type_counts
    
    def _calculate_density(self, graph: Dict) -> float:
        """Calculate graph density (relationships per entity)."""
        entity_count = len(graph.get("entities", []))
        relationship_count = len(graph.get("relationships", []))
        
        if entity_count == 0:
            return 0
        return relationship_count / entity_count
    
    def _estimate_completeness(self, graph: Dict) -> float:
        """
        Estimate graph completeness based on entity attributes.
        Returns a score between 0 and 1.
        """
        if not graph.get("entities"):
            return 0
            
        total_attributes = 0
        filled_attributes = 0
        
        for entity in graph.get("entities", []):
            # Count all attributes except 'id' and 'type'
            entity_attrs = [key for key in entity.keys() if key not in ['id', 'type']]
            total_attributes += len(entity_attrs)
            
            # Count non-empty attributes
            filled_attributes += sum(1 for key in entity_attrs if entity[key])
        
        if total_attributes == 0:
            return 0
        return filled_attributes / total_attributes
    
    def compare_graphs(self, graph1: Dict, graph2: Dict) -> Dict:
        """Compare two knowledge graphs and return differences."""
        metrics1 = self.calculate_metrics(graph1)
        metrics2 = self.calculate_metrics(graph2)
        
        comparison = {
            "graph1_metrics": metrics1,
            "graph2_metrics": metrics2,
            "entity_count_diff": metrics2["entity_count"] - metrics1["entity_count"],
            "relationship_count_diff": metrics2["relationship_count"] - metrics1["relationship_count"],
            "density_diff": metrics2["density"] - metrics1["density"],
            "completeness_diff": metrics2["completeness"] - metrics1["completeness"]
        }
        
        return comparison
