import os
import json
import base64
import pymupdf
from io import BytesIO
from PIL import Image
import concurrent.futures
from typing import List, Dict, Tuple, Optional
from llm_client import AzureOpenAIClient
from KG_visualizer import KnowledgeGraphVisualizer
from dotenv import load_dotenv
from pathlib import Path
from ontology.loader import PEKGOntology
from utils.pdf_utils import PDFProcessor
from utils.kg_utils import (
    merge_knowledge_graphs, merge_multiple_knowledge_graphs, 
    clean_knowledge_graph, normalize_entity_ids
)

load_dotenv()

class FinancialKGBuilder:
    """
    A unified class to build financial knowledge graphs from PDF documents using Azure OpenAI.
    
    It supports:
    - Text-only extraction: Processes text content from PDF documents
    - Multimodal extraction: Processes both text and visual elements (tables, charts, etc.)
    
    Construction modes:
    - "iterative": Processes the PDF page by page, using previous pages' graphs as context
    - "onego": Processes all content at once or independently and then merges results
    - "parallel": Processes pages independently in parallel using multiple LLM instances
    
    The class provides functionality to extract, merge, consolidate, and visualize 
    knowledge graphs based on a predefined ontology.
    """
    
    def __init__(
        self, 
        model_name, 
        deployment_name, 
        project_name, 
        construction_mode="iterative",
        extraction_mode="text",
        max_workers=4,  # Number of parallel workers for parallel mode
        ontology_path: str = Path(__file__).resolve().parent / "ontology" / "pekg_ontology.yaml"
    ):
        """
        Initialize the FinancialKGBuilder with the model name and deployment name.
        
        Args:
            model_name (str): The name of the model to be used for extraction.
            deployment_name (str): The name of the deployment in Azure OpenAI.
            project_name (str): The name of the project for file naming.
            construction_mode (str): "iterative", "onego", or "parallel" for the KG construction approach.
            extraction_mode (str): Either "text" or "multimodal" for the extraction method.
            max_workers (int): Maximum number of parallel workers (for parallel mode only).
            ontology_path (str): Path to the ontology file.
        """
        self.model_name = model_name
        self.project_name = project_name
        self.llm = AzureOpenAIClient(model_name=model_name)
        self.deployment_name = deployment_name
        self.ontology = PEKGOntology(ontology_path)
        self.pdf_path = Path(__file__).resolve().parents[3] / "pages" / project_name / f"Project_{project_name}_Teaser.pdf"
        self.page_dpi = 300  # For image rendering in multimodal mode
        self.vizualizer = KnowledgeGraphVisualizer()
        self.pdf_processor = PDFProcessor(self.pdf_path)
        self.max_workers = max_workers
        
        # Validate construction mode
        if construction_mode not in ["iterative", "onego", "parallel"]:
            raise ValueError("construction_mode must be one of: 'iterative', 'onego', 'parallel'")
        self.construction_mode = construction_mode
        
        # Validate extraction mode
        if extraction_mode not in ["text", "multimodal"]:
            raise ValueError("extraction_mode must be either 'text' or 'multimodal'")
        self.extraction_mode = extraction_mode

    # ---------- TEXT-BASED EXTRACTION METHODS ----------

    def build_prompt_for_text_analysis(self, text: str, previous_graph: Dict = None) -> str:
        """
        Build the prompt for the LLM based on the provided text, ontology, and previous graph.
        
        Args:
            text (str): The text to be analyzed.
            previous_graph (Dict, optional): The merged graph from previous pages to provide context.
        
        Returns:
            str: The formatted prompt for the LLM.
        """
        ontology_desc = self.ontology.format_for_prompt()
        previous_graph_json = json.dumps(previous_graph) if previous_graph else "{}"
        
        # Base prompt
        prompt = f"""
        You are a financial information extraction expert.
        Your task is to extract an extensive and structured knowledge graph from the financial text provided.
        This is a financial document concerning the company {self.project_name}.
        The knowledge graph should include entities, relationships, and attributes based on the provided ontology.
        """
        
        # Add previous graph context for iterative mode
        if self.construction_mode == "iterative" and previous_graph:
            prompt += f"""
            Use the previous knowledge graph context to inform your extraction.
            
            Previous knowledge graph context (from previous pages):
            {previous_graph_json}
            """
        
        # Common format instructions
        prompt += f"""
        The ontology is as follows:

        {ontology_desc}

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
        """
        
        # Add iterative-specific instructions
        if self.construction_mode == "iterative" and previous_graph:
            prompt += """
            - If entities from previous pages appear again, use the same IDs for consistency.
            """
        
        # Common response instructions
        prompt += """
        ### RESPONSE ###
        Respond with *only* valid JSON in the specified format. Do not include any commentary. Do not include Markdown syntax (no ```json). Do not include explanations.
        """
        
        return prompt

    def analyze_text_with_llm(self, text: str, previous_graph: Dict = None) -> Dict:
        """
        Analyze the provided text using the LLM to extract a knowledge graph.
        
        Args:
            text (str): The text to be analyzed.
            previous_graph (dict, optional): The merged graph from previous pages to provide context.
        
        Returns:
            dict: The extracted knowledge graph in JSON format.
        """
        prompt = self.build_prompt_for_text_analysis(text, previous_graph)

        response = self.llm.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a financial information extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10000
        )
        
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.lstrip("```json").rstrip("```").strip()

        try:
            return json.loads(content)
        except Exception as e:
            print("Error parsing LLM response:", e)
            return {"entities": [], "relationships": []}

    # ---------- PARALLEL PROCESSING METHODS ----------

    def _create_llm_client(self):
        """
        Create a new LLM client instance. 
        Used to create separate clients for parallel processing.
        
        Returns:
            AzureOpenAIClient: A new instance of the LLM client.
        """
        # Create a completely new client instance to ensure isolation
        return AzureOpenAIClient(model_name=self.model_name)

    def _process_page_parallel(self, page_info: Dict) -> Dict:
        """
        Process a single page in parallel mode.
        This function is designed to be completely independent and thread-safe.
        
        Args:
            page_info (Dict): Dictionary containing page number, page data, and processing parameters.
            
        Returns:
            Dict: Knowledge graph extracted from the page.
        """
        page_num = page_info["page_num"]
        page_data = page_info["page_data"]
        
        # Create a dedicated LLM client for this worker
        local_llm_client = self._create_llm_client()
        
        print(f"Worker processing page {page_num + 1}...")
        
        try:
            # For text-only extraction
            if self.extraction_mode == "text":
                # Build the prompt for text analysis
                prompt = self.build_prompt_for_text_analysis(page_data["text"])
                
                # Make the API call directly with the local client
                response = local_llm_client.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a financial information extraction assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=10000
                )
                
                content = response.choices[0].message.content.strip()
                if content.startswith("```json"):
                    content = content.lstrip("```json").rstrip("```").strip()
                
                try:
                    result = json.loads(content)
                except Exception as e:
                    print(f"Error parsing LLM response for page {page_num + 1}: {e}")
                    result = {"entities": [], "relationships": []}
                
            # For multimodal extraction
            else:
                # First identify visual elements
                visual_analysis_prompt = f"""
                You are a financial document analysis expert.
                This is a financial document concerning the company {self.project_name}.
                Analyze this page from a financial document and identify all visual elements:
                1. Tables
                2. Charts/Graphs
                3. Diagrams
                5. Organizational charts
                6. Flow charts
                7. Financial statements
                8. Logoes

                For each identified element:
                - Describe what the element represents
                - Describe the key information presented

                This is page {page_data["page_num"]} of the document.
                """
                
                visual_analysis_response = local_llm_client.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a financial document analysis assistant capable of processing images."},
                        {"role": "user", "content": [
                            {"type": "text", "text": visual_analysis_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_data['image_base64']}"}}
                        ]}
                    ],
                    temperature=0.3,
                    max_tokens=4000
                )
                
                visual_analysis = {
                    "page_num": page_data["page_num"],
                    "analysis": visual_analysis_response.choices[0].message.content.strip()
                }
                
                # Extract data from visuals
                ontology_desc = self.ontology.format_for_prompt()
                visual_data_prompt = f"""
                You are a financial data extraction expert.  
                Your task is to extract an extensive and structured knowledge graph from the financial text provided.
                This is a financial document concerning the company {self.project_name}.
                The knowledge graph should include entities, relationships, and attributes respecting the provided ontology.
                
                The ontology we're using is:
                
                {ontology_desc}
                
                Based on your previous analysis:
                {visual_analysis["analysis"]}
                
                For each visual element (table, chart, graph, etc.):
                1. Extract all relevant entities, relationships, and attributes that match our ontology
                2. For tables, extract the data in structured form
                3. For charts/graphs, identify trends, key values, and relationships
                4. For diagrams/flowcharts, identify entities and their relationships
                5. For logos, associate them with the relevant company or brand
                
                Format your response as JSON following this structure:
                {{
                    "visual_elements": [
                        {{
                            "element_type": "table|chart|graph|diagram|statement",
                            "description": "Brief description of what this element shows",
                            "entities": [
                                {{"id": "e1", "type": "pekg:Company", "name": "ABC Capital"}},
                                {{"id": "e2", "type": "pekg:FundingRound", "roundAmount": 5000000, "roundDate": "2022-06-01"}}
                            ],
                            "relationships": [
                                {{"source": "e1", "target": "e2", "type": "pekg:receivedInvestment"}}
                            ],
                            "raw_data": {{}} // Include structured table data or key metrics when relevant
                        }}
                    ]
                }}
                
                Focus on extracting as much structured information as possible that aligns with our ontology.

                ### INSTRUCTIONS ###
                - Pay particular attention to numerical values, dates, and monetary amounts.
                - If the same entity appears under slightly different names (e.g. "DECK" vs. "DESK"), assume they refer to the same entity and normalize to the most frequent or contextually correct name.
                - Use your understanding of context to correct obvious typos.
                - Resolve entity mentions that refer to the same company/person/etc., and merge them into a single entity.

                ### RESPONSE ###
                Respond with *only* valid JSON in the specified format. Do not include any commentary. Do not include Markdown syntax. Do not include explanations.
                """

                visual_data_response = local_llm_client.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a financial data extraction assistant specializing in visual elements."},
                        {"role": "user", "content": [
                            {"type": "text", "text": visual_data_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_data['image_base64']}"}}
                        ]}
                    ],
                    temperature=0.1,
                    max_tokens=8000
                )
                
                visual_content = visual_data_response.choices[0].message.content.strip()
                if visual_content.startswith("```json"):
                    visual_content = visual_content.lstrip("```json").rstrip("```").strip()
                
                try:
                    visual_kg = json.loads(visual_content)
                except Exception as e:
                    print(f"Error parsing visual KG from page {page_num + 1}: {e}")
                    visual_kg = {"visual_elements": []}
                
                # Process text separately
                text_prompt = self.build_prompt_for_text_analysis(page_data["text"])
                text_response = local_llm_client.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a financial information extraction assistant."},
                        {"role": "user", "content": text_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=10000
                )
                
                text_content = text_response.choices[0].message.content.strip()
                if text_content.startswith("```json"):
                    text_content = text_content.lstrip("```json").rstrip("```").strip()
                
                try:
                    text_kg = json.loads(text_content)
                except Exception as e:
                    print(f"Error parsing text KG from page {page_num + 1}: {e}")
                    text_kg = {"entities": [], "relationships": []}
                
                # Merge visual and text knowledge graphs
                graphs_to_merge = []
                if "visual_elements" in visual_kg:
                    for element in visual_kg["visual_elements"]:
                        element_kg = {
                            "entities": element.get("entities", []),
                            "relationships": element.get("relationships", [])
                        }
                        graphs_to_merge.append(element_kg)
                
                graphs_to_merge.append(text_kg)
                result = merge_multiple_knowledge_graphs(graphs_to_merge)
            
            print(f"Worker completed page {page_num + 1}")
            
            # Return the result with the page number for identification
            return {
                "page_num": page_num,
                "graph": result
            }
            
        except Exception as e:
            print(f"Error in worker processing page {page_num + 1}: {e}")
            # Return an empty graph rather than failing the entire process
            return {
                "page_num": page_num,
                "graph": {"entities": [], "relationships": []}
            }
    
    def _build_knowledge_graph_parallel(self, dump: bool = False) -> Dict:
        """
        Build a knowledge graph from a PDF file by processing pages in parallel.
        Each page is processed completely independently by a separate thread.
        
        Args:
            dump (bool, optional): Flag to indicate if intermediate knowledge subgraphs should be saved.
            
        Returns:
            Dict: The final merged knowledge graph.
        """
        print(f"Starting parallel processing with {self.max_workers} workers")
        
        # Open the PDF document to get the number of pages
        doc = pymupdf.open(self.pdf_path)
        num_pages = len(doc)
        doc.close()
        
        # Extract all pages data first to avoid file access conflicts during parallel processing
        page_inputs = []
        for page_num in range(num_pages):
            print(f"Preparing data for page {page_num + 1}...")
            if self.extraction_mode == "multimodal":
                page_data = self.pdf_processor.extract_page_from_pdf(self.pdf_path, page_num)
            else:
                # For text-only, we need a similar structure but with just the text
                text = self.pdf_processor.extract_text_from_page(page_num)
                page_data = {
                    "page_num": page_num,
                    "text": text
                }
            
            page_inputs.append({
                "page_num": page_num,
                "page_data": page_data
            })
        
        print(f"Prepared data for {num_pages} pages, starting parallel processing...")
        
        # Process pages in parallel using ThreadPoolExecutor
        all_page_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all page processing tasks at once
            futures = [
                executor.submit(self._process_page_parallel, page_input)
                for page_input in page_inputs
            ]
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    all_page_results.append(result)
                    page_num = result["page_num"]
                    print(f"Received results for page {page_num + 1}")
                    
                    # Dump intermediate results if requested
                    if dump and result["graph"]:
                        self._save_page_graph(
                            result["graph"],
                            page_num,
                            "multimodal" if self.extraction_mode == "multimodal" else "text",
                            is_iterative=False
                        )
                except Exception as e:
                    print(f"Error retrieving result from worker: {e}")
        
        # Sort results by page number to ensure consistent merging
        all_page_results.sort(key=lambda x: x["page_num"])
        
        # Check if we have valid results
        if not all_page_results:
            print("Warning: No valid results were returned from parallel processing")
            return {"entities": [], "relationships": []}
            
        # Extract the graph data from results
        page_kgs = [result["graph"] for result in all_page_results if "graph" in result]
        
        print(f"Successfully processed {len(page_kgs)} pages out of {num_pages}")
        
        # Merge all page knowledge graphs
        print("Merging knowledge graphs from all pages...")
        merged_kg = merge_multiple_knowledge_graphs(page_kgs)
        
        # Final cleanup and normalization
        merged_kg = normalize_entity_ids(clean_knowledge_graph(merged_kg))
        
        return merged_kg

    def identify_visual_elements(self, page_data: Dict, previous_graph: Dict = None) -> Dict:
        """
        Use multimodal LLM to identify and classify visual elements on a page.
        
        Args:
            page_data (Dict): Dictionary containing page image and metadata.
            previous_graph (Dict, optional): The merged graph from previous pages to provide context.
        
        Returns:
            Dict: Information about identified visual elements.
        """
        prompt = f"""
        You are a financial document analysis expert.
        This is a financial document concerning the company {self.project_name}.
        Analyze this page from a financial document and identify all visual elements:
        1. Tables
        2. Charts/Graphs
        3. Diagrams
        5. Organizational charts
        6. Flow charts
        7. Financial statements
        8. Logoes

        For each identified element:
        - Describe what the element represents
        - Describe the key information presented

        This is page {page_data["page_num"]} of the document.
        """
        
        # Add previous graph context for iterative mode
        if self.construction_mode == "iterative" and previous_graph:
            previous_graph_json = json.dumps(previous_graph) if previous_graph else "{}"
            prompt += f"""
            Use the previous knowledge graph context to inform your analysis.
            
            Previous knowledge graph context (from previous pages):
            {previous_graph_json}
            """

        response = self.llm.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a financial document analysis assistant capable of processing images."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_data['image_base64']}"}}
                ]}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        visual_analysis = response.choices[0].message.content.strip()
        
        if visual_analysis.startswith("```json"):
            visual_analysis = visual_analysis.lstrip("```json").rstrip("```").strip()
        
        return {
            "page_num": page_data["page_num"],
            "analysis": visual_analysis
        }

    def extract_data_from_visuals(self, page_data: Dict, visual_analysis: Dict, previous_graph: Dict = None) -> Dict:
        """
        Extract structured data from identified visual elements using multimodal LLM.
        
        Args:
            page_data (Dict): Dictionary containing page image and metadata.
            visual_analysis (Dict): Analysis of visual elements on the page.
            previous_graph (Dict, optional): The merged graph from previous pages to provide context.
        
        Returns:
            Dict: Extracted data from visual elements in structured format.
        """
        ontology_desc = self.ontology.format_for_prompt()
        
        prompt = f"""
        You are a financial data extraction expert.  
        Your task is to extract an extensive and structured knowledge graph from the financial text provided.
        This is a financial document concerning the company {self.project_name}.
        The knowledge graph should include entities, relationships, and attributes respecting the provided ontology.
        
        The ontology we're using is:
        
        {ontology_desc}
        
        Based on your previous analysis:
        {visual_analysis["analysis"]}
        """
        
        # Add previous graph context for iterative mode
        if self.construction_mode == "iterative" and previous_graph:
            previous_graph_json = json.dumps(previous_graph) if previous_graph else "{}"
            prompt += f"""
            Previous knowledge graph context (from previous pages):
            {previous_graph_json}
            
            Use the previous knowledge graph to maintain consistency in entity naming and IDs.
            """
        
        # Common instructions
        prompt += """
        For each visual element (table, chart, graph, etc.):
        1. Extract all relevant entities, relationships, and attributes that match our ontology
        2. For tables, extract the data in structured form
        3. For charts/graphs, identify trends, key values, and relationships
        4. For diagrams/flowcharts, identify entities and their relationships
        5. For logos, associate them with the relevant company or brand
        
        Format your response as JSON following this structure:
        {
            "visual_elements": [
                {
                    "element_type": "table|chart|graph|diagram|statement",
                    "description": "Brief description of what this element shows",
                    "entities": [
                        {"id": "e1", "type": "pekg:Company", "name": "ABC Capital"},
                        {"id": "e2", "type": "pekg:FundingRound", "roundAmount": 5000000, "roundDate": "2022-06-01"}
                    ],
                    "relationships": [
                        {"source": "e1", "target": "e2", "type": "pekg:receivedInvestment"}
                    ],
                    "raw_data": {} // Include structured table data or key metrics when relevant
                }
            ]
        }
        
        Focus on extracting as much structured information as possible that aligns with our ontology.

        ### INSTRUCTIONS ###
        - Pay particular attention to numerical values, dates, and monetary amounts.
        - If the same entity appears under slightly different names (e.g. "DECK" vs. "DESK"), assume they refer to the same entity and normalize to the most frequent or contextually correct name.
        - Use your understanding of context to correct obvious typos.
        - Resolve entity mentions that refer to the same company/person/etc., and merge them into a single entity.

        ### RESPONSE ###
        Respond with *only* valid JSON in the specified format. Do not include any commentary. Do not include Markdown syntax. Do not include explanations.
        """

        response = self.llm.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a financial data extraction assistant specializing in visual elements."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_data['image_base64']}"}}
                ]}
            ],
            temperature=0.1,
            max_tokens=8000
        )
        
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```json"):
            content = content.lstrip("```json").rstrip("```").strip()
          
        try:
            return json.loads(content)
        except Exception as e:
            print(f"Error parsing JSON from page {page_data['page_num']}:", e)
            print("Raw content:", content)
            return {"visual_elements": []}

    def analyze_page(self, page_data: Dict, previous_graph: Dict = None) -> Dict:
        """
        Perform comprehensive analysis of a single page, combining text and visual analysis.
        
        Args:
            page_data (Dict): Dictionary containing page image and metadata.
            previous_graph (Dict, optional): The merged graph from previous pages to provide context.
        
        Returns:
            Dict: Combined knowledge graph from the page.
        """
        # For multimodal extraction
        if self.extraction_mode == "multimodal":
            # We use the previous graph as context for extraction, but don't merge with it here
            visual_analysis = self.identify_visual_elements(page_data, previous_graph)
            visual_kg = self.extract_data_from_visuals(page_data, visual_analysis, previous_graph)
            text_kg = self.analyze_text_with_llm(page_data["text"], previous_graph)
            
            graphs_to_merge = []
            
            if "visual_elements" in visual_kg:
                for element in visual_kg["visual_elements"]:
                    element_kg = {
                        "entities": element.get("entities", []),
                        "relationships": element.get("relationships", [])
                    }
                    graphs_to_merge.append(element_kg)
            
            graphs_to_merge.append(text_kg)
            
            # For the iterative mode, we'll just combine the current page's components
            # without merging with previous_graph (that will happen in the calling function)
            page_kg = merge_multiple_knowledge_graphs(graphs_to_merge)
            
            return page_kg
        
        # For text-only extraction
        else:
            return self.analyze_text_with_llm(page_data["text"], previous_graph)

    # ---------- MAIN KNOWLEDGE GRAPH BUILDING METHODS ----------

    def build_knowledge_graph_from_pdf(self, dump: bool = False) -> Dict:
        """
        Build a knowledge graph from a PDF file based on the configured extraction and construction modes.
        
        Args:
            dump (bool, optional): Flag to indicate if intermediate knowledge subgraphs should be saved.
        
        Returns:
            dict: The final merged knowledge graph.
        """
        # Handle parallel construction mode
        if self.construction_mode == "parallel":
            merged_graph = self._build_knowledge_graph_parallel(dump)
        
        # Handle text-only extraction (non-parallel)
        elif self.extraction_mode == "text":
            if self.construction_mode == "onego":
                # One-go approach: process entire document at once
                text = self.pdf_processor.extract_text()
                merged_graph = self.analyze_text_with_llm(text)
            else:
                # Iterative approach: process page by page
                pages_text = self.pdf_processor.extract_text_as_list()
                merged_graph = {"entities": [], "relationships": []}

                for i, page_text in enumerate(pages_text):
                    print(f"Processing page {i+1}...")
                    page_graph = self.analyze_text_with_llm(page_text, merged_graph)
                    merged_graph = merge_knowledge_graphs(merged_graph, page_graph)

                    if dump:
                        self._save_page_graph(page_graph, i, "text")
        
        # Handle multimodal extraction (non-parallel)
        else:
            if self.construction_mode == "iterative":
                merged_graph = self._build_multimodal_knowledge_graph_iterative(dump)
            else:
                merged_graph = self._build_multimodal_knowledge_graph_onego(dump)

        # Final cleanup and normalization
        merged_graph = normalize_entity_ids(clean_knowledge_graph(merged_graph))
        
        # Optionally consolidate the graph
        if len(merged_graph.get("entities", [])) > 20:  # Only consolidate larger graphs
            merged_graph = self.consolidate_knowledge_graph(merged_graph)
            
        print("Knowledge graph building process completed.")
        return merged_graph

    def _build_multimodal_knowledge_graph_iterative(self, dump: bool = False) -> Dict:
        """
        Build a knowledge graph iteratively from the pages of a PDF using multimodal analysis.
        Each page's subgraph is merged with the context of previous pages.
        
        Args:
            dump (bool, optional): Flag to indicate if the knowledge subgraphs should be saved.
        
        Returns:
            dict: The final merged knowledge graph.
        """
        doc = pymupdf.open(self.pdf_path)
        num_pages = len(doc)
        doc.close()
        
        merged_graph = {"entities": [], "relationships": []}
        
        for page_num in range(num_pages):
            print(f"Processing page {page_num + 1} of {num_pages}...")
            page_data = self.pdf_processor.extract_page_from_pdf(self.pdf_path, page_num)
            
            # First, extract the graphs from the visual elements and text on the current page
            # Use the previous graph as context but don't merge with it yet
            visual_analysis = self.identify_visual_elements(page_data, merged_graph)
            visual_kg = self.extract_data_from_visuals(page_data, visual_analysis, merged_graph)
            text_kg = self.analyze_text_with_llm(page_data["text"], merged_graph)
            
            # Collect all subgraphs from the current page only
            current_page_graphs = []
            if "visual_elements" in visual_kg:
                for element in visual_kg["visual_elements"]:
                    element_kg = {
                        "entities": element.get("entities", []),
                        "relationships": element.get("relationships", [])
                    }
                    current_page_graphs.append(element_kg)
            current_page_graphs.append(text_kg)
            
            # Create a subgraph for the current page only
            page_only_graph = merge_multiple_knowledge_graphs(current_page_graphs)
            
            if dump:
                self._save_page_graph(page_only_graph, page_num, "multimodal")
            
            # Now merge the current page's graph with the accumulated graph
            merged_graph = merge_knowledge_graphs(merged_graph, page_only_graph)
            merged_graph = clean_knowledge_graph(merged_graph)
            
            print(f"Completed page {page_num + 1}/{num_pages}")
            print(f"Current graph: {len(merged_graph['entities'])} entities, {len(merged_graph['relationships'])} relationships")
        
        # Final cleanup and normalization of entity IDs
        merged_graph = normalize_entity_ids(clean_knowledge_graph(merged_graph))
        return merged_graph

    def _build_multimodal_knowledge_graph_onego(self, dump: bool = False) -> Dict:
        """
        Build the knowledge graph from a PDF file using one-go multimodal analysis.
        Processes all pages independently and then merges results.
        
        Args:
            dump (bool, optional): Flag to indicate if the individual page subgraphs should be saved.
        
        Returns:
            dict: The extracted knowledge graph in JSON format.
        """
        doc = pymupdf.open(self.pdf_path)
        num_pages = len(doc)
        doc.close()
        
        page_kgs = []
        for page_num in range(num_pages):
            print(f"Processing page {page_num + 1} of {num_pages}...")
            page_data = self.pdf_processor.extract_page_from_pdf(self.pdf_path, page_num)
            
            # Extract knowledge graph from this page
            page_kg = self.analyze_page(page_data)
            
            if dump:
                self._save_page_graph(page_kg, page_num, "multimodal", is_iterative=False)
            
            page_kgs.append(page_kg)
            print(f"Completed page {page_num + 1}/{num_pages}")
        
        print("Merging all page knowledge graphs...")
        merged_kg = merge_multiple_knowledge_graphs(page_kgs)
        
        # Final cleanup and normalization
        merged_kg = normalize_entity_ids(clean_knowledge_graph(merged_kg))
        
        return merged_kg

    # ---------- UTILITY METHODS ----------

    def _save_page_graph(self, graph: Dict, page_num: int, mode: str, is_iterative: bool = True) -> None:
        """
        Save a page's knowledge graph to disk as JSON and HTML visualization.
        
        Args:
            graph (Dict): The knowledge graph to save.
            page_num (int): The page number.
            mode (str): The extraction mode ("text" or "multimodal").
            is_iterative (bool): Whether this is from iterative mode or one-go mode.
        """
        # Clean and normalize the page graph for visualization
        page_viz_graph = normalize_entity_ids(clean_knowledge_graph(graph))
        
        # Create output directory
        output_dir = Path(__file__).resolve().parents[3] / "outputs" / self.project_name / "pages"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        construction_suffix = "iterative" if is_iterative else "onego"
        
        # Save JSON
        output_file = output_dir / f"{mode}_knowledge_graph_page_{page_num + 1}_{self.model_name}_{construction_suffix}.json"
        with open(output_file, "w") as f:
            json.dump(page_viz_graph, f, indent=2)
        
        # Save HTML visualization
        output_file = str(output_dir / f"{mode}_knowledge_graph_page_{page_num + 1}_{self.model_name}_{construction_suffix}.html")
        self.vizualizer.export_interactive_html(page_viz_graph, output_file)
        print(f"Knowledge graph visualization for page {page_num + 1} saved to {output_file}")

    def consolidate_knowledge_graph(self, kg: Dict) -> Dict:
        """
        Use LLM to consolidate and clean up the knowledge graph, resolving duplicates
        and inconsistencies.
        
        Args:
            kg (Dict): The raw knowledge graph to consolidate.
        
        Returns:
            Dict: The consolidated knowledge graph.
        """
        kg_str = json.dumps(kg, indent=2)
        
        prompt = f"""
        I need you to clean up and consolidate this financial knowledge graph. 
        Resolve any duplicates or inconsistencies to create a coherent, unified graph.
        
        Here's the current graph:
        
        ```json
        {kg_str}
        ```
        
        Your tasks:
        
        1. Identify and merge duplicate entities (entities that refer to the same real-world object)
        2. Standardize entity attributes (e.g., consistent date formats, number formats)
        3. Ensure relationship consistency (remove redundant relationships)
        4. Clean up any missing or null values
        
        Return the consolidated graph in the same JSON format, with only these two top-level keys:
        - "entities": list of entity objects
        - "relationships": list of relationship objects
        
        Do not add any commentary or explanation. Respond with valid JSON only.
        """
        
        response = self.llm.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a knowledge graph consolidation expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=12000
        )
        
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.lstrip("```json").rstrip("```").strip()
        
        try:
            consolidated_kg = json.loads(content)
            # Apply additional merging of similar entities that the LLM might have missed
            return normalize_entity_ids(clean_knowledge_graph(consolidated_kg))
        except Exception as e:
            print("Error parsing consolidated knowledge graph:", e)
            return normalize_entity_ids(clean_knowledge_graph(kg))

    def save_knowledge_graph(self, data: dict):
        """
        Save the knowledge graph data to a JSON file and HTML visualization.
        
        Args:
            data (dict): The knowledge graph data to be saved.
        """
        # Create output directory
        output_dir = Path(__file__).resolve().parents[3] / "outputs" / self.project_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename based on modes
        filename_prefix = "multimodal" if self.extraction_mode == "multimodal" else "text"
        
        # Save JSON file
        json_output_file = output_dir / f"{filename_prefix}_knowledge_graph_{self.project_name}_{self.model_name}_{self.construction_mode}.json"
        with open(json_output_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Knowledge graph saved to {json_output_file}")
        
        # Save HTML visualization
        html_output_file = str(output_dir / f"{filename_prefix}_knowledge_graph_{self.project_name}_{self.model_name}_{self.construction_mode}.html")
        self.vizualizer.export_interactive_html(data, html_output_file)
        print(f"Knowledge graph visualization saved to {html_output_file}")
        
        # Save performance metrics if in parallel mode
        if self.construction_mode == "parallel":
            metrics = {
                "construction_mode": self.construction_mode,
                "extraction_mode": self.extraction_mode,
                "max_workers": self.max_workers,
                "entity_count": len(data.get("entities", [])),
                "relationship_count": len(data.get("relationships", [])),
            }
            
            metrics_file = output_dir / f"{filename_prefix}_metrics_{self.project_name}_{self.model_name}_{self.construction_mode}.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Performance metrics saved to {metrics_file}")
