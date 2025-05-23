import os
import json
import base64
import pymupdf
from io import BytesIO
from PIL import Image
import concurrent.futures
from typing import List, Dict, Tuple, Optional

from llm_client import AzureOpenAIClient, VertexAIClient

from KG_visualizer import KnowledgeGraphVisualizer
from company_identifier import CompanyIdentifier 
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
    A unified class to build financial knowledge graphs from PDF documents using 
    Azure OpenAI or Google Gemini (via google.generativeai SDK).

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
        model_name: str,
        project_name: str,
        llm_provider: str = "azure",
        deployment_name: Optional[str] = None,
        construction_mode: str = "iterative",
        extraction_mode: str = "text",
        max_workers: int = 4,
        ontology_path: str = str(Path(__file__).resolve().parent / "ontology" / "pekg_ontology.yaml")
    ):
        self.model_name = model_name 
        self.project_name = project_name
        self.llm_provider = llm_provider.lower()
        self.deployment_name = deployment_name

        if self.llm_provider == "azure":
            if not self.deployment_name:
                raise ValueError("deployment_name is required for Azure OpenAI provider.")
            self.llm = AzureOpenAIClient(model_name=self.model_name)
            self.llm_company_id = AzureOpenAIClient(model_name="gpt-4.1-mini")
        elif self.llm_provider == "vertexai":
            self.llm = VertexAIClient(model_name=self.model_name)
            self.llm_company_id = VertexAIClient(model_name="gemini-2.5-flash-preview-04-17")
        else:
            raise ValueError(f"Unsupported llm_provider: {self.llm_provider}. Choose 'azure' or 'vertexai'.")

        self.ontology = PEKGOntology(ontology_path)
        pdf_dir = Path(__file__).resolve().parents[3] / "pages" / project_name
        pdf_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_path = pdf_dir / f"Project_{project_name}_Teaser.pdf"
        
        # Define json_output_path attribute
        self.json_output_path = Path(__file__).resolve().parents[3] / "outputs" / self.project_name
        self.json_output_path.mkdir(parents=True, exist_ok=True)
        
        if not self.pdf_path.exists():
            print(f"Warning: PDF file not found at {self.pdf_path}. Certain operations might fail or use defaults.")

        self.page_dpi = 300
        self.vizualizer = KnowledgeGraphVisualizer()
        self.pdf_processor = PDFProcessor(self.pdf_path)
        self.max_workers = max_workers

        if construction_mode not in ["iterative", "onego", "parallel"]:
            raise ValueError("construction_mode must be one of: 'iterative', 'onego', 'parallel'")
        self.construction_mode = construction_mode

        if extraction_mode not in ["text", "multimodal"]:
            raise ValueError("extraction_mode must be either 'text' or 'multimodal'")
        self.extraction_mode = extraction_mode
        
        try:
            print("Identifying companies from first and last pages of the document...")
            self.company_identifier = CompanyIdentifier(
                llm_client_wrapper=self.llm_company_id,
                llm_provider=self.llm_provider,
                pdf_processor=self.pdf_processor,
                azure_deployment_name=self.deployment_name if self.llm_provider == "azure" else None, 
                pages_to_analyze=3
            )
            self.companies_info = self.company_identifier.identify_companies(project_name)

            print(f"\nIdentified target company: {self.companies_info['target_company']['name']}")
            print(f"Target company description: {self.companies_info['target_company']['description']}")
            print(f"\nIdentified advisory firms:")
            for firm in self.companies_info['advisory_firms']:
                print(f"- {firm['name']} ({firm['role']})")
            print(f"\nProject codename: {self.companies_info['project_codename']}")
            print("\nCompany identification complete. Proceeding with knowledge graph extraction...\n")
            
        except FileNotFoundError:
             print(f"Could not identify companies as PDF file was not found at {self.pdf_path}. Skipping company identification.")
             self.companies_info = {
                "target_company": {"name": "Unknown (PDF not found)", "description": ""},
                "advisory_firms": [],
                "project_codename": "Unknown (PDF not found)"
            }
        except Exception as e:
            print(f"Error during company identification: {e}. Using default company info.")
            self.companies_info = {
                "target_company": {"name": "Unknown (Error in identification)", "description": ""},
                "advisory_firms": [],
                "project_codename": "Unknown (Error in identification)"
            }

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

        target_company = self.companies_info["target_company"]["name"]
        advisory_firms = [firm["name"] for firm in self.companies_info["advisory_firms"]]
        project_codename = self.companies_info["project_codename"]

        prompt = f"""
        You are a financial information extraction expert.
        Your task is to extract an extensive and structured knowledge graph from the financial text provided.
        
        This document concerns:
        - Target Company: {target_company} (the main company being analyzed/offered)
        - Project Codename: {project_codename} (this is just a codename, not the actual company)
        - Advisory Firms: {', '.join(advisory_firms) if advisory_firms else "None listed"} (these firms prepared the document but are not the subject)
        
        When extracting entities, MAKE SURE to clearly distinguish between:
        1. The TARGET COMPANY ({target_company}) - this is the main subject of the document
        2. The ADVISORY FIRMS ({', '.join(advisory_firms) if advisory_firms else "None listed"}) - these firms prepared the document
        3. The PROJECT CODENAME ({project_codename}) - this is just a codename, not a real company
        
        The knowledge graph should include entities, relationships, and attributes based on the provided ontology.
        """

        if self.construction_mode == "iterative" and previous_graph:
            prompt += f"""
            Use the previous knowledge graph context to inform your extraction.
            
            Previous knowledge graph context (from previous pages):
            {previous_graph_json}
            """

        prompt += f"""
        The ontology is as follows:

        {ontology_desc}

        ###FORMAT ###
        Output a JSON object like:
        {{
        "entities": [
            {{"id": "e1", "type": "pekg:Company", "name": "{target_company}"}},
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
        - IMPORTANT: Always clearly distinguish between the target company ({target_company}), advisory firms, and the project codename.
        - Do not confuse the project codename ({project_codename}) with the actual target company ({target_company}).
        """

        if self.construction_mode == "iterative" and previous_graph:
            prompt += """
            - If entities from previous pages appear again, use the same IDs for consistency.
            """

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
        prompt_string = self.build_prompt_for_text_analysis(text, previous_graph)
        content = ""

        try:
            if self.llm_provider == "azure":
                messages = [
                    {"role": "system", "content": "You are a financial information extraction assistant."},
                    {"role": "user", "content": prompt_string}
                ]
                response = self.llm.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=0.1
                )
                content = response.choices[0].message.content.strip()
            elif self.llm_provider == "vertexai": 
                generation_config_dict = {
                    "temperature": 0.1,
                    "max_output_tokens": 60000,
                    "response_mime_type": "application/json" 
                }
                response = self.llm.client.generate_content(
                    contents=[prompt_string],
                    generation_config=generation_config_dict
                )
                content = response.text.strip() 

            if content.startswith("```json"):
                content = content.lstrip("```json").rstrip("```").strip()
            return json.loads(content)

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response from LLM ({self.llm_provider}): {e}")
            print(f"Raw LLM Content that failed parsing:\n---\n{content}\n---")
            return {"entities": [], "relationships": []}
        except Exception as e:
            print(f"Error during LLM text analysis ({self.llm_provider}): {e}")
            print(f"Raw LLM Content (if available):\n---\n{content}\n---")
            return {"entities": [], "relationships": []}


    # ---------- PARALLEL PROCESSING METHODS ----------


    def _create_llm_client(self):
        """
        Create a new LLM client instance. 
        Used to create separate clients for parallel processing.
        
        Returns:
            AzureOpenAIClient: A new instance of the LLM client.
            VertexAIClient: A new instance of the LLM client.
        """
        if self.llm_provider == "azure":
            return AzureOpenAIClient(model_name=self.model_name)
        elif self.llm_provider == "vertexai":
            return VertexAIClient(model_name=self.model_name) 
        else:
            raise ValueError(f"Unsupported llm_provider in _create_llm_client: {self.llm_provider}")

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
        local_llm_client_wrapper = self._create_llm_client()
        sdk_client = local_llm_client_wrapper.client 

        print(f"Worker (Provider: {self.llm_provider}) processing page {page_num + 1}...")
        result_graph = {"entities": [], "relationships": []}

        try:
            if self.extraction_mode == "text":
                prompt = self.build_prompt_for_text_analysis(page_data["text"])
                content = ""
                if self.llm_provider == "azure":
                    messages = [
                        {"role": "system", "content": "You are a financial information extraction assistant."},
                        {"role": "user", "content": prompt}
                    ]
                    response = sdk_client.chat.completions.create(
                        model=self.deployment_name,
                        messages=messages,
                        temperature=0.1
                    )
                    content = response.choices[0].message.content.strip()
                elif self.llm_provider == "vertexai":
                    generation_config_dict = {
                        "temperature": 0.1,
                        "max_output_tokens": 60000,
                        "response_mime_type": "application/json"
                    }
                    response = sdk_client.generate_content(
                        contents=[prompt],
                        generation_config=generation_config_dict
                    )
                    content = response.text.strip()
                
                if content.startswith("```json"):
                    content = content.lstrip("```json").rstrip("```").strip()
                result_graph = json.loads(content)

            else: # Multimodal extraction
                target_company_name = self.companies_info["target_company"]["name"]
                
                visual_analysis_prompt_text = f"""
                You are a financial document analysis expert.
                This is a financial document concerning the company {target_company_name}.
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
                image_base64 = page_data['image_base64']
                visual_analysis_content = ""

                if self.llm_provider == "azure":
                    user_content_visual_analysis = [
                        {"type": "text", "text": visual_analysis_prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                    messages_visual_analysis = [
                        {"role": "system", "content": "You are a financial document analysis assistant capable of processing images."},
                        {"role": "user", "content": user_content_visual_analysis}
                    ]
                    response_visual_analysis = sdk_client.chat.completions.create(
                        model=self.deployment_name,
                        messages=messages_visual_analysis,
                        temperature=0.3,
                        max_tokens=4096
                    )
                    visual_analysis_content = response_visual_analysis.choices[0].message.content.strip()
                elif self.llm_provider == "vertexai":
                    image_bytes = base64.b64decode(image_base64)
                    vertex_parts_visual_analysis = [
                        {'text': visual_analysis_prompt_text},
                        {'inline_data': {'mime_type': 'image/png', 'data': image_bytes}}
                    ]
                    generation_config_dict_va = {
                        "temperature": 0.3,
                        "max_output_tokens": 4096
                    }
                    response_visual_analysis = sdk_client.generate_content(
                        contents=vertex_parts_visual_analysis,
                        generation_config=generation_config_dict_va
                    )
                    visual_analysis_content = response_visual_analysis.text.strip()
                
                visual_analysis_result = {
                    "page_num": page_data["page_num"],
                    "analysis": visual_analysis_content
                }

                ontology_desc = self.ontology.format_for_prompt()
                visual_data_prompt_text = f"""
                You are a financial data extraction expert.  
                Your task is to extract an extensive and structured knowledge graph from the financial text provided.
                This is a financial document concerning the company {target_company_name}.
                The knowledge graph should include entities, relationships, and attributes respecting the provided ontology.
                
                The ontology we're using is:
                {ontology_desc}

                Based on your previous analysis:
                {visual_analysis_result["analysis"]}

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

                Respond with *only* valid JSON. No commentary, Markdown, or explanations.
                """
                visual_data_content = ""
                if self.llm_provider == "azure":
                    user_content_visual_data = [
                        {"type": "text", "text": visual_data_prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                    messages_visual_data = [
                        {"role": "system", "content": "You are a financial data extraction assistant specializing in visual elements."},
                        {"role": "user", "content": user_content_visual_data}
                    ]
                    response_visual_data = sdk_client.chat.completions.create(
                        model=self.deployment_name,
                        messages=messages_visual_data,
                        temperature=0.1
                    )
                    visual_data_content = response_visual_data.choices[0].message.content.strip()
                elif self.llm_provider == "vertexai":
                    image_bytes_data = base64.b64decode(image_base64)
                    vertex_parts_visual_data = [
                        {'text': visual_data_prompt_text},
                        {'inline_data': {'mime_type': 'image/png', 'data': image_bytes_data}}
                    ]
                    generation_config_dict_vd = {
                        "temperature": 0.1,
                        "max_output_tokens": 60000,
                        "response_mime_type": "application/json"
                    }
                    response_visual_data = sdk_client.generate_content(
                        contents=vertex_parts_visual_data,
                        generation_config=generation_config_dict_vd
                    )
                    visual_data_content = response_visual_data.text.strip()

                if visual_data_content.startswith("```json"):
                    visual_data_content = visual_data_content.lstrip("```json").rstrip("```").strip()
                visual_kg = json.loads(visual_data_content) if visual_data_content else {"visual_elements": []}

                text_prompt = self.build_prompt_for_text_analysis(page_data["text"])
                text_content_kg = ""
                if self.llm_provider == "azure":
                    messages_text_kg = [
                        {"role": "system", "content": "You are a financial information extraction assistant."},
                        {"role": "user", "content": text_prompt}
                    ]
                    response_text_kg = sdk_client.chat.completions.create(
                        model=self.deployment_name,
                        messages=messages_text_kg,
                        temperature=0.1
                    )
                    text_content_kg = response_text_kg.choices[0].message.content.strip()
                elif self.llm_provider == "vertexai":
                    generation_config_dict_text = {
                        "temperature": 0.1,
                        "max_output_tokens": 60000,
                        "response_mime_type": "application/json"
                    }
                    response_text_kg = sdk_client.generate_content(
                        contents=[text_prompt],
                        generation_config=generation_config_dict_text
                    )
                    text_content_kg = response_text_kg.text.strip()
                
                if text_content_kg.startswith("```json"):
                    text_content_kg = text_content_kg.lstrip("```json").rstrip("```").strip()
                text_kg = json.loads(text_content_kg) if text_content_kg else {"entities": [], "relationships": []}
                
                graphs_to_merge = []
                if "visual_elements" in visual_kg:
                    for element in visual_kg["visual_elements"]:
                        graphs_to_merge.append({
                            "entities": element.get("entities", []),
                            "relationships": element.get("relationships", [])
                        })
                graphs_to_merge.append(text_kg)
                result_graph = merge_multiple_knowledge_graphs(graphs_to_merge)

            print(f"Worker completed page {page_num + 1}")
            return {
                "page_num": page_num, 
                "graph": result_graph}

        except json.JSONDecodeError as e:
            # failing_content_snippet = "Content too long or unavailable for snippet" # Not used
            print(f"Error parsing JSON in worker for page {page_num + 1} ({self.llm_provider}): {e}")
            return {"page_num": page_num, "graph": {"entities": [], "relationships": []}}
        except Exception as e:
            print(f"Error in worker processing page {page_num + 1} ({self.llm_provider}): {type(e).__name__} - {e}")
            return {"page_num": page_num, "graph": {"entities": [], "relationships": []}}


    def _build_knowledge_graph_parallel(self, dump: bool = False) -> Dict:
        """
        Build a knowledge graph from a PDF file by processing pages in parallel,
        but merging them in sequential order to maintain document flow.
        
        Args:
            dump (bool, optional): Flag to indicate if intermediate knowledge subgraphs should be saved.
            
        Returns:
            Dict: The final merged knowledge graph.
        """
        print(f"Starting parallel processing with {self.max_workers} workers (Provider: {self.llm_provider})")
        
        if not self.pdf_path.exists():
            print(f"Error: PDF file {self.pdf_path} not found. Cannot build knowledge graph in parallel.")
            return {"entities": [], "relationships": []}
            
        doc = pymupdf.open(self.pdf_path)
        num_pages = len(doc)
        doc.close()

        page_inputs = []
        for page_num_loop in range(num_pages): 
            print(f"Preparing data for page {page_num_loop + 1}...")
            if self.extraction_mode == "multimodal":
                page_data_content = self.pdf_processor.extract_page_from_pdf(page_num_loop)
            else:
                text = self.pdf_processor.extract_page_text(page_num_loop)
                page_data_content = {"page_num": page_num_loop, "text": text, "image_base64": None} 
            
            page_inputs.append({"page_num": page_num_loop, "page_data": page_data_content})
        
        print(f"Prepared data for {num_pages} pages, starting parallel processing...")
        merged_kg = {"entities": [], "relationships": []}

        from threading import Lock
        merge_lock = Lock()
        completed_pages = {}
        next_page_to_merge = 0
        
        def merge_pages_in_order():
            nonlocal next_page_to_merge, merged_kg 

            while next_page_to_merge in completed_pages:
                page_graph_to_merge = completed_pages.pop(next_page_to_merge) 
                entity_count_before = len(merged_kg["entities"])
                rel_count_before = len(merged_kg["relationships"])
                merged_kg = merge_knowledge_graphs(merged_kg, page_graph_to_merge)
                entity_count_after = len(merged_kg["entities"])
                rel_count_after = len(merged_kg["relationships"])
                print(f"Merged page {next_page_to_merge + 1} in sequence - Added {entity_count_after - entity_count_before} entities and "
                    f"{rel_count_after - rel_count_before} relationships")
                if next_page_to_merge % 5 == 0: # Clean every 5 pages
                    merged_kg = clean_knowledge_graph(merged_kg)
                next_page_to_merge += 1
                print(f"Current graph has {len(merged_kg['entities'])} entities and {len(merged_kg['relationships'])} relationships")

        processed_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_page_parallel, page_input) for page_input in page_inputs]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    page_num_res = result["page_num"] 
                    page_graph_res = result["graph"]  
                    if dump and page_graph_res: 
                        self._save_page_graph(page_graph_res, page_num_res, self.extraction_mode, is_iterative=False)
                    
                    with merge_lock: 
                        print(f"Page {page_num_res + 1} processing completed, storing for ordered merging...")
                        completed_pages[page_num_res] = page_graph_res
                        processed_count += 1
                        merge_pages_in_order() 
                        print(f"Processed {processed_count}/{num_pages} pages - Waiting for page {next_page_to_merge + 1} to continue sequential merging")
                except Exception as e:
                    print(f"Error retrieving result from worker: {type(e).__name__} - {e}")

        with merge_lock: 
            if completed_pages: 
                print(f"Merging {len(completed_pages)} remaining out-of-order pages...")
                for page_num_rem in sorted(completed_pages.keys()): 
                    print(f"Merging remaining page {page_num_rem + 1}...")
                    merged_kg = merge_knowledge_graphs(merged_kg, completed_pages[page_num_rem])
                completed_pages.clear()

        print(f"Successfully processed {processed_count} pages out of {num_pages}")
        print("Performing final cleanup and normalization of the merged knowledge graph...")
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
        target_company = self.companies_info["target_company"]["name"]
        advisory_firms_names = [firm["name"] for firm in self.companies_info["advisory_firms"]]
        project_codename = self.companies_info["project_codename"]

        prompt_text = f"""
        You are a financial document analysis expert.
        This document concerns:
        - Target Company: {target_company} (the main company being analyzed/offered)
        - Project Codename: {project_codename} (this is just a codename, not the actual company)
        - Advisory Firms: {', '.join(advisory_firms_names) if advisory_firms_names else "None listed"}(these firms prepared the document but are not the subject)

        Analyze this page from a financial document and identify all visual elements:
        1. Tables
        2. Charts/Graphs
        3. Diagrams
        5. Organizational charts
        6. Flow charts
        7. Financial statements
        8. Logos

        For each identified element:
        - Describe what the element represents
        - Describe the key information presented
        - IMPORTANT: Clearly indicate if the visual relates to the target company ({target_company}) or an advisory firm
        
        This is page {page_data["page_num"]} of the document.
        """
        if self.construction_mode == "iterative" and previous_graph:
            previous_graph_json = json.dumps(previous_graph) if previous_graph else "{}"
            prompt_text += f"\nPrevious knowledge graph context:\n{previous_graph_json}"

        image_base64 = page_data['image_base64']
        visual_analysis_str = ""

        try:
            if self.llm_provider == "azure":
                user_content = [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
                messages = [
                    {"role": "system", "content": "You are a financial document analysis assistant capable of processing images."},
                    {"role": "user", "content": user_content}
                ]
                response = self.llm.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=4096
                )
                visual_analysis_str = response.choices[0].message.content.strip()
            elif self.llm_provider == "vertexai":
                image_bytes = base64.b64decode(image_base64)
                vertex_parts = [
                    {'text': prompt_text},
                    {'inline_data': {'mime_type': 'image/png', 'data': image_bytes}}
                ]
                generation_config_dict = {
                    "temperature": 0.3,
                    "max_output_tokens": 4096
                }
                response = self.llm.client.generate_content(
                    contents=vertex_parts,
                    generation_config=generation_config_dict
                )
                visual_analysis_str = response.text.strip()
            
            if visual_analysis_str.startswith("```json"): # Though expecting text, good to sanitize
                visual_analysis_str = visual_analysis_str.lstrip("```json").rstrip("```").strip()
            return {"page_num": page_data["page_num"], "analysis": visual_analysis_str}

        except Exception as e:
            print(f"Error identifying visual elements on page {page_data['page_num']} ({self.llm_provider}): {type(e).__name__} - {e}")
            return {"page_num": page_data["page_num"], "analysis": "Error in visual analysis."}


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
        target_company = self.companies_info["target_company"]["name"]
        advisory_firms_names = [firm["name"] for firm in self.companies_info["advisory_firms"]]
        project_codename = self.companies_info["project_codename"]

        prompt_text = f"""
        You are a financial data extraction expert.  
        Your task is to extract an extensive and structured knowledge graph from the financial text provided.
        
        This document concerns:
        - Target Company: {target_company} (the main company being analyzed/offered)
        - Project Codename: {project_codename} (this is just a codename, not the actual company)
        - Advisory Firms: {', '.join(advisory_firms_names)} (these firms prepared the document but are not the subject)
        
        When extracting entities, MAKE SURE to clearly distinguish between:
        1. The TARGET COMPANY ({target_company}) - this is the main subject of the document
        2. The ADVISORY FIRMS ({', '.join(advisory_firms_names)}) - these firms prepared the document
        3. The PROJECT CODENAME ({project_codename}) - this is just a codename, not a real company
        
        The knowledge graph should include entities, relationships, and attributes respecting the provided ontology.
        
        The ontology we're using is:
        
        {ontology_desc}
        
        Based on your previous analysis:
        {visual_analysis["analysis"]}
        """
        if self.construction_mode == "iterative" and previous_graph:
            previous_graph_json = json.dumps(previous_graph) if previous_graph else "{}"
            prompt_text += f"\nPrevious knowledge graph context (use for consistency):\n{previous_graph_json}"
        
        prompt_text += """
        For each visual element (table, chart, etc.):
        1. Extract relevant entities, relationships, attributes matching ontology.
        2. Tables: extract structured data.
        3. Charts/Graphs: identify trends, key values, relationships.
        4. Diagrams/Flowcharts: identify entities and relationships.
        5. Logos: associate with company/brand.
        Format response as JSON: {"visual_elements": [{"element_type": "...", "description": "...", "entities": [...], "relationships": [...], "raw_data": {}}]}
         ### INSTRUCTIONS ###
        - Pay particular attention to numerical values, dates, and monetary amounts.
        - If the same entity appears under slightly different names (e.g. "DECK" vs. "DESK"), assume they refer to the same entity and normalize to the most frequent or contextually correct name.
        - Use your understanding of context to correct obvious typos.
        - Resolve entity mentions that refer to the same company/person/etc., and merge them into a single entity.
        - IMPORTANT: Always clearly distinguish between the target company, advisory firms, and the project codename.
        Respond with *only* valid JSON. No commentary, Markdown, or explanations.
        """
        image_base64 = page_data['image_base64']
        content = ""

        try:
            if self.llm_provider == "azure":
                user_content = [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
                messages = [
                    {"role": "system", "content": "You are a financial data extraction assistant specializing in visual elements."},
                    {"role": "user", "content": user_content}
                ]
                response = self.llm.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=0.1
                )
                content = response.choices[0].message.content.strip()
            elif self.llm_provider == "vertexai":
                image_bytes = base64.b64decode(image_base64)
                vertex_parts = [
                    {'text': prompt_text},
                    {'inline_data': {'mime_type': 'image/png', 'data': image_bytes}}
                ]
                generation_config_dict = {
                    "temperature": 0.1,
                    "max_output_tokens": 60000,
                    "response_mime_type": "application/json"
                }
                response = self.llm.client.generate_content(
                    contents=vertex_parts,
                    generation_config=generation_config_dict
                )
                content = response.text.strip()

            if content.startswith("```json"):
                content = content.lstrip("```json").rstrip("```").strip()
            return json.loads(content)

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from visual data extraction on page {page_data.get('page_num', 'N/A')} ({self.llm_provider}): {e}")
            print(f"Raw LLM Content that failed parsing:\n---\n{content}\n---")
            return {"visual_elements": []}
        except Exception as e:
            print(f"Error extracting data from visuals on page {page_data.get('page_num', 'N/A')} ({self.llm_provider}): {type(e).__name__} - {e}")
            print(f"Raw LLM Content (if available):\n---\n{content}\n---")
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
        page_kg = {"entities": [], "relationships": []}
        if not page_data.get("image_base64") and self.extraction_mode == "multimodal":
            print(f"Warning: image_base64 not found in page_data for page {page_data.get('page_num')}, but multimodal extraction was requested. Falling back to text-only for this page.")
            page_kg = self.analyze_text_with_llm(page_data["text"], previous_graph)
            return page_kg

        if self.extraction_mode == "multimodal":
            visual_analysis = self.identify_visual_elements(page_data, previous_graph)
            visual_kg_data = self.extract_data_from_visuals(page_data, visual_analysis, previous_graph)
            text_kg_data = self.analyze_text_with_llm(page_data["text"], previous_graph)

            graphs_to_merge = []
            if visual_kg_data and "visual_elements" in visual_kg_data: 
                for element in visual_kg_data.get("visual_elements", []): 
                    graphs_to_merge.append({
                        "entities": element.get("entities", []),
                        "relationships": element.get("relationships", [])
                    })
            if text_kg_data: 
                graphs_to_merge.append(text_kg_data)
            
            if graphs_to_merge:
                page_kg = merge_multiple_knowledge_graphs(graphs_to_merge)
            else: 
                 page_kg = {"entities": [], "relationships": []}

        else: 
            page_kg = self.analyze_text_with_llm(page_data["text"], previous_graph)
        
        return page_kg

    # ---------- MAIN KNOWLEDGE GRAPH BUILDING METHODS ----------

    def build_knowledge_graph_from_pdf(self, dump: bool = False) -> Dict:
        """
        Build a knowledge graph from a PDF file based on the configured extraction and construction modes.
        
        Args:
            dump (bool, optional): Flag to indicate if intermediate knowledge subgraphs should be saved.
        
        Returns:
            dict: The final merged knowledge graph.
        """
        print(f"Building knowledge graph with LLM provider: {self.llm_provider}, Model: {self.model_name}")
        if not self.pdf_path.exists():
            print(f"Error: PDF file {self.pdf_path} not found. Cannot build knowledge graph.")
            return {"entities": [], "relationships": []}

        merged_graph = {"entities": [], "relationships": []}
        if self.construction_mode == "parallel":
            merged_graph = self._build_knowledge_graph_parallel(dump)
        elif self.extraction_mode == "text":
            if self.construction_mode == "onego":
                text = self.pdf_processor.extract_text()
                if text: 
                    merged_graph = self.analyze_text_with_llm(text)
                else:
                    print("Warning: No text extracted from PDF for onego text mode.")
            else: # iterative text mode
                pages_text = self.pdf_processor.extract_text_as_list()
                for i, page_text in enumerate(pages_text):
                    print(f"Processing page {i+1} (text-only, iterative)...")
                    if page_text.strip():
                        page_graph = self.analyze_text_with_llm(page_text, merged_graph)
                        merged_graph = merge_knowledge_graphs(merged_graph, page_graph)
                        if dump:
                            self._save_page_graph(page_graph, i, "text", is_iterative=True)
                    else:
                        print(f"Skipping page {i+1} as it has no text content.")
        else: # multimodal extraction mode
            if self.construction_mode == "iterative":
                merged_graph = self._build_multimodal_knowledge_graph_iterative(dump)
            else: # onego multimodal mode
                merged_graph = self._build_multimodal_knowledge_graph_onego(dump)
        
        print("Performing final graph cleanup and normalization...")
        merged_graph = normalize_entity_ids(clean_knowledge_graph(merged_graph))
        
        if len(merged_graph.get("entities", [])) > 20: # Arbitrary threshold for consolidation
            print("Consolidating larger graph...")
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
        if not self.pdf_path.exists(): 
            print(f"Error: PDF file {self.pdf_path} not found for iterative multimodal build.")
            return {"entities": [], "relationships": []}
        doc = pymupdf.open(self.pdf_path)
        num_pages = len(doc)
        doc.close()
        
        merged_graph = {"entities": [], "relationships": []}
        for page_num in range(num_pages):
            print(f"Processing page {page_num + 1} of {num_pages} (multimodal, iterative)...")
            page_data = self.pdf_processor.extract_page_from_pdf(page_num)
            if not page_data["text"].strip() and not page_data["image_base64"]:
                print(f"Skipping page {page_num +1} due to no text or image content.")
                continue

            page_only_graph = self.analyze_page(page_data, merged_graph)
            
            if dump:
                self._save_page_graph(page_only_graph, page_num, "multimodal", is_iterative=True)
            
            merged_graph = merge_knowledge_graphs(merged_graph, page_only_graph)
            merged_graph = clean_knowledge_graph(merged_graph) # Clean after each merge
            
            print(f"Completed page {page_num + 1}/{num_pages}. Graph: {len(merged_graph['entities'])} entities, {len(merged_graph['relationships'])} relationships")
        return merged_graph 

    def _build_multimodal_knowledge_graph_onego(self, dump: bool = False) -> Dict:
        if not self.pdf_path.exists(): 
            print(f"Error: PDF file {self.pdf_path} not found for onego multimodal build.")
            return {"entities": [], "relationships": []}
        doc = pymupdf.open(self.pdf_path)
        num_pages = len(doc)
        doc.close()
        
        page_kgs = []
        for page_num in range(num_pages):
            print(f"Processing page {page_num + 1} of {num_pages} (multimodal, onego)...")
            page_data = self.pdf_processor.extract_page_from_pdf(page_num)
            if not page_data["text"].strip() and not page_data["image_base64"]:
                print(f"Skipping page {page_num +1} due to no text or image content.")
                continue
            page_kg = self.analyze_page(page_data, previous_graph=None) # No previous graph context for onego
            if dump:
                self._save_page_graph(page_kg, page_num, "multimodal", is_iterative=False)
            page_kgs.append(page_kg)
        
        print("Merging all page knowledge graphs (multimodal, onego)...")
        merged_kg = merge_multiple_knowledge_graphs(page_kgs) if page_kgs else {"entities": [], "relationships": []}
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
        if not graph or (not graph.get("entities") and not graph.get("relationships")):
            print(f"Skipping save for page {page_num + 1}, graph is empty.")
            return

        page_viz_graph = normalize_entity_ids(clean_knowledge_graph(graph))
        # Use self.json_output_path for consistency, create 'pages' subdirectory within it
        page_output_dir = self.json_output_path / "pages"
        page_output_dir.mkdir(parents=True, exist_ok=True)
        
        construction_suffix = "iterative" if is_iterative else "onego"
        provider_suffix = self.llm_provider
        
        base_filename = f"{mode}_kg_pg{page_num + 1}_{self.model_name.replace('/', '_')}_{provider_suffix}_{construction_suffix}"
        
        json_output_file = page_output_dir / f"{base_filename}.json"
        with open(json_output_file, "w") as f:
            json.dump(page_viz_graph, f, indent=2)
        
        html_output_file = str(page_output_dir / f"{base_filename}.html")
        try:
            if page_viz_graph.get("entities"): 
                 self.vizualizer.export_interactive_html(page_viz_graph, html_output_file)
                 print(f"KG visualization for page {page_num + 1} saved to {html_output_file}")
            else:
                print(f"Skipping HTML visualization for page {page_num + 1}, no entities to visualize.")
        except Exception as e:
            print(f"Could not save HTML visualization for page {page_num + 1}: {e}")


    def consolidate_knowledge_graph(self, kg: Dict) -> Dict:
        """
        Consolidate and clean up the knowledge graph using deterministic functions.
        This version does NOT use an LLM. It applies cleaning and ID normalization.
            
        Args:
            kg (Dict): The raw knowledge graph to consolidate.
            
        Returns:
            Dict: The consolidated and cleaned knowledge graph.
        """
        print("Consolidating knowledge graph (no LLM): applying cleaning and ID normalization...")
            
        # Step 1: Clean the graph (e.g., remove orphaned relationships, deduplicate relationships)
        # Ensure clean_knowledge_graph is robust enough for your needs.
        # It's assumed clean_knowledge_graph is imported from utils.kg_utils
        cleaned_kg = clean_knowledge_graph(kg)
            
        # Step 2: Normalize entity IDs (e.g., to e1, e2, ...) 
        # and update relationship references.
        # It's assumed normalize_entity_ids is imported from utils.kg_utils
        normalized_kg = normalize_entity_ids(cleaned_kg)
            
        print("Knowledge graph consolidation (no LLM) complete.")
        return normalized_kg

    def save_knowledge_graph(self, data: dict):
        """Saves the final knowledge graph and its visualization."""
        # self.json_output_path is already defined in __init__ and ensures the directory exists
        
        filename_prefix = "multimodal" if self.extraction_mode == "multimodal" else "text"
        provider_suffix = self.llm_provider
        construction_details = self.construction_mode
        if self.construction_mode == "parallel":
            construction_details += f"_{self.max_workers}w"
            
        # Use self.model_name directly, replace '/' for filename safety
        model_name_sanitized = self.model_name.replace("/", "_")
        base_filename = f"{filename_prefix}_kg_{self.project_name}_{model_name_sanitized}_{provider_suffix}_{construction_details}"

        json_output_file = self.json_output_path / f"{base_filename}.json"
        with open(json_output_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Knowledge graph saved to {json_output_file}")
        
        html_output_file = str(self.json_output_path / f"{base_filename}.html")
        try:
            if data.get("entities"):
                self.vizualizer.export_interactive_html(data, html_output_file)
                print(f"Knowledge graph visualization saved to {html_output_file}")
            else:
                print(f"Skipping final HTML visualization, no entities to visualize.")
        except Exception as e:
            print(f"Could not save final HTML visualization: {e}")
