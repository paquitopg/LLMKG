
import os
import json
import base64
import pymupdf # PyMuPDF, for PDF processing
from io import BytesIO
from PIL import Image # Pillow, for image manipulation (if needed)
import concurrent.futures
from typing import List, Dict, Tuple, Optional

# Assuming these are custom local modules
from llm_client import AzureOpenAIClient, VertexAIClient
from KG_visualizer import KnowledgeGraphVisualizer
from company_identifier import CompanyIdentifier 
from dotenv import load_dotenv
from pathlib import Path
from ontology.loader import PEKGOntology # Custom ontology loader
from utils.pdf_utils import PDFProcessor # Custom PDF utilities
from utils.kg_utils import (
    merge_knowledge_graphs, merge_multiple_knowledge_graphs,
    clean_knowledge_graph, normalize_entity_ids
)

load_dotenv()
class FinancialKGBuilder:
    """
    A unified class to build financial knowledge graphs from PDF documents using 
    Azure OpenAI or Google Gemini.
    Reverted to detailed prompts and removed explicit output token limits,
    relying on the model's context window.
    Keeps direct multimodal approach.
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
        ontology_path: str = str(Path(__file__).resolve().parent / "ontology" / "pekg_ontology3.yaml")
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
            self.llm_company_id = VertexAIClient(model_name="gemini-2.5-flash-preview-05-20")
        else:
            raise ValueError(f"Unsupported llm_provider: {self.llm_provider}. Choose 'azure' or 'vertexai'.")
        self.ontology_path = ontology_path
        self.ontology = PEKGOntology(ontology_path)

        pdf_dir = Path(__file__).resolve().parents[3] / "pages" / project_name
        pdf_dir.mkdir(parents=True, exist_ok=True) 
        self.pdf_path = pdf_dir / f"Project_{project_name}_Teaser.pdf"
        
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
        
        self.companies_info = {
            "target_company": {"name": self.project_name, "description": "Target company (defaulted to project name)"},
            "advisory_firms": [],
            "project_codename": self.project_name
        }
        try:
            print("Identifying companies from first and last pages of the document...")

            self.company_identifier = CompanyIdentifier(
                llm_client_wrapper=self.llm_company_id, 
                llm_provider=self.llm_provider, 
                pdf_processor=self.pdf_processor,
                azure_deployment_name= self.deployment_name if self.llm_provider == "azure" else None
            )

            identified_info = self.company_identifier.identify_companies(self.project_name) 

            if identified_info and identified_info.get("target_company") and \
               identified_info["target_company"]["name"] and \
               identified_info["target_company"]["name"].lower() not in ["unknown", self.project_name.lower(), "unknown (error in identification)"]:
                self.companies_info = identified_info
                if not self.companies_info.get("project_codename"): # Ensure project_codename is set
                    self.companies_info["project_codename"] = self.project_name
            else:
                print(f"Company identification returned default/error or project name, using project name '{self.project_name}' as target company and codename.")
                self.companies_info["target_company"]["name"] = self.project_name
                if self.companies_info["target_company"].get("description") is None or self.companies_info["target_company"]["name"] == "Unknown (Error in identification)":
                    self.companies_info["target_company"]["description"] = f"Target company (defaulted to project name: {self.project_name})"
                self.companies_info["project_codename"] = self.project_name


            print(f"\nIdentified target company: {self.companies_info['target_company']['name']}")
            print(f"Target company description: {self.companies_info['target_company']['description']}")
            print(f"\nIdentified advisory firms:")
            if self.companies_info.get('advisory_firms'):
                for firm in self.companies_info['advisory_firms']:
                    print(f"- {firm['name']} ({firm['role']})")
            else:
                print("  None identified.")
            print(f"\nProject codename: {self.companies_info['project_codename']}")
            print("\nCompany identification complete. Proceeding with knowledge graph extraction...\n")
            
        except FileNotFoundError:
             print(f"Could not identify companies as PDF file was not found at {self.pdf_path}. Skipping company identification, using project name as fallback.")
        except Exception as e:
            print(f"Error during company identification: {type(e).__name__} - {e}. Using project name as fallback for company info.")

    # REMOVED: _truncate_previous_graph_for_gemini
    # REMOVED: _summarize_ontology_for_gemini

    def build_prompt_for_text_analysis(self, text: str, previous_graph: Dict = None) -> str:
        """
        Build the detailed prompt for text analysis (style from merged_kg_builder_firstversion.py).
        """
        ontology_desc = self.ontology.format_for_prompt() # Full ontology
        
        previous_graph_json_for_prompt = "{}" # Full previous_graph, no truncation
        if self.construction_mode == "iterative" and previous_graph and previous_graph.get("entities"):
            previous_graph_json_for_prompt = json.dumps(previous_graph)

        target_company_name = self.companies_info["target_company"]["name"]
        advisory_firms_names = [firm["name"] for firm in self.companies_info.get("advisory_firms", [])]
        project_codename_val = self.companies_info["project_codename"]

        # This prompt structure is based on the user's 'merged_kg_builder_firstversion.py' style
        prompt = f"""
        You are a financial information extraction expert.
        Your task is to extract an extensive and structured knowledge graph from the financial text provided.
        
        This document concerns:
        - Target Company: {target_company_name} (the main company being analyzed/offered)
        - Project Codename: {project_codename_val} (this is just a codename, not the actual company)
        - Advisory Firms: {', '.join(advisory_firms_names) if advisory_firms_names else "None listed"} (these firms prepared the document but are not the subject)
        
        When extracting entities, MAKE SURE to clearly distinguish between:
        1. The TARGET COMPANY ({target_company_name}) - this is the main subject of the document
        2. The ADVISORY FIRMS ({', '.join(advisory_firms_names) if advisory_firms_names else "None listed"}) - these firms prepared the document
        3. The PROJECT CODENAME ({project_codename_val}) - this is just a codename, not a real company
        
        The knowledge graph should include entities, relationships, and attributes based on the provided ontology.
        Your PRIMARY TASK is to extract entities and relationships *ONLY* from the "Current Page Text to Analyze" provided below.
        """

        if self.construction_mode == "iterative" and previous_graph and previous_graph.get("entities"):
            prompt += f"""
            Use the previous knowledge graph context to inform your extraction for ID consistency and linking NEW information from the CURRENT page.
            **CRITICAL: DO NOT re-extract or re-list any entities or relationships from this "Previous knowledge graph context" section in your output, UNLESS that same information is also explicitly present in the "Current Page Text to Analyze". Your output for this page must solely reflect the content of the CURRENT PAGE.**
            
            Previous knowledge graph context (from previous pages):
            {previous_graph_json_for_prompt}
            """
        else:
             prompt += "\nNo previous graph context provided or it was empty.\n"

        prompt += f"""
        The ontology is as follows:
        {ontology_desc}

        Output Format Example (Attributes are direct entity properties, NOT nested under "properties"):
        {{
        "entities": [
            {{"id": "e1", "type": "pekg:Company", "name": "{target_company_name}"}},
            {{"id": "e2", "type": "pekg:FinancialMetric", "name": "FY23 Revenue", "metricValue": 15000000, "metricCurrency": "USD", "year": 2023}}
        ],
        "relationships": [
            {{"source": "e1", "target": "e2", "type": "pekg:reportsMetric"}}
        ]
        }}
        
        Current Page Text to Analyze:
        \"\"\"{text}\"\"\"

        Extraction Instructions for CURRENT PAGE TEXT:
        1.  From the "Current Page Text to Analyze" *only*, identify all entities MATCHING THE ONTOLOGY.
        2.  For these entities, extract all relevant attributes specified in the ontology, based *only* on the "Current Page Text to Analyze". Attributes should be direct properties of the entity object (e.g., "name": "X", "value": Y), NOT nested under a "properties" key.
        3.  Identify and create relationships between entities, based *only* on information in the "Current Page Text to Analyze".
        4.  Ensure every entity extracted from the "Current Page Text to Analyze" is connected if the current text provides relational context. Avoid disconnected nodes if linkage is present in the *current page's text*.
        5.  Pay particular attention to numerical values, dates, currencies, and units found in the "Current Page Text to Analyze".
        6.  Ensure the JSON is valid.

        Respond with *ONLY* a single, valid JSON object representing the knowledge graph extracted *solely from the Current Page Text to Analyze*. Do not include any commentary. Do not include Markdown syntax (no ```json). Do not include explanations.
        """
        return prompt

    def analyze_text_with_llm(self, text: str, previous_graph: Dict = None, page_num_for_logging: str = "N/A") -> Dict:
        prompt_string = self.build_prompt_for_text_analysis(text, previous_graph)
        content = "" 

        try:
            if self.llm_provider == "azure":
                messages = [
                    {"role": "system", "content": "You are a financial information extraction assistant designed to output JSON."},
                    {"role": "user", "content": prompt_string}
                ]
                response = self.llm.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=0.1
                    # max_tokens parameter removed
                )
                content = response.choices[0].message.content.strip()
            elif self.llm_provider == "vertexai": 
                generation_config_dict = {
                    "temperature": 0.1,
                    # "max_output_tokens" parameter removed
                    "response_mime_type": "application/json" 
                }
                response = self.llm.client.generate_content(
                    contents=[prompt_string], 
                    generation_config=generation_config_dict
                )
                if not response.candidates:
                    prompt_feedback = response.prompt_feedback if hasattr(response, 'prompt_feedback') else None
                    block_reason = prompt_feedback.block_reason if prompt_feedback and hasattr(prompt_feedback, 'block_reason') else 'N/A'
                    print(f"Warning: Gemini response for text analysis (page {page_num_for_logging}) has no candidates. Finish reason from prompt feedback (if any): {block_reason}")
                    return {"entities": [], "relationships": []}
                
                candidate = response.candidates[0]
                finish_reason_name = candidate.finish_reason.name if hasattr(candidate.finish_reason, 'name') else str(candidate.finish_reason)

                if finish_reason_name != "STOP": 
                    print(f"Warning: Gemini call for text analysis (page {page_num_for_logging}) finished with reason '{finish_reason_name}'. Response may be empty or incomplete.")
                    if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                        for rating in candidate.safety_ratings:
                            print(f"  Safety Rating: {rating.category.name} - {rating.probability.name}")
                    if finish_reason_name == "SAFETY":
                         print("  Response blocked due to SAFETY. Returning empty graph.")
                         return {"entities": [], "relationships": []}
                    if finish_reason_name == "MAX_TOKENS":
                         print("  Response stopped due to MAX_TOKENS. Input prompt might be too long or output too verbose. Returning empty/partial graph.")
                         # Content might be partial here, attempt to use if available for debugging, but expect JSON load to fail.
                
                if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                    content = candidate.content.parts[0].text.strip()
                else:
                    print(f"Warning: Gemini response for text analysis (page {page_num_for_logging}) has no content parts. Finish Reason: {finish_reason_name}")
                    content = ""


            if content.startswith("```json"):
                content = content.lstrip("```json").rstrip("```").strip()
            
            if not content: 
                print(f"Warning: Empty content received from LLM for text analysis (page {page_num_for_logging}). Returning empty graph.")
                return {"entities": [], "relationships": []}
            
            return json.loads(content)

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response from LLM ({self.llm_provider}) for text analysis (page {page_num_for_logging}): {e}")
            print(f"Model: {self.model_name}, Deployment (Azure): {self.deployment_name}")
            print(f"Raw LLM Content that failed parsing (first 500 chars):\n---\n{content[:500]}\n---")
            return {"entities": [], "relationships": []} 
        except Exception as e:
            if isinstance(e, ValueError) and "response.text" in str(e) and "valid Part" in str(e) and self.llm_provider == "vertexai":
                 print(f"ValueError accessing response.text for text analysis (page {page_num_for_logging}), likely due to blocked content or no parts. Error: {e}")
                 return {"entities": [], "relationships": []}
            print(f"Error during LLM text analysis ({self.llm_provider}, page {page_num_for_logging}): {type(e).__name__} - {e}")
            print(f"Model: {self.model_name}, Deployment (Azure): {self.deployment_name}")
            print(f"Raw LLM Content (if available, first 500 chars):\n---\n{content[:500]}\n---")
            return {"entities": [], "relationships": []} 

    def _create_llm_client(self):
        # This function now uses the specific llm_company_id or main llm based on context
        # However, _process_page_parallel should use the main llm instance configured for the builder.
        if self.llm_provider == "azure":
            return AzureOpenAIClient(model_name=self.model_name)
        elif self.llm_provider == "vertexai":
            return VertexAIClient(model_name=self.model_name) 
        else:
            raise ValueError(f"Unsupported llm_provider in _create_llm_client: {self.llm_provider}")

    def _process_page_parallel(self, page_info: Dict) -> Dict:
        page_num = page_info["page_num"]
        page_data = page_info["page_data"]
        # Workers should use the main model specified for the KGBuilder instance
        # The self.llm is already initialized with the correct provider and model_name.
        # Creating new clients here might be redundant if self.llm itself can be used by threads,
        # but new instances ensure thread isolation if clients are not thread-safe.
        local_llm_client_wrapper = self._create_llm_client() 
        sdk_client = local_llm_client_wrapper.client 
        page_num_display = page_num + 1 # For logging (1-indexed)
        result_graph = {"entities": [], "relationships": []}
        content = "" 
        print(f"Worker processing page {page_num_display}...") # Added print statement

        try:
            if self.extraction_mode == "text":
                # For parallel processing, previous_graph is None as pages are processed independently before merging.
                prompt = self.build_prompt_for_text_analysis(page_data["text"], previous_graph=None) 
                
                if self.llm_provider == "azure":
                    messages = [
                        {"role": "system", "content": "You are a financial information extraction assistant designed to output JSON."},
                        {"role": "user", "content": prompt}
                    ]
                    response = sdk_client.chat.completions.create(
                        model=self.deployment_name, 
                        messages=messages,
                        temperature=0.1
                        # max_tokens removed
                    )
                    content = response.choices[0].message.content.strip()
                elif self.llm_provider == "vertexai":
                    generation_config_dict = {
                        "temperature": 0.1,
                        # "max_output_tokens" removed
                        "response_mime_type": "application/json"
                    }
                    response = sdk_client.generate_content(
                        contents=[prompt],
                        generation_config=generation_config_dict
                    )
                    if not response.candidates:
                        prompt_feedback = response.prompt_feedback if hasattr(response, 'prompt_feedback') else None
                        block_reason = prompt_feedback.block_reason if prompt_feedback and hasattr(prompt_feedback, 'block_reason') else 'N/A'
                        print(f"Warning: Gemini response for parallel text (page {page_num_display}) has no candidates. Prompt Feedback: {block_reason}")
                        return {"page_num": page_num, "graph": {"entities": [], "relationships": []}} # Return structure for parallel
                    candidate = response.candidates[0]
                    finish_reason_name = candidate.finish_reason.name if hasattr(candidate.finish_reason, 'name') else str(candidate.finish_reason)
                    if finish_reason_name != "STOP":
                        print(f"Warning: Gemini call for parallel text (page {page_num_display}) finished with reason '{finish_reason_name}'.")
                        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                            for rating in candidate.safety_ratings: print(f"  Safety Rating: {rating.category.name} - {rating.probability.name}")
                        if finish_reason_name == "SAFETY": return {"page_num": page_num, "graph": {"entities": [], "relationships": []}}
                        if finish_reason_name == "MAX_TOKENS": print(f"  Response for parallel text (page {page_num_display}) stopped due to MAX_TOKENS.")
                    if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts: content = candidate.content.parts[0].text.strip()
                    else: content = ""
                
                if content.startswith("```json"):
                    content = content.lstrip("```json").rstrip("```").strip()
                if not content: result_graph = {"entities": [], "relationships": []}
                else: result_graph = json.loads(content)

            else: # Multimodal extraction in parallel
                target_company_name = self.companies_info["target_company"]["name"]
                project_codename_val = self.companies_info["project_codename"]
                advisory_firms_names = [firm["name"] for firm in self.companies_info.get("advisory_firms", [])]
                
                ontology_desc = self.ontology.format_for_prompt() # Full ontology

                # Using the detailed direct multimodal prompt structure.
                # For parallel processing, previous_graph context is not used here.
                # It's applied during the sequential merge if needed or in iterative mode.
                direct_multimodal_prompt = f"""
                You are an expert financial information extraction system.
                Analyze page {page_num_display} (text & image) from a financial document concerning Target Company: "{target_company_name}".
                Project Codename: {project_codename_val}
                Advisory Firms: {', '.join(advisory_firms_names) if advisory_firms_names else "None listed"}

                Your PRIMARY TASK is to extract entities and relationships *ONLY* from the content of THIS CURRENT PAGE (both its text and visual elements).
                DO NOT use any information from previous pages or external knowledge. Focus solely on the provided image and text for this page.

                Ontology for Knowledge Graph Extraction:
                {ontology_desc} 
                
                Current Page Text to Analyze (use in conjunction with the image):
                \"\"\"{page_data["text"]}\"\"\"

                Task for CURRENT PAGE {page_num_display} (Image and Text):
                Extract a comprehensive knowledge graph (entities and relationships) from BOTH the "Current Page Text" AND any visual elements (tables, charts, diagrams) in the accompanying image.
                1. From the "Current Page Text" and Image *only*, identify all entities matching the ontology.
                2. For these entities, extract all relevant attributes. Attributes should be direct properties of the entity object (e.g., "name": "X"), NOT nested under a "properties" key.
                3. Identify and create relationships between entities, based *only* on information in the "Current Page Text" and Image.
                    - **Crucially, prioritize creating meaningful relationships between extracted entities.** Connect entities to the Target Company "{target_company_name}" and to each other as supported by the page content. Graphs with disconnected entities are not useful.
                4. Assign new unique IDs (e.g., "e1") for new entities from *current page*.
                5. Be accurate and comprehensive. Ensure the JSON is valid.

                Output Format Example (Attributes are direct entity properties, relationships are critical):
                {{
                  "entities": [
                    {{"id": "e1", "type": "pekg:Company", "name": "{target_company_name}"}},
                    {{"id": "e2", "type": "pekg:FinancialMetric", "name": "FY23 Revenue", "metricValue": 15000000, "metricCurrency": "USD"}},
                    {{"id": "e3", "type": "pekg:Product", "name": "VisionMax Suite"}}
                  ],
                  "relationships": [
                    {{"source": "e1", "target": "e2", "type": "pekg:reportsMetric"}},
                    {{"source": "e1", "target": "e3", "type": "pekg:developsProduct"}}
                  ]
                }}
                Output ONLY a single, valid JSON object. No commentary or markdown.
                """
                content = "" 
                image_base64 = page_data['image_base64']
                if self.llm_provider == "azure":
                    user_content = [
                        {"type": "text", "text": direct_multimodal_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                    messages = [
                        {"role": "system", "content": "You are a financial KG extraction assistant for multimodal inputs, outputting JSON."},
                        {"role": "user", "content": user_content}
                    ]
                    response = sdk_client.chat.completions.create(
                        model=self.deployment_name, 
                        messages=messages,
                        temperature=0.1
                        # max_tokens removed
                    )
                    content = response.choices[0].message.content.strip()
                elif self.llm_provider == "vertexai":
                    image_bytes = base64.b64decode(image_base64)
                    vertex_parts = [
                        {'text': direct_multimodal_prompt},
                        {'inline_data': {'mime_type': 'image/png', 'data': image_bytes}}
                    ]
                    generation_config_dict = {
                        "temperature": 0.1,
                        # "max_output_tokens" removed
                        "response_mime_type": "application/json"
                    }
                    response = sdk_client.generate_content(
                        contents=[{'parts': vertex_parts}], 
                        generation_config=generation_config_dict
                    )
                    if not response.candidates:
                        prompt_feedback = response.prompt_feedback if hasattr(response, 'prompt_feedback') else None
                        block_reason = prompt_feedback.block_reason if prompt_feedback and hasattr(prompt_feedback, 'block_reason') else 'N/A'
                        print(f"Warning: Gemini response for parallel multimodal (page {page_num_display}) has no candidates. Prompt Feedback: {block_reason}")
                        return {"page_num": page_num, "graph": {"entities": [], "relationships": []}}
                    candidate = response.candidates[0]
                    finish_reason_name = candidate.finish_reason.name if hasattr(candidate.finish_reason, 'name') else str(candidate.finish_reason)
                    if finish_reason_name != "STOP":
                        print(f"Warning: Gemini call for parallel multimodal (page {page_num_display}) finished with reason '{finish_reason_name}'.")
                        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                            for rating in candidate.safety_ratings: print(f"  Safety Rating: {rating.category.name} - {rating.probability.name}")
                        if finish_reason_name == "SAFETY": return {"page_num": page_num, "graph": {"entities": [], "relationships": []}}
                        if finish_reason_name == "MAX_TOKENS": print(f"  Response for parallel multimodal (page {page_num_display}) stopped due to MAX_TOKENS.")
                    if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts: content = candidate.content.parts[0].text.strip()
                    else: content = ""


                if content.startswith("```json"):
                    content = content.lstrip("```json").rstrip("```").strip()
                if not content: result_graph = {"entities": [], "relationships": []}
                else: result_graph = json.loads(content)
            
            print(f"Worker completed page {page_num_display}. Entities: {len(result_graph.get('entities',[]))}, Relationships: {len(result_graph.get('relationships',[]))}")
            return {"page_num": page_num, "graph": result_graph}

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in worker for page {page_num_display} ({self.llm_provider}, Model: {self.model_name}): {e}")
            print(f"Raw LLM Content that failed parsing in worker (first 500 chars):\n---\n{content[:500]}\n---") 
            return {"page_num": page_num, "graph": {"entities": [], "relationships": []}}
        except Exception as e:
            if isinstance(e, ValueError) and "response.text" in str(e) and "valid Part" in str(e) and self.llm_provider == "vertexai":
                 print(f"ValueError accessing response.text in worker for page {page_num_display}, likely due to blocked content or no parts. Error: {e}")
                 return {"page_num": page_num, "graph": {"entities": [], "relationships": []}}
            print(f"Error in worker processing page {page_num_display} ({self.llm_provider}, Model: {self.model_name}): {type(e).__name__} - {e}")
            print(f"Raw LLM Content (if available, first 500 chars):\n---\n{content[:500]}\n---")
            return {"page_num": page_num, "graph": {"entities": [], "relationships": []}}

    def _build_knowledge_graph_parallel(self, dump: bool = False) -> Dict:
        print(f"Starting parallel processing with {self.max_workers} workers (Provider: {self.llm_provider}, Model: {self.model_name})")
        
        if not self.pdf_path.exists():
            print(f"Error: PDF file {self.pdf_path} not found. Cannot build knowledge graph in parallel.")
            return {"entities": [], "relationships": []}
            
        doc = pymupdf.open(self.pdf_path)
        num_pages = len(doc)
        doc.close()
        print(f"PDF has {num_pages} pages.")

        page_inputs = []
        print("Preparing data for all pages...")
        for page_num_loop in range(num_pages): 
            if self.extraction_mode == "multimodal":
                page_data_content = self.pdf_processor.extract_page_from_pdf(page_num_loop)
            else: 
                text = self.pdf_processor.extract_page_text(page_num_loop)
                page_data_content = {"page_num": page_num_loop, "text": text, "image_base64": None} 
            
            page_inputs.append({"page_num": page_num_loop, "page_data": page_data_content})
        print(f"Prepared data for {len(page_inputs)} pages. Starting parallel processing...")
        
        merged_kg = {"entities": [], "relationships": []}

        from threading import Lock
        merge_lock = Lock() 
        completed_pages = {} 
        next_page_to_merge = 0 
        processed_count = 0
        
        # This function assumes the merge_lock is ALREADY HELD by the caller.
        def merge_pages_in_order_unsafe(): # Renamed to indicate it needs external lock
            nonlocal next_page_to_merge, merged_kg 
            # The lock is acquired by the caller in the as_completed loop
            while next_page_to_merge in completed_pages: 
                page_graph_to_merge = completed_pages.pop(next_page_to_merge) 
                
                entity_count_before = len(merged_kg.get("entities", []))
                rel_count_before = len(merged_kg.get("relationships", []))
                
                merged_kg = merge_knowledge_graphs(merged_kg, page_graph_to_merge)
                
                entity_count_after = len(merged_kg.get("entities", []))
                rel_count_after = len(merged_kg.get("relationships", []))

                print(f"Merged page {next_page_to_merge + 1} in sequence. Added {entity_count_after - entity_count_before} entities, {rel_count_after - rel_count_before} relationships.")
                print(f"  Current graph: {entity_count_after} entities, {rel_count_after} relationships.")
                
                # Periodic cleaning, similar to merged_kg_builder.py
                # Cleans after merging page 0, page 5, page 10, etc. (0-indexed)
                if next_page_to_merge % 5 == 0: 
                    print(f"Cleaning graph after merging page {next_page_to_merge + 1}...")
                    merged_kg = clean_knowledge_graph(merged_kg)
                    print(f"  Graph after cleaning: {len(merged_kg.get('entities',[]))} entities, {len(merged_kg.get('relationships',[]))} relationships.")
                
                next_page_to_merge += 1
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_page_parallel, page_input): page_input["page_num"] for page_input in page_inputs}
            
            for future in concurrent.futures.as_completed(futures):
                original_page_num = futures[future] 
                try:
                    result = future.result() 
                    page_num_res = result["page_num"] 
                    page_graph_res = result["graph"]  
                    
                    if dump and page_graph_res and (page_graph_res.get("entities") or page_graph_res.get("relationships")): 
                        self._save_page_graph(page_graph_res, page_num_res, self.extraction_mode, is_iterative=False) 
                    
                    with merge_lock: 
                        print(f"Page {page_num_res + 1} processing completed, storing for ordered merging.")
                        completed_pages[page_num_res] = page_graph_res
                        processed_count += 1
                        merge_pages_in_order_unsafe() # Call the unsafe version as lock is held
                        print(f"Processed {processed_count}/{num_pages} pages. Next page to merge sequentially: {next_page_to_merge + 1}.")
                        
                except Exception as e:
                    print(f"Error retrieving result from worker for page {original_page_num + 1}: {type(e).__name__} - {e}")
                    # Store an empty graph for failed pages to allow sequential merging to continue if possible
                    with merge_lock:
                        completed_pages[original_page_num] = {"entities": [], "relationships": []}
                        processed_count +=1 
                        merge_pages_in_order_unsafe() # Call the unsafe version
                        print(f"Processed {processed_count}/{num_pages} (page {original_page_num+1} failed). Next to merge: {next_page_to_merge + 1}.")

        # Final merge of any remaining pages that might not have been merged in strict order
        # (e.g., if some pages arrived out of order and were skipped by merge_pages_in_order_unsafe)
        with merge_lock: 
            print("Attempting final merge of any remaining pages...")
            merge_pages_in_order_unsafe() # One last attempt to merge in sequence
            if completed_pages: 
                print(f"Warning: {len(completed_pages)} page(s) still in 'completed_pages' after sequential merging. Merging them now out of strict order if necessary.")
                for page_num_rem in sorted(completed_pages.keys()): 
                    print(f"Merging remaining page {page_num_rem + 1} from fallback...")
                    page_graph_to_merge = completed_pages.pop(page_num_rem)
                    merged_kg = merge_knowledge_graphs(merged_kg, page_graph_to_merge)
                completed_pages.clear() # Should be empty now

        print(f"Successfully processed (or attempted to process and handle errors for) {processed_count} pages out of {num_pages}.")
        print("Performing final graph cleanup and normalization...")
        merged_kg = normalize_entity_ids(clean_knowledge_graph(merged_kg))
        return merged_kg


    def analyze_page(self, page_data: Dict, previous_graph: Dict = None) -> Dict:
        """
        Perform comprehensive analysis of a single page.
        Uses a direct multimodal prompt for KG extraction if mode is multimodal.
        Reverted to detailed prompts and removed explicit output token limits.
        """
        page_kg = {"entities": [], "relationships": []}
        page_num_0_indexed = page_data.get('page_num', -1) 
        page_num_display = page_num_0_indexed + 1 if page_num_0_indexed != -1 else 'N/A'

        if self.extraction_mode == "multimodal":
            if not page_data.get("image_base64"):
                print(f"Warning: image_base64 not found for page {page_num_display} in multimodal mode. Falling back to text-only.")
                if page_data.get("text","").strip():
                    page_kg = self.analyze_text_with_llm(page_data["text"], previous_graph, page_num_for_logging=page_num_display) 
                return page_kg 
            
            # Full previous_graph_json for all providers now
            previous_graph_json_for_prompt = "{}"
            if self.construction_mode == "iterative" and previous_graph and previous_graph.get("entities"):
                previous_graph_json_for_prompt = json.dumps(previous_graph) # No Gemini-specific truncation
            
            ontology_desc = self.ontology.format_for_prompt() # Full ontology for all
            target_company_name = self.companies_info["target_company"]["name"]
            advisory_firms_names = [firm["name"] for firm in self.companies_info.get("advisory_firms", [])]
            project_codename_val = self.companies_info["project_codename"]

            # Detailed Direct Multimodal Prompt (style of merged_kg_builder_firstversion.py's text prompt)
            direct_multimodal_prompt = f"""
            You are an expert financial information extraction system.
            Your task is to extract an extensive and structured knowledge graph from the provided page, considering both its text and visual elements (image).
            
            This document concerns:
            - Target Company: {target_company_name} (the main company being analyzed/offered)
            - Project Codename: {project_codename_val} (this is just a codename, not the actual company)
            - Advisory Firms: {', '.join(advisory_firms_names) if advisory_firms_names else "None listed"} (these firms prepared the document but are not the subject)
            
            When extracting entities, MAKE SURE to clearly distinguish between the Target Company, Advisory Firms, and the Project Codename.
            Your PRIMARY TASK is to extract entities and relationships *ONLY* from the content of the CURRENT PAGE (both its text and visual elements).

            The ontology for Knowledge Graph Extraction is as follows:
            {ontology_desc}
            """
            if self.construction_mode == "iterative" and previous_graph and previous_graph.get("entities"):
                direct_multimodal_prompt += f"""
            Use the previous knowledge graph context to inform your extraction for ID consistency and linking NEW information from the CURRENT page.
            **CRITICAL: DO NOT re-extract or re-list any entities or relationships from this "Previous knowledge graph context" section in your output, UNLESS that same information is also explicitly present in the "Current Page Text" or "Current Page Image". Your output for this page must solely reflect the content of the CURRENT PAGE.**
            
            Previous knowledge graph context (from previous pages):
            {previous_graph_json_for_prompt}
            """
            else: # Ensure this block is part of the main prompt string
                direct_multimodal_prompt += "\nNo previous graph context provided or it was empty.\n"

            direct_multimodal_prompt += f"""
            Current Page Text to Analyze (use in conjunction with the image):
            \"\"\"{page_data["text"]}\"\"\"

            Task for CURRENT PAGE {page_num_display} (Image and Text):
            Extract a comprehensive knowledge graph (entities and relationships) from BOTH the "Current Page Text" AND any visual elements (tables, charts, diagrams) in the accompanying image.
            1.  From the "Current Page Text" and Image *only*, identify all entities matching the ontology.
            2.  For these entities, extract all relevant attributes. Attributes should be direct properties of the entity object (e.g., "name": "X"), NOT nested under a "properties" key.
            3.  Identify and create relationships between entities, based *only* on information in the "Current Page Text" and Image.
                - **Crucially, prioritize creating meaningful relationships between extracted entities.** Connect entities to the Target Company "{target_company_name}" and to each other as supported by the page content. Graphs with disconnected entities are not useful.
            4.  Assign new unique IDs (e.g., "e1") for new entities from *current page*. Reuse IDs from "Previous Pages Context" if an entity is re-identified *on the current page*.
            5.  Be accurate and comprehensive. Ensure the JSON is valid.

            Output Format Example (Attributes are direct entity properties, relationships are critical):
            {{
              "entities": [
                {{"id": "e1", "type": "pekg:Company", "name": "{target_company_name}"}},
                {{"id": "e2", "type": "pekg:FinancialMetric", "name": "FY23 Revenue", "metricValue": 15000000, "metricCurrency": "USD"}},
                {{"id": "e3", "type": "pekg:Product", "name": "VisionMax Suite"}}
              ],
              "relationships": [
                {{"source": "e1", "target": "e2", "type": "pekg:reportsMetric"}},
                {{"source": "e1", "target": "e3", "type": "pekg:developsProduct"}}
              ]
            }}
            Output ONLY a single, valid JSON object. No commentary or markdown.
            """
            content = "" 
            try:
                image_base64 = page_data['image_base64']
                if self.llm_provider == "azure":
                    user_content = [
                        {"type": "text", "text": direct_multimodal_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                    messages = [
                        {"role": "system", "content": "You are a financial KG extraction assistant for multimodal inputs, outputting JSON."},
                        {"role": "user", "content": user_content}
                    ]
                    response = self.llm.client.chat.completions.create(
                        model=self.deployment_name, 
                        messages=messages,
                        temperature=0.1
                        # max_tokens removed
                    )
                    content = response.choices[0].message.content.strip()
                elif self.llm_provider == "vertexai":
                    image_bytes = base64.b64decode(image_base64)
                    vertex_parts = [
                        {'text': direct_multimodal_prompt},
                        {'inline_data': {'mime_type': 'image/png', 'data': image_bytes}}
                    ]
                    generation_config_dict = {
                        "temperature": 0.1,
                        # "max_output_tokens" removed
                        "response_mime_type": "application/json"
                    }
                    response = self.llm.client.generate_content(
                        contents=[{'parts': vertex_parts}], 
                        generation_config=generation_config_dict
                    )
                    
                    if not response.candidates:
                        prompt_feedback = response.prompt_feedback if hasattr(response, 'prompt_feedback') else None
                        block_reason = prompt_feedback.block_reason if prompt_feedback and hasattr(prompt_feedback, 'block_reason') else 'N/A'
                        print(f"Warning: Gemini response for page {page_num_display} has no candidates. Prompt Feedback: {block_reason}")
                        return {"entities": [], "relationships": []} 

                    candidate = response.candidates[0]
                    finish_reason_name = candidate.finish_reason.name if hasattr(candidate.finish_reason, 'name') else str(candidate.finish_reason)
                    
                    if finish_reason_name != "STOP":
                        print(f"Warning: Gemini call for page {page_num_display} finished with reason '{finish_reason_name}'. Response may be empty or incomplete.")
                        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings: 
                            for rating in candidate.safety_ratings:
                                print(f"  Safety Rating for page {page_num_display}: {rating.category.name} - {rating.probability.name}")
                        if finish_reason_name == "SAFETY":
                             print(f"  Response for page {page_num_display} was blocked due to SAFETY settings.")
                             return {"entities": [], "relationships": []} 
                        if finish_reason_name == "MAX_TOKENS":
                             print(f"  Response for page {page_num_display} stopped due to MAX_TOKENS. Input prompt may be too large or output too verbose.")
                    
                    if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                        content = candidate.content.parts[0].text.strip()
                    else: 
                        print(f"Warning: Gemini response for page {page_num_display} has no content parts. Finish Reason: {finish_reason_name}")
                        content = "" 

                if content.startswith("```json"):
                    content = content.lstrip("```json").rstrip("```").strip()
                
                if not content: 
                    print(f"Warning: Empty content received from LLM for page {page_num_display}. Returning empty graph.")
                    page_kg = {"entities": [], "relationships": []}
                else:
                    page_kg = json.loads(content)
            
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from direct multimodal analysis on page {page_num_display} ({self.llm_provider}, Model: {self.model_name}): {e}")
                print(f"Raw LLM Content that failed parsing (first 500 chars):\n---\n{content[:500]}\n---")
                page_kg = {"entities": [], "relationships": []}
            except ValueError as e: 
                if "response.text" in str(e) and "valid Part" in str(e) and self.llm_provider == "vertexai": 
                    print(f"ValueError accessing response.text for page {page_num_display}, likely due to blocked content or no parts. Error: {e}")
                    page_kg = {"entities": [], "relationships": []}
                else: 
                    raise 
            except Exception as e:
                print(f"Error during direct multimodal analysis on page {page_num_display} ({self.llm_provider}, Model: {self.model_name}): {type(e).__name__} - {e}")
                print(f"Raw LLM Content (if available, first 500 chars):\n---\n{content[:500]}\n---")
                page_kg = {"entities": [], "relationships": []}
        else: 
            if page_data.get("text","").strip():
                page_kg = self.analyze_text_with_llm(page_data["text"], previous_graph, page_num_for_logging=page_num_display)
            else: 
                print(f"Skipping text-only analysis for page {page_num_display} as it has no text content.")
                page_kg = {"entities": [], "relationships": []} 
        
        return page_kg

    def build_knowledge_graph_from_pdf(self, dump: bool = False) -> Dict:
        print(f"Building knowledge graph with LLM provider: {self.llm_provider}, Model: {self.model_name}, Project: {self.project_name}")
        print(f"Extraction Mode: {self.extraction_mode}, Construction Mode: {self.construction_mode}")

        if not self.pdf_path.exists():
            print(f"Error: PDF file {self.pdf_path} not found. Cannot build knowledge graph.")
            return {"entities": [], "relationships": []}

        merged_graph = {"entities": [], "relationships": []}

        if self.construction_mode == "parallel":
            merged_graph = self._build_knowledge_graph_parallel(dump)
        elif self.construction_mode == "onego":
            if self.extraction_mode == "text":
                print("Processing entire document text in 'onego' mode...")
                full_text = self.pdf_processor.extract_text()
                if full_text.strip(): 
                    merged_graph = self.analyze_text_with_llm(full_text, previous_graph=None, page_num_for_logging="all_pages_onego")
                else:
                    print("Warning: No text extracted from PDF for 'onego' text mode.")
            else: 
                print("Processing all pages for 'onego' multimodal mode (individual analysis then merge)...")
                merged_graph = self._build_multimodal_knowledge_graph_onego(dump)
        
        elif self.construction_mode == "iterative":
            if self.extraction_mode == "text":
                print("Processing PDF iteratively (text-only)...")
                pages_text = self.pdf_processor.extract_text_as_list()
                for i, page_text in enumerate(pages_text):
                    page_num_display = i + 1
                    print(f"Processing page {page_num_display}/{len(pages_text)} (text-only, iterative)...")
                    if page_text.strip():
                        page_graph = self.analyze_text_with_llm(page_text, merged_graph, page_num_for_logging=str(page_num_display))
                        if dump and page_graph.get("entities"): 
                            self._save_page_graph(page_graph, i, "text", is_iterative=True)
                        merged_graph = merge_knowledge_graphs(merged_graph, page_graph)
                        if page_num_display % 5 == 0: 
                             merged_graph = clean_knowledge_graph(merged_graph)
                    else:
                        print(f"Skipping page {page_num_display} as it has no text content.")
            else: 
                print("Processing PDF iteratively (multimodal)...")
                merged_graph = self._build_multimodal_knowledge_graph_iterative(dump)
        
        print("Performing final graph cleanup and normalization...")
        merged_graph = normalize_entity_ids(clean_knowledge_graph(merged_graph))
        
        if len(merged_graph.get("entities", [])) > 10: # Arbitrary threshold for consolidation
            # The consolidation via LLM might be very expensive and slow. 
            # Consider if this step is always necessary or if it should be optional.
            # For now, keeping it as it was.
            print(f"Consolidating final graph with {len(merged_graph.get('entities',[]))} entities...")
            merged_graph = self.consolidate_knowledge_graph(merged_graph) 
            
        print(f"Knowledge graph building process completed. Final graph: {len(merged_graph.get('entities',[]))} entities, {len(merged_graph.get('relationships',[]))} relationships.")
        return merged_graph


    def _build_multimodal_knowledge_graph_iterative(self, dump: bool = False) -> Dict:
        if not self.pdf_path.exists(): 
            print(f"Error: PDF file {self.pdf_path} not found for iterative multimodal build.")
            return {"entities": [], "relationships": []}
        
        doc = pymupdf.open(self.pdf_path)
        num_pages = len(doc)
        doc.close() 
        
        merged_graph = {"entities": [], "relationships": []}
        for page_num in range(num_pages): 
            current_page_num_display = page_num + 1
            print(f"Processing page {current_page_num_display}/{num_pages} (multimodal, iterative)...")
            
            page_data = self.pdf_processor.extract_page_from_pdf(page_num)
            
            if not page_data.get("text","").strip() and not page_data.get("image_base64"):
                print(f"Skipping page {current_page_num_display} due to no text or image content.")
                continue
            
            page_data_for_analysis = page_data.copy() 
            page_data_for_analysis['page_num'] = page_num 

            page_only_graph = self.analyze_page(page_data_for_analysis, previous_graph=merged_graph) 
            
            if dump and page_only_graph.get("entities"): 
                self._save_page_graph(page_only_graph, page_num, "multimodal", is_iterative=True)
            
            merged_graph = merge_knowledge_graphs(merged_graph, page_only_graph)
            if current_page_num_display % 5 == 0: 
                merged_graph = clean_knowledge_graph(merged_graph) 
            
            print(f"Completed page {current_page_num_display}/{num_pages}. Graph: {len(merged_graph.get('entities',[]))} entities, {len(merged_graph.get('relationships',[]))} relationships")
        
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
            current_page_num_display = page_num + 1
            print(f"Processing page {current_page_num_display}/{num_pages} (multimodal, onego - independent analysis)...")
            
            page_data = self.pdf_processor.extract_page_from_pdf(page_num)
            if not page_data.get("text","").strip() and not page_data.get("image_base64"):
                print(f"Skipping page {current_page_num_display} due to no text or image content.")
                continue
            
            page_data_for_analysis = page_data.copy()
            page_data_for_analysis['page_num'] = page_num 

            page_kg = self.analyze_page(page_data_for_analysis, previous_graph=None) 
            
            if dump and page_kg.get("entities"): 
                self._save_page_graph(page_kg, page_num, "multimodal", is_iterative=False) 
            
            if page_kg.get("entities") or page_kg.get("relationships"): # Ensure we only add non-empty graphs
                page_kgs.append(page_kg)
        
        print("Merging all page knowledge graphs (multimodal, onego)...")
        merged_kg = merge_multiple_knowledge_graphs(page_kgs) if page_kgs else {"entities": [], "relationships": []}
        return merged_kg

    def _save_page_graph(self, graph: Dict, page_num: int, mode: str, is_iterative: bool) -> None:
        if not graph or (not graph.get("entities") and not graph.get("relationships")): # Check both entities and relationships
            # print(f"Skipping save for page {page_num + 1} as graph is empty.") # Optional: uncomment for debugging
            return

        graph_to_save = {}
        try:
            if isinstance(graph, dict) and ("entities" in graph or "relationships" in graph): # Allow graphs with only relationships or only entities
                 graph_to_save = normalize_entity_ids(clean_knowledge_graph(graph.copy()))
            else:
                print(f"Warning: Graph for page {page_num + 1} is malformed before saving. Graph: {str(graph)[:200]}...") 
                graph_to_save = {"entities": [], "relationships": []} 

        except Exception as e:
            print(f"Error during pre-save cleaning/normalization for page {page_num + 1}: {e}")
            # Fallback to saving the original graph if cleaning fails, or an empty one if really problematic
            graph_to_save = graph.copy() if isinstance(graph, dict) else {"entities": [], "relationships": []}
        
        page_output_dir = self.json_output_path / "pages_kg" 
        page_output_dir.mkdir(parents=True, exist_ok=True)
        
        construction_suffix = self.construction_mode 
        if self.construction_mode == "parallel" and not is_iterative: 
             pass # Already "parallel"
        elif is_iterative: 
            construction_suffix = "iterative" # Overrides if it was parallel but called from an iterative context (should not happen for page dumps)
        else: # onego specific page
            construction_suffix = "onego_ind_page"


        model_name_sanitized = self.model_name.replace('/', '_').replace(':', '_')
        base_filename = f"{mode}_kg_pg{page_num + 1}_{model_name_sanitized}_{self.llm_provider}_{construction_suffix}"
        
        json_output_file = page_output_dir / f"{base_filename}.json"
        try:
            with open(json_output_file, "w") as f:
                json.dump(graph_to_save, f, indent=2)
        except Exception as e:
            print(f"Error saving JSON for page {page_num + 1}: {e}")
        
        html_output_file = str(page_output_dir / f"{base_filename}.html")
        try:
            # Ensure entities exist and have 'type' for visualization
            if graph_to_save.get("entities") and \
               isinstance(graph_to_save["entities"], list) and \
               (len(graph_to_save["entities"]) > 0 or graph_to_save.get("relationships")) and \
               all(isinstance(e, dict) and e.get('id') and e.get('type') for e in graph_to_save["entities"]): # Check type and id
                self.vizualizer.export_interactive_html(graph_to_save, html_output_file)
                # print(f"✅ Graph for page {page_num+1} saved to: {html_output_file}") # Less verbose
            elif graph_to_save.get("entities") or graph_to_save.get("relationships"): # If graph exists but not suitable for viz
                # print(f"Skipping HTML visualization for page {page_num + 1} as entities may be malformed or empty for visualization.")
                pass


        except KeyError as e: 
            print(f"Could not save HTML visualization for page {page_num + 1} due to missing key (likely 'type' or 'id' in entities): {e}")
        except Exception as e:
            print(f"Could not save HTML visualization for page {page_num + 1}: {type(e).__name__} - {e}")

    def consolidate_knowledge_graph(self, kg: Dict) -> Dict:
        cleaned_kg = clean_knowledge_graph(kg)
        normalized_kg = normalize_entity_ids(cleaned_kg)
        return normalized_kg

    def save_knowledge_graph(self, data: dict):
        if not data or (not data.get("entities") and not data.get("relationships")):
            print("Skipping save of final knowledge graph as it is empty or invalid.")
            return

        filename_prefix = self.extraction_mode 
        provider_suffix = self.llm_provider
        construction_details = self.construction_mode
        if self.construction_mode == "parallel":
            construction_details += f"_{self.max_workers}w" 
            
        model_name_sanitized = self.model_name.replace("/", "_").replace(":", "_")
        base_filename = f"{filename_prefix}_kg_{self.project_name}_{model_name_sanitized}_{provider_suffix}_{construction_details}"

        json_output_file = self.json_output_path / f"{base_filename}.json"
        try:
            with open(json_output_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Final knowledge graph saved to {json_output_file}")
        except Exception as e:
            print(f"Error saving final JSON knowledge graph: {e}")
        
        html_output_file = str(self.json_output_path / f"{base_filename}.html")
        try:
            if data.get("entities"): 
                self.vizualizer.export_interactive_html(data, html_output_file)
                print(f"Final knowledge graph visualization saved to {html_output_file}")
        except Exception as e:
            print(f"Could not save final HTML visualization: {e}")