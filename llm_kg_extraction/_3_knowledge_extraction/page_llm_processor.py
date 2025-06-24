import json
import base64
from typing import Dict, Any, Optional, List

# Assuming LLM wrappers and Ontology loader are structured as previously discussed
# Adjust these import paths based on your final project structure
from llm_integrations.base_llm_wrapper import BaseLLMWrapper
from llm_integrations.azure_llm import AzureLLM # For type checking if needed for specific params
from llm_integrations.vertex_llm import VertexLLM # For type checking

from ontology_management.ontology_loader import PEKGOntology # Assuming PEKGOntology or a base class

# Semantic chunking configuration
config = {
    "chunk_size": 4000,
    "min_chunk_size": 500, 
    "chunk_overlap": 200,
    "respect_sentence_boundaries": True,
    "detect_topic_shifts": True
}

class PageLLMProcessor:
    """
    Processes a single document page (text or multimodal) using an LLM
    to extract a knowledge graph.
    """

    def __init__(self,
                 llm_client: BaseLLMWrapper,
                 ontology: PEKGOntology, 
                 extraction_mode: str,
                 use_ontology: bool = True): 
        """
        Initializes the PageLLMProcessor.

        Args:
            llm_client (BaseLLMWrapper): An instance of an LLM client wrapper.
            ontology (PEKGOntology): An instance of the ontology loader.
        """
        self.llm_client = llm_client
        self.ontology = ontology
        self.temperature = 0.1
        self.extraction_mode = extraction_mode
        self.use_ontology = use_ontology

    def _build_text_extraction_prompt(self,
                                      page_text: str,
                                      document_context_info: Dict[str, Any],
                                      previous_graph_context: Optional[Dict[str, Any]] = None,
                                      construction_mode: str = "iterative" 
                                     ) -> str:
        """
        Builds the detailed prompt for text-based knowledge graph extraction.
        """
        if self.use_ontology:
            ontology_desc = self.ontology.format_for_prompt()
        
        else :
            ontology_desc = "No ontology is being used for this extraction. Entities and relationships will be extracted based on general relevance."
        
        previous_graph_json_for_prompt = "{}"
    
        if construction_mode == "iterative" and previous_graph_context and previous_graph_context.get("entities"):
            previous_graph_json_for_prompt = json.dumps(previous_graph_context)

        # Adapt prompt based on document_context_info
        # This example focuses on financial_teaser context.
        # For other doc types, this section would need to be more dynamic.
        doc_type = document_context_info.get("identified_document_type", "document")
        context_specific_intro = ""
        primary_task_focus = "extract entities and relationships *ONLY* from the \"Current Page Text to Analyze\" provided below."

        if doc_type == "financial_teaser":
    
            context_specific_intro = f"""
            This document is a '{doc_type}' concerning a main company, which sould not be confused with the advisory firms or project codename.
            The document can be summarized as follows:
            - Document Summary: {document_context_info.get("document_summary", "No summary provided")}
            - The main entity of focus for this document is understood to be '{document_context_info.get("primary_entity_name", "")}'.

            When extracting entities, MAKE SURE to clearly distinguish between:
            1. The TARGET COMPANY 
            2. The ADVISORY FIRMS
            3. The PROJECT CODENAME
            """

        else:
            main_entity_name = document_context_info.get("primary_entity_name") # Fallback
            context_specific_intro = f"""
            This document is identified as a '{doc_type}'.
            The primary entity of focus for this document is understood to be '{main_entity_name}'.
            The document can be summarized as follows:
            - Document Summary: {document_context_info.get("document_summary", "No summary provided")}

            Please analyze the content in relation to this primary entity and in light of the overall document context.
            """
            primary_task_focus = f"Your PRIMARY TASK is to extract entities and relationships relevant to a '{doc_type}' *ONLY* from the \"Current Page Text to Analyze\"."


        prompt = f"""
        You are an expert information extraction system.
        Your task is to extract an extensive and structured knowledge graph from the financial text provided.
        {context_specific_intro}
        Your PRIMARY TASK is to {primary_task_focus}
        """

        if construction_mode == "iterative" and previous_graph_context and previous_graph_context.get("entities"):
            prompt += f"""
            Use the previous knowledge graph context to inform your extraction for ID consistency and linking NEW information from the CURRENT page.
            **CRITICAL: DO NOT re-extract or re-list any entities or relationships from this "Previous knowledge graph context" section in your output, UNLESS that same information is also explicitly present in the "Current Page Text to Analyze". Your output for this page must solely reflect the content of the CURRENT PAGE.**

            Previous knowledge graph context (from previous pages):
            {previous_graph_json_for_prompt}
            """

        elif construction_mode != "onego": 
             prompt += "\nNo previous graph context provided or it was empty. Extract all information fresh from the current page.\n"

        if self.use_ontology: 
            prompt += f"""
            The ontology for Knowledge Graph Extraction is as follows:
            {ontology_desc}

            **CRITICAL: Ensure you extract entities and relationships respecting the ontology. Do not deviate from the specified types.**
        """
        prompt += f"""
        Output Format Example (Attributes are direct entity properties, NOT nested under "properties"):
        {{
        "entities": [
            {{"id": "e1", "type": "pekg:Company", "name": "ExampleCorp"}},
            {{"id": "e2", "type": "pekg:FinancialMetric", "name": "FY23 Revenue", "metricValue": 15000000, "metricCurrency": "USD", "year": 2023}}
        ],
        "relationships": [
            {{"source": "e1", "target": "e2", "type": "pekg:reportsMetric"}}
        ]
        }}

        Current Page Text to Analyze:
        \"\"\"{page_text}\"\"\"

        Extraction Instructions for CURRENT PAGE TEXT:
        """
        if self.use_ontology:
             prompt += f"""
             1.  From the "Current Page Text to Analyze" *only*, identify all entities MATCHING THE ONTOLOGY.
            **CRITICAL: Ensure you extract entities and relationships respecting the ontology structure. Do not deviate from the specified types**
            2.  For these entities, extract all relevant attributes specified in the ontology, based *only* on the "Current Page Text to Analyze". Attributes should be direct properties of the entity object (e.g., "name": "X", "value": Y), NOT nested under a "properties" key.
            """
        else:
            prompt += f"""
            1.  From the "Current Page Text to Analyze" *only*, identify all entities and relations you consider relevant.
            """
        prompt += f"""     
        3.  Identify and create relationships between entities, based *only* on information in the "Current Page Text to Analyze".
        4.  Ensure every entity extracted from the "Current Page Text to Analyze" is connected if the current text provides relational context. Avoid disconnected nodes if linkage is present in the *current page's text*.
        5.  Pay particular attention to numerical values, dates, currencies, and units found in the "Current Page Text to Analyze".
        6.  Ensure the JSON is valid.

        Respond with *ONLY* a single, valid JSON object representing the knowledge graph extracted *solely from the Current Page Text to Analyze*. Do not include any commentary. Do not include Markdown syntax (no ```json). Do not include explanations.
        """
        return prompt

    def _build_multimodal_extraction_prompt(self,
                                           page_text: str,
                                           page_number: str,
                                           document_context_info: Dict[str, Any],
                                           previous_graph_context: Optional[Dict[str, Any]] = None,
                                           construction_mode: str = "iterative"
                                          ) -> str:
        """
        Builds the detailed prompt for multimodal (text + image) KG extraction.
        """
        if self.use_ontology:
            ontology_desc = self.ontology.format_for_prompt()
        
        else:
            ontology_desc = "No ontology is being used for this extraction. Entities and relationships will be extracted based on general relevance."
            
        previous_graph_json_for_prompt = "{}"
    
        if construction_mode == "iterative" and previous_graph_context and previous_graph_context.get("entities"):
            previous_graph_json_for_prompt = json.dumps(previous_graph_context)

        # Adapt prompt based on document_context_info
        # This example focuses on financial_teaser context.
        # For other doc types, this section would need to be more dynamic.
        doc_type = document_context_info.get("identified_document_type", "document")
        context_specific_intro = ""
        primary_task_focus = "extract entities and relationships *ONLY* from the \"Current Page Text to Analyze\" provided below."

        if doc_type == "financial_teaser":
    
            context_specific_intro = f"""
            This document is a '{doc_type}' concerning a main company, which sould not be confused with the advisory firms or project codename.
            The document can be summarized as follows:
            - Document Summary: {document_context_info.get("document_summary", "No summary provided")}
            - The main entity of focus for this document is understood to be '{document_context_info.get("primary_entity_name", "")}'.

            When extracting entities, MAKE SURE to clearly distinguish between:
            1. The TARGET COMPANY 
            2. The ADVISORY FIRMS
            3. The PROJECT CODENAME
            """

        else:
            main_entity_name = document_context_info.get("primary_entity_name") # Fallback
            context_specific_intro = f"""
            This document is identified as a '{doc_type}'.
            The primary entity of focus for this document is understood to be '{main_entity_name}'.
            The document can be summarized as follows:
            - Document Summary: {document_context_info.get("document_summary", "No summary provided")}

            Please analyze the content in relation to this primary entity and in light of the overall document context.
            """
            primary_task_focus = f"Your PRIMARY TASK is to extract entities and relationships relevant to a '{doc_type}' *ONLY* from the \"Current Page Text to Analyze\"."

        prompt = f"""
        You are an expert financial information extraction system.
        Your task is to extract an extensive and structured knowledge graph from the provided page, considering both its text and visual elements (image).
        {context_specific_intro}
        Your PRIMARY TASK is to {primary_task_focus}
        DO NOT use any information from previous pages or external knowledge unless explicitly provided as "Previous knowledge graph context". Focus solely on the provided image and text for this current page.
        """
        if construction_mode == "iterative" and previous_graph_context and previous_graph_context.get("entities"):
            prompt += f"""
            Use the previous knowledge graph context to inform your extraction for ID consistency and linking NEW information from the CURRENT page.
            **CRITICAL: DO NOT re-extract or re-list any entities or relationships from this "Previous knowledge graph context" section in your output, UNLESS that same information is also explicitly present in the "Current Page Text" or "Current Page Image". Your output for this page must solely reflect the content of the CURRENT PAGE.**

            Previous knowledge graph context (from previous pages):
            {previous_graph_json_for_prompt}
            """
        elif construction_mode != "onego":
            prompt += "\nNo previous graph context provided or it was empty. Extract all information fresh from the current page.\n"

        if self.use_ontology:
            prompt += f"""
            The ontology for Knowledge Graph Extraction is as follows:
            {ontology_desc}

            **CRITICAL: Ensure you extract entities and relationships respecting the ontology. Do not deviate from the specified types.**
        """
            
        prompt += f"""

        Current Page Text to Analyze (use in conjunction with the image):
        \"\"\"{page_text}\"\"\"

        Task for CURRENT PAGE {page_number} (Image and Text):
        """
        if self.use_ontology:
             prompt += f"""
            1.  From the "Current Page Text" and Image *only*, identify all entities matching the ontology.
            """
        else:
            prompt += f"""
            1.  From the "Current Page Text" and Image *only*, identify all entities and relations you consider relevant.
            """
            
        prompt += f"""
        2.  For these entities, extract all relevant attributes. Attributes should be direct properties of the entity object (e.g., "name": "X"), NOT nested under a "properties" key.
        3.  Identify and create relationships between entities, based *only* on information in the "Current Page Text" and Image.
            - **Crucially, prioritize creating meaningful relationships between extracted entities.** Connect entities as supported by the page content. Graphs with disconnected entities are not useful.
        4.  Assign new unique IDs (e.g., "e1_pgX") for new entities from *this current page*. If reusing an entity from "Previous Pages Context" that is re-identified *on the current page*, use its existing ID.
        5.  Be accurate and comprehensive. Ensure the JSON is valid.

        Output Format Example (Attributes are direct entity properties, relationships are critical):
        {{
        "entities": [
            {{"id": "e1", "type": "pekg:Company", "name": "ExampleCorp"}},
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
        return prompt


    def process_page(self,
                     page_data: Dict[str, Any],
                     document_context_info: Dict[str, Any],
                     construction_mode: str = "iterative",
                     previous_graph_context: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, List[Any]]:
        """
        Method to process a single page to extract a knowledge graph.
        Handles both text and multimodal extraction based on self.extraction_mode.

        Args:
            page_data (Dict[str, Any]): Dict containing 'page_number', 'text' and optionally 'image_base64'.
            document_context_info (Dict[str, Any]): Contextual info about the document.
            construction_mode (str): How the overall KG is being built ("iterative", "parallel", "onego").
            previous_graph_context (Optional[Dict[str, Any]]): KG from previous pages.

        Returns:
            Dict[str, List[Any]]: The extracted knowledge graph for the page (entities and relationships).
                                  Returns {"entities": [], "relationships": []} on error or no content.
        """
        page_number = page_data.get("page_number", "N/A")
        page_text = page_data.get("text", "")
        page_image_base64 = page_data.get("image_base64")
        llm_response_content: Optional[str] = None
        
        # Validate input based on extraction mode
        if self.extraction_mode == "text":
            if not page_text.strip():
                print(f"Skipping page {page_number} for text analysis as it has no text content.")
                return {"entities": [], "relationships": []}
        elif self.extraction_mode == "multimodal":
            if not page_image_base64:
                print(f"Warning: image_base64 not found for page {page_number} in multimodal mode.")
                if not page_text.strip():
                    print(f"No text content either. Skipping page {page_number}.")
                    return {"entities": [], "relationships": []}
                print(f"Falling back to text-only processing for page {page_number}.")
                # Will process as text-only below
        
        try:
            # Determine processing approach
            use_multimodal = (self.extraction_mode == "multimodal" and page_image_base64)
            
            if use_multimodal:
                prompt_str = self._build_multimodal_extraction_prompt(
                    page_text=page_text,
                    page_number=page_number,
                    document_context_info=document_context_info,
                    previous_graph_context=previous_graph_context,
                    construction_mode=construction_mode
                )
                
                # Prepare multimodal input for LLM client
                if isinstance(self.llm_client, AzureLLM):
                    user_content_parts = [
                        {"type": "text", "text": prompt_str},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_image_base64}"}}
                    ]
                    messages = [
                        {"role": "system", "content": "You are a financial KG extraction assistant for multimodal inputs, outputting JSON."},
                        {"role": "user", "content": user_content_parts}
                    ]
                    llm_response_content = self.llm_client.chat_completion(
                        messages=messages,
                        temperature=self.temperature
                    )

                elif isinstance(self.llm_client, VertexLLM):
                    # Fix: Add type check to ensure page_image_base64 is not None
                    if page_image_base64 is not None:
                        image_bytes = base64.b64decode(page_image_base64)
                        vertex_parts = [
                            {'text': prompt_str},
                            {'inline_data': {'mime_type': 'image/png', 'data': image_bytes}}
                        ]
                        llm_response_content = self.llm_client.generate_content(
                            prompt=vertex_parts,
                            temperature=self.temperature,
                            response_mime_type="application/json"
                        )
                    else:
                        print(f"Warning: page_image_base64 is None for VertexLLM processing on page {page_number}")
                        use_multimodal = False
                else:
                    print(f"Multimodal extraction not configured for LLM type: {self.llm_client.__class__.__name__}. Falling back to text-only.")
                    use_multimodal = False
            
            # Text-only processing (either by design or fallback)
            if not use_multimodal:
                prompt_str = self._build_text_extraction_prompt(
                    page_text=page_text,
                    document_context_info=document_context_info,
                    previous_graph_context=previous_graph_context,
                    construction_mode=construction_mode
                )
                
                if isinstance(self.llm_client, AzureLLM):
                    messages = [
                        {"role": "system", "content": "You are a financial information extraction assistant designed to output JSON."},
                        {"role": "user", "content": prompt_str}
                    ]
                    llm_response_content = self.llm_client.chat_completion(
                        messages=messages,
                        temperature=self.temperature
                    )
                elif isinstance(self.llm_client, VertexLLM):
                    llm_response_content = self.llm_client.generate_content(
                        prompt=prompt_str,
                        temperature=self.temperature,
                        response_mime_type="application/json"
                    )
                else: # Generic fallback
                    messages = [
                        {"role": "system", "content": "You are an information extraction assistant designed to output JSON."},
                        {"role": "user", "content": prompt_str}
                    ]
                    llm_response_content = self.llm_client.chat_completion(messages=messages, temperature=self.temperature)

            # Parse response
            if not llm_response_content:
                print(f"Warning: Empty content received from LLM for page {page_number}. Returning empty graph.")
                return {"entities": [], "relationships": []}

            # Clean potential markdown ```json
            clean_content = llm_response_content.strip()
            if clean_content.startswith("```json"):
                clean_content = clean_content.lstrip("```json").rstrip("```").strip()
            elif clean_content.startswith("```"):
                 clean_content = clean_content.lstrip("```").rstrip("```").strip()
                 if clean_content.startswith("json"):
                    clean_content = clean_content.lstrip("json").strip()
            
            if not clean_content:
                print(f"Warning: Content became empty after stripping markdown for page {page_number}. Raw: '{llm_response_content[:100]}...'")
                return {"entities": [], "relationships": []}

            result = json.loads(clean_content)
            
            # Add page_number to result for tracking
            result["page_number"] = page_number

            return result

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response from LLM ({self.llm_client.__class__.__name__}) for page {page_number}: {e}")
            print(f"Model: {self.llm_client.model_name}")
            raw_content_snippet = llm_response_content[:500] if llm_response_content else "N/A"
            print(f"Raw LLM Content that failed parsing (first 500 chars):\n---\n{raw_content_snippet}\n---")
            return {"entities": [], "relationships": []}
        except Exception as e:
            if "response.text" in str(e) and "valid Part" in str(e) and isinstance(self.llm_client, VertexLLM):
                 print(f"ValueError accessing response.text for page {page_number} (Vertex), likely due to blocked content or no parts. Error: {e}")
                 return {"entities": [], "relationships": []}
            print(f"Error during LLM page processing ({self.llm_client.__class__.__name__}, page {page_number}): {type(e).__name__} - {e}")
            print(f"Model: {self.llm_client.model_name}")
            raw_content_snippet = llm_response_content[:500] if llm_response_content else "N/A"
            print(f"Raw LLM Content (if available, first 500 chars):\n---\n{raw_content_snippet}\n---")
            return {"entities": [], "relationships": []}
