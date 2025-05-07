import os
import json
import base64
import pymupdf
from io import BytesIO
from PIL import Image
from typing import List, Dict, Tuple, Optional
from llm_client import AzureOpenAIClient
from KG_visualizer import KnowledgeGraphVisualizer
from dotenv import load_dotenv
from pathlib import Path
from ontology.loader import PEKGOntology
from utils.pdf_utils import PDFProcessor
from utils.kg_utils import merge_graphs, merge_multiple_knowledge_graphs

load_dotenv()

class MultimodalFinancialKGBuilder:
    """
    A class to build a financial knowledge graph from PDF documents using Azure OpenAI's
    multimodal capabilities. It extracts entities and relationships from both text and
    visual elements (graphs, tables, charts, etc.) based on a predefined ontology.
    
    The class supports two construction modes:
    - "iterative": Processes the PDF page by page, using previous pages' graphs as context
    - "onego": Processes all pages independently and then merges the results
    """
    
    def __init__(self, model_name, deployment_name, project_name, construction_mode="iterative", 
                 ontology_path: str = Path(__file__).resolve().parent / "ontology" / "pekg_ontology.yaml"):
        """
        Initialize the MultimodalFinancialKGBuilder with the model name and deployment name.
        Args:
            model_name (str): The name of the model to be used for extraction.
            deployment_name (str): The name of the deployment in Azure OpenAI.
            construction_mode (str): Either "iterative" or "onego" for the KG construction approach.
            ontology_path (str): Path to the ontology file.
        """
        self.model_name = model_name
        self.project_name = project_name
        self.llm = AzureOpenAIClient(model_name=model_name)
        self.deployment_name = deployment_name
        self.ontology = PEKGOntology(ontology_path)
        self.pdf_path = Path(__file__).resolve().parents[3] / "pages" / project_name / f"Project_{project_name}_Teaser.pdf"
        self.page_dpi = 300
        self.vizualizer = KnowledgeGraphVisualizer()
        self.pdf_processor = PDFProcessor(self.pdf_path)
        
        if construction_mode not in ["iterative", "onego"]:
            raise ValueError("construction_mode must be either 'iterative' or 'onego'")
        self.construction_mode = construction_mode

    def identify_visual_elements(self, page_data: Dict, previous_graph: Dict = None) -> Dict:
        """
        Use multimodal LLM to identify and classify visual elements on a page.
        Args:
            page_data (Dict): Dictionary containing page image and metadata.
            previous_graph (Dict, optional): The merged graph from previous pages to provide context.
        Returns:
            Dict: Information about identified visual elements.
        """
        if self.construction_mode == "iterative" and previous_graph:
            previous_graph_json = json.dumps(previous_graph) if previous_graph else "{}"
            prompt = f"""
            You are a financial document analysis expert.
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

            Use the previous knowledge graph context to inform your analysis.
            Previous knowledge graph context (from previous pages):
            {previous_graph_json}

            This is page {page_data["page_num"]} of the document.
            """
        else:
            prompt = f"""
            You are a financial document analysis expert.
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
        
        if self.construction_mode == "iterative" and previous_graph:
            previous_graph_json = json.dumps(previous_graph) if previous_graph else "{}"
            prompt = f"""
            I need you to extract financial data from the visual elements in this page according to our ontology.
            
            The ontology we're using is:
            
            {ontology_desc}
            
            Based on your previous analysis:
            {visual_analysis["analysis"]}
            
            Previous knowledge graph context (from previous pages):
            {previous_graph_json}
            
            For each visual element (table, chart, graph, etc.):
            1. Extract all relevant entities, relationships, and attributes that match our ontology
            2. For tables, extract the data in structured form
            3. For charts/graphs, identify trends, key values, and relationships
            4. For diagrams/flowcharts, identify entities and their relationships
            5. For logos, associate them with the relevant company or brand
            6. Use the previous knowledge graph to maintain consistency in entity naming and IDs
            
            Format your response as JSON following this structure:
            {{
                "visual_elements": [
                    {{
                        "element_type": "table|chart|graph|heatmap|diagram|statement",
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
            """
        else:
            prompt = f"""
            You are a financial data extraction expert.  
            Your task is to extract an extensive and structured knowledge graph from the financial text provided.
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
        
        if self.construction_mode == "iterative" and previous_graph:
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
            - If entities from previous pages appear again, use the same IDs for consistency.

            ### RESPONSE ###
            Respond with *only* valid JSON in the specified format. Do not include any commentary. Do not include Markdown syntax (no ```json). Do not include explanations.
            """
        else:
            prompt = f"""
            You are a financial information extraction expert.
            Your task is to extract an extensive and structured knowledge graph from the financial text provided.
            The knowledge graph should include entities, relationships, and attributes based on the provided ontology.
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

            ### RESPONSE ###
            Respond with *only* valid JSON in the specified format. Do not include any commentary. Do not include Markdown syntax (no ```json). Do not include explanations.
            """
        return prompt

    def analyze_page_text_with_llm(self, text: str, previous_graph: Dict = None) -> Dict:
        """
        Analyze the provided text using the LLM to extract a knowledge graph.
        Args:
            text (str): The text to be analyzed.
            previous_graph (Dict, optional): The merged graph from previous pages to provide context.
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

    def analyze_page(self, page_data: Dict, previous_graph: Dict = None) -> Dict:
        """
        Perform comprehensive analysis of a single page, combining text and visual analysis.
        Args:
            page_data (Dict): Dictionary containing page image and metadata.
            previous_graph (Dict, optional): The merged graph from previous pages to provide context.
        Returns:
            Dict: Combined knowledge graph from the page.
        """
        
        visual_analysis = self.identify_visual_elements(page_data, previous_graph)
        visual_kg = self.extract_data_from_visuals(page_data, visual_analysis, previous_graph)
        text_kg = self.analyze_page_text_with_llm(page_data["text"], previous_graph)
    
        graphs_to_merge = []
      
        if "visual_elements" in visual_kg:
            for element in visual_kg["visual_elements"]:
                element_kg = {
                    "entities": element.get("entities", []),
                    "relationships": element.get("relationships", [])
                }
                graphs_to_merge.append(element_kg)
    
        graphs_to_merge.append(text_kg)
        
        if self.construction_mode == "iterative":
            page_kg = merge_graphs({}, graphs_to_merge)
        else:
            page_kg = merge_multiple_knowledge_graphs(graphs_to_merge)
        
        return page_kg

    def build_knowledge_graph_from_pdf(self, dump: bool = False) -> Dict:
        """
        Build a knowledge graph from a PDF file using the specified construction mode.
        
        Args:
            dump (bool, optional): Flag to indicate if the knowledge subgraphs should be saved.
        Returns:
            dict: The final merged knowledge graph.
        """
        if self.construction_mode == "iterative":
            return self._build_knowledge_graph_iterative(dump)
        else:
            return self._build_knowledge_graph_onego()
    
    def _build_knowledge_graph_iterative(self, dump: bool = False) -> Dict:
        """
        Build a knowledge graph iteratively from the pages of a PDF.
        Each page's subgraph is merged with the context of previous pages.
        
        Args:
            file_path (str): The path to the PDF file.
            dump (bool, optional): Flag to indicate if the knowledge subgraphs should be saved.
            project_name (str, optional): Project name for file naming when saving outputs.
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
            
            page_graph = self.analyze_page(page_data, merged_graph)

            if dump:
                entity_ids = {entity['id'] for entity in page_graph.get("entities", [])}
                filtered_relationships = [
                    rel for rel in page_graph.get("relationships", [])
                    if rel["source"] in entity_ids and rel["target"] in entity_ids
                ]
                page_graph["relationships"] = filtered_relationships
                
                output_dir = Path(__file__).resolve().parents[3] / "outputs" /  self.project_name / "pages"
                
                output_file = output_dir / f"multimodal_knowledge_graph_page_{page_num + 1}_{self.model_name}_iterative.json"
                with open(output_file, "w") as f:
                    json.dump(page_graph, f, indent=2)
                    
                output_file = str(output_dir / f"multimodal_knowledge_graph_page_{page_num + 1}_{self.model_name}_iterative.html")
                self.vizualizer.export_interactive_html(page_graph, output_file)
                print(f"Knowledge graph visualization saved to {output_file}")
            
            merged_graph = merge_graphs(merged_graph, [page_graph])
            
            print(f"Completed page {page_num + 1}/{num_pages}")
            print(f"Current graph: {len(merged_graph['entities'])} entities, {len(merged_graph['relationships'])} relationships")
        
        return merged_graph
    
    def _build_knowledge_graph_onego(self) -> Dict:
        """
        Build the knowledge graph from a PDF file using one-go multimodal analysis.
        Processes all pages independently and then merges results.
        
        Args:
            file_path (str): Path to the PDF file.
            project_name (str, optional): Project name for file naming when saving outputs.
        Returns:
            dict: The extracted knowledge graph in JSON format.
        """
        pages = self.extract_pages_from_pdf(self.pdf_path)
        print(f"Extracted {len(pages)} pages from {self.pdf_path}.")
        
        page_kgs = []
        for page_data in pages:
            page_kg = self.analyze_page(page_data)
            page_kgs.append(page_kg)
        
        merged_kg = merge_multiple_knowledge_graphs(page_kgs)
        
        return merged_kg

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
        3. Resolve any contradictory information
        4. Ensure relationship consistency (remove redundant relationships)
        5. Clean up any missing or null values
        
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
            return json.loads(content)
        except Exception as e:
            print("Error parsing consolidated knowledge graph:", e)
            return kg 

    def save_knowledge_graph(self, data: dict):
        """
        Save the knowledge graph data to a JSON file.
        Args:
            data (dict): The knowledge graph data to be saved.
            project_name (str): The name of the project for file naming.
        """
        json_output_file: str = Path(__file__).resolve().parents[3] / "outputs" / self.project_name / f"multimodal_knowledge_graph_{self.project_name}_{self.model_name}_{self.construction_mode}.json"
        with open(json_output_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Knowledge graph saved to {json_output_file}")
        
        html_output_file = str(Path(__file__).resolve().parents[3] / "outputs" / self.project_name / f"multimodal_knowledge_graph_{self.project_name}_{self.model_name}_{self.construction_mode}.html")
        self.vizualizer.export_interactive_html(data, html_output_file)
        print(f"Knowledge graph visualization saved to {html_output_file}")

