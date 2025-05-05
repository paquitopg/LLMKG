import os
import json
import base64
import pymupdf
from io import BytesIO
from PIL import Image
from typing import List, Dict, Tuple, Optional
from openai import AzureOpenAI
from KG_visualizer import KnowledgeGraphVisualizer
from dotenv import load_dotenv
from pathlib import Path
from ontology.loader import PEKGOntology

load_dotenv()

class MultimodalFinancialKGBuilder:
    """
    A class to build a financial knowledge graph from PDF documents using Azure OpenAI's
    multimodal capabilities. It extracts entities and relationships from both text and
    visual elements (graphs, tables, charts, etc.) based on a predefined ontology.
    """
    
    def __init__(self, model_name, deployment_name, ontology_path: str = Path(__file__).resolve().parent / "ontology" / "pekg_ontology.yaml"):
        """
        Initialize the MultimodalFinancialKGBuilder with the model name and deployment name.
        Args:
            model_name (str): The name of the model to be used for extraction.
            deployment_name (str): The name of the deployment in Azure OpenAI.
            ontology_path (str): Path to the ontology file.
        """
        self.model_name = model_name
        self.client = self.make_client(self.model_name)
        self.deployment_name = deployment_name
        self.ontology = PEKGOntology(ontology_path)
        self.page_dpi = 300

    @staticmethod
    def make_client(model_name: str) -> AzureOpenAI:
        """
        Create an Azure OpenAI client based on the model name.
        Args:
            model_name (str): The name of the model to be used for extraction.
        Returns:
            AzureOpenAI: The Azure OpenAI client.
        """
        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT_" + model_name)
        AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY_" + model_name)
        AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME_" + model_name)
        AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION_" + model_name)

        return AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )

    def extract_pages_from_pdf(self, file_path: str) -> List[Dict]: 
        # To modify : shoudl do this only once and save it, and not every time we run the code
        # To do : integrate the code with the JPEG files I already have
        """
        Extract pages from a PDF file as images using PyMuPDF.
        Args:
            file_path (str): Path to the PDF file.
        Returns:
            List[Dict]: List of dictionaries containing page images and metadata.
        """
        pages = []
        doc = pymupdf.open(file_path)
        
        for page_num, page in enumerate(doc):

            width, height = page.rect.width, page.rect.height
            
            # Render page to a pixmap (image)
            matrix = pymupdf.Matrix(self.page_dpi/72, self.page_dpi/72) 
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            
            # Convert pixmap to PIL Image
            img_data = pixmap.tobytes("png")
            img = Image.open(BytesIO(img_data))
            
            # Convert image to base64 for API
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Get page text as fallback/additional context
            text = page.get_text()
            
            # Store the page data
            pages.append({
                "page_num": page_num + 1,
                "width": width,
                "height": height,
                "image_base64": img_base64,
                "text": text
            })
        
        return pages

    def identify_visual_elements(self, page_data: Dict) -> Dict:
        """
        Use multimodal LLM to identify and classify visual elements on a page.
        Args:
            page_data (Dict): Dictionary containing page image and metadata.
        Returns:
            Dict: Information about identified visual elements.
        """
        prompt = f"""
        Analyze this page from a financial document and identify all visual elements:
        1. Tables
        2. Charts/Graphs
        3. Diagrams
        4. Heatmaps
        5. Organizational charts
        6. Flow charts
        7. Financial statements

        For each identified element:
        - Describe what the element represents
        - Note the approximate position on the page
        - Describe the key information presented

        This is page {page_data["page_num"]} of the document.
        """

        response = self.client.chat.completions.create(
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
        
        # Parse the visual analysis into structured data
        return {
            "page_num": page_data["page_num"],
            "analysis": visual_analysis
        }

    def extract_data_from_visuals(self, page_data: Dict, visual_analysis: Dict) -> Dict:
        """
        Extract structured data from identified visual elements using multimodal LLM.
        Args:
            page_data (Dict): Dictionary containing page image and metadata.
            visual_analysis (Dict): Analysis of visual elements on the page.
        Returns:
            Dict: Extracted data from visual elements in structured format.
        """
        ontology_desc = self.ontology.format_for_prompt()
        
        prompt = f"""
        I need you to extract financial data from the visual elements in this page according to our ontology.
        
        The ontology we're using is:
        
        {ontology_desc}
        
        Based on your previous analysis:
        {visual_analysis["analysis"]}
        
        For each visual element (table, chart, graph, etc.):
        1. Extract all relevant entities, relationships, and attributes that match our ontology
        2. For tables, extract the data in structured form
        3. For charts/graphs, identify trends, key values, and relationships
        4. For diagrams/flowcharts, identify entities and their relationships
        
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

        response = self.client.chat.completions.create(
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
        
        # Clean up the response to ensure it's valid JSON
        if content.startswith("```json"):
            content = content.lstrip("```json").rstrip("```").strip()
        elif "```" in content:
            # Extract the JSON part if it's embedded in explanatory text
            start = content.find("```")
            end = content.rfind("```")
            if start != -1 and end != -1:
                content = content[start+3:end].strip()
                if content.startswith("json"):
                    content = content[4:].strip()
                    
        try:
            return json.loads(content)
        except Exception as e:
            print(f"Error parsing JSON from page {page_data['page_num']}:", e)
            print("Raw content:", content)
            # Return a minimal valid structure
            return {"visual_elements": []}

    def build_prompt_for_text_analysis(self, text: str) -> str:
        """
        Build the prompt for the LLM based on the provided text and ontology.
        Args:
            text (str): The text to be analyzed.
        Returns:
            str: The formatted prompt for the LLM.
        """
        ontology_desc = self.ontology.format_for_prompt()
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

    def analyze_page_text_with_llm(self, text: str) -> Dict:
        """
        Analyze the provided text using the LLM to extract a knowledge graph.
        Args:
            text (str): The text to be analyzed.
        Returns:
            dict: The extracted knowledge graph in JSON format.
        """
        prompt = self.build_prompt_for_text_analysis(text)

        response = self.client.chat.completions.create(
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

    def merge_knowledge_graphs(self, graphs: List[Dict]) -> Dict:
        """
        Merge multiple knowledge graphs into a single unified graph.
        Args:
            graphs (List[Dict]): List of knowledge graphs to merge.
        Returns:
            Dict: Merged knowledge graph.
        """
        # Initialize the merged graph
        merged_graph = {"entities": [], "relationships": []}
        
        # Track entities by their normalized name to avoid duplicates
        entity_map = {}  # Maps normalized entity name to entity ID
        next_entity_id = 1
        next_rel_id = 1
        
        for graph in graphs:
            # Process entities
            if "entities" in graph:
                for entity in graph["entities"]:
                    # Create a normalized key based on entity type and name
                    entity_key = None
                    if "name" in entity:
                        entity_key = f"{entity['type']}:{entity['name'].lower()}"
                    elif "id" in entity:  # Use ID as fallback
                        entity_key = f"{entity['type']}:{entity['id']}"
                    
                    if entity_key:
                        if entity_key not in entity_map:
                            # Create a new unique ID for this entity
                            new_id = f"e{next_entity_id}"
                            next_entity_id += 1
                            
                            # Replace the original ID with our new one
                            old_id = entity["id"]
                            entity["id"] = new_id
                            entity_map[entity_key] = {"new_id": new_id, "old_id": old_id}
                            
                            # Add to the merged entities
                            merged_graph["entities"].append(entity)
            
            # Process relationships
            if "relationships" in graph:
                for rel in graph["relationships"]:
                    # Find the new IDs for the source and target entities
                    source_old = rel["source"]
                    target_old = rel["target"]
                    
                    # Find the corresponding new IDs
                    source_new = None
                    target_new = None
                    
                    for key, value in entity_map.items():
                        if value["old_id"] == source_old:
                            source_new = value["new_id"]
                        if value["old_id"] == target_old:
                            target_new = value["new_id"]
                    
                    # Only add the relationship if we found both entities
                    if source_new and target_new:
                        new_rel = {
                            "id": f"r{next_rel_id}",
                            "source": source_new,
                            "target": target_new,
                            "type": rel["type"]
                        }
                        next_rel_id += 1
                        merged_graph["relationships"].append(new_rel)
        
        return merged_graph

    def consolidate_knowledge_graph(self, kg: Dict) -> Dict:
        """
        Use LLM to consolidate and clean up the knowledge graph, resolving duplicates
        and inconsistencies.
        Args:
            kg (Dict): The raw knowledge graph to consolidate.
        Returns:
            Dict: The consolidated knowledge graph.
        """
        # Convert the knowledge graph to a string
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
        
        response = self.client.chat.completions.create(
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
            return kg  # Return original if consolidation fails

    def analyze_page(self, page_data: Dict) -> Dict:
        """
        Perform comprehensive analysis of a single page, combining text and visual analysis.
        Args:
            page_data (Dict): Dictionary containing page image and metadata.
        Returns:
            Dict: Combined knowledge graph from the page.
        """
        print(f"Processing page {page_data['page_num']}...")
        
        # Identify visual elements on the page
        visual_analysis = self.identify_visual_elements(page_data)
        
        # Extract structured data from visual elements
        visual_kg = self.extract_data_from_visuals(page_data, visual_analysis)
        
        # Process the text on the page
        text_kg = self.analyze_page_text_with_llm(page_data["text"])
        
        # Combine the knowledge graphs from visuals and text
        graphs_to_merge = []
        
        # Add visual element knowledge graphs
        if "visual_elements" in visual_kg:
            for element in visual_kg["visual_elements"]:
                element_kg = {
                    "entities": element.get("entities", []),
                    "relationships": element.get("relationships", [])
                }
                graphs_to_merge.append(element_kg)
        
        # Add text knowledge graph
        graphs_to_merge.append(text_kg)
        
        # Merge all the knowledge graphs
        return self.merge_knowledge_graphs(graphs_to_merge)

    def build_knowledge_graph_from_pdf(self, pdf_path: str) -> Dict:
        """
        Build the knowledge graph from a PDF file using multimodal analysis.
        Args:
            pdf_path (str): Path to the PDF file.
        Returns:
            dict: The extracted knowledge graph in JSON format.
        """
        # Extract pages from PDF
        pages = self.extract_pages_from_pdf(pdf_path)
        print(f"Extracted {len(pages)} pages from {pdf_path}")
        
        # Analyze each page
        page_kgs = []
        for page_data in pages:
            page_kg = self.analyze_page(page_data)
            page_kgs.append(page_kg)
        
        # Merge knowledge graphs from all pages
        merged_kg = self.merge_knowledge_graphs(page_kgs)
        
        # Consolidate the final knowledge graph
        final_kg = self.consolidate_knowledge_graph(merged_kg)
        
        return final_kg
    
    def save_knowledge_graph(self, data: dict, project_name: str):
        """
        Save the knowledge graph data to a JSON file.
        Args:
            data (dict): The knowledge graph data to be saved.
            project_name (str): The name of the project for file naming.
        """
        output_file: str = Path(__file__).resolve().parents[1] / "examples" / f"multimodal_knowledge_graph_{project_name}_{self.model_name}.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Knowledge graph saved to {output_file}")
        
    def visualize_knowledge_graph(self, data: dict, project_name: str):
        """
        Visualize the knowledge graph using the KnowledgeGraphVisualizer.
        Args:
            data (dict): The knowledge graph data to visualize.
            project_name (str): The name of the project for file naming.
        """
        visualizer = KnowledgeGraphVisualizer()
        output_path = Path(__file__).resolve().parents[1] / "examples" / f"multimodal_knowledge_graph_{project_name}_{self.model_name}.html"
        visualizer.visualize(data, str(output_path))
        
        print(f"Knowledge graph visualization saved to {output_path}")
