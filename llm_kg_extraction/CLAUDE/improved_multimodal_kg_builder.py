import os
import json
import base64
import pymupdf
import numpy as np
import cv2
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

try:
    import pytesseract  # For OCR
except ImportError:
    print("Warning: pytesseract not installed. OCR functionality will be limited.")

load_dotenv()

class EnhancedMultimodalFinancialKGBuilder:
    """
    An enhanced version of the MultimodalFinancialKGBuilder that processes visual elements
    directly as images rather than through descriptions.
    
    Key improvements:
    1. Direct extraction and processing of visual elements
    2. Specialized handling for different visual element types
    3. OCR integration for text-based visual elements
    4. Spatial relationship analysis between visual elements
    5. Confidence scoring for extracted information
    """
    
    def __init__(self, model_name, deployment_name, project_name, construction_mode="iterative", 
                 ontology_path: str = Path(__file__).resolve().parent / "ontology" / "pekg_ontology.yaml"):
        """
        Initialize the EnhancedMultimodalFinancialKGBuilder.
        Args:
            model_name (str): The name of the model to be used for extraction.
            deployment_name (str): The name of the deployment in Azure OpenAI.
            project_name (str): The name of the project for file naming.
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
        
        # Thresholds for visual element extraction
        self.min_element_area = 5000  # Minimum area to consider as a visual element
        self.element_confidence_threshold = 0.7  # Minimum confidence for element classification

    def extract_visual_elements(self, page_data: Dict) -> List[Dict]:
        """
        Extract and segment individual visual elements from a page image.
        Returns each visual element as a separate image with its bounding box.
        
        Args:
            page_data (Dict): Dictionary containing page image and metadata.
            
        Returns:
            List[Dict]: List of visual elements with their attributes.
        """
        # Convert base64 to image
        img_bytes = base64.b64decode(page_data['image_base64'])
        img = Image.open(BytesIO(img_bytes))
        img_np = np.array(img)
        
        # Convert to grayscale for processing
        if len(img_np.shape) == 3:  # Color image
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:  # Already grayscale
            gray = img_np
        
        # Use adaptive thresholding for better results with variable lighting
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Alternative: Use morphological operations to better identify tables and charts
        kernel = np.ones((5,5), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours_morph, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine approaches
        all_contours = contours + contours_morph
        
        # Filter and consolidate overlapping contours
        filtered_contours = self._filter_and_merge_contours(all_contours)
        
        visual_elements = []
        for i, contour in enumerate(filtered_contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by minimum size
            if w * h < self.min_element_area:
                continue
                
            # Crop the element with a small margin
            margin = 10
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(img_np.shape[1], x + w + margin)
            y_end = min(img_np.shape[0], y + h + margin)
            
            element_img = img_np[y_start:y_end, x_start:x_end]
            element_pil = Image.fromarray(element_img)
            
            # Convert to base64
            buffered = BytesIO()
            element_pil.save(buffered, format="PNG")
            element_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Extract text using OCR if available
            try:
                element_text = pytesseract.image_to_string(element_pil)
            except:
                element_text = ""
            
            visual_elements.append({
                "id": f"element_{page_data['page_num']}_{i}",
                "position": {"x": x, "y": y, "width": w, "height": h},
                "image_base64": element_base64,
                "page_num": page_data["page_num"],
                "ocr_text": element_text
            })
        
        return visual_elements
    
    def _filter_and_merge_contours(self, contours: List) -> List:
        """
        Filter out small contours and merge overlapping ones.
        
        Args:
            contours (List): List of contours to process
            
        Returns:
            List: Filtered and merged contours
        """
        # Filter by minimum area
        filtered = [c for c in contours if cv2.contourArea(c) > self.min_element_area / 2]
        
        # If no contours pass the filter, return empty list
        if not filtered:
            return []
        
        # Convert contours to bounding boxes
        bboxes = [cv2.boundingRect(c) for c in filtered]
        
        # Merge overlapping boxes
        merged_bboxes = []
        while bboxes:
            # Take the first box
            current = bboxes.pop(0)
            x1, y1, w1, h1 = current
            
            # Check if it overlaps with any other box
            i = 0
            while i < len(bboxes):
                x2, y2, w2, h2 = bboxes[i]
                
                # Check for overlap
                if (x1 < x2 + w2 and x1 + w1 > x2 and
                    y1 < y2 + h2 and y1 + h1 > y2):
                    # Merge boxes
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = max(x1 + w1, x2 + w2) - x
                    h = max(y1 + h1, y2 + h2) - y
                    
                    # Update current box
                    current = (x, y, w, h)
                    x1, y1, w1, h1 = current
                    
                    # Remove the merged box
                    bboxes.pop(i)
                else:
                    i += 1
            
            merged_bboxes.append(current)
        
        # Convert bounding boxes back to contours
        result_contours = []
        for x, y, w, h in merged_bboxes:
            points = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
            result_contours.append(points)
        
        return result_contours

    def classify_visual_element(self, element: Dict) -> Tuple[str, float]:
        """
        Use multimodal LLM to classify the type of visual element.
        
        Args:
            element (Dict): Visual element data including image
            
        Returns:
            Tuple[str, float]: Element type and confidence score
        """
        prompt = """
        What type of visual element is this from a financial document?
        Classify it as one of:
        - table
        - chart/graph
        - diagram
        - organizational chart
        - flow chart
        - financial statement
        - logo
        - text block
        - other
        
        Just respond with the element type and a confidence score between 0 and 1.
        Format: "type:confidence"
        Example: "table:0.95" or "chart/graph:0.87"
        """
        
        response = self.llm.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a financial document analysis assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{element['image_base64']}"}}
                ]}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        content = response.choices[0].message.content.strip()
        
        try:
            element_type, confidence_str = content.split(":")
            element_type = element_type.strip().lower()
            confidence = float(confidence_str.strip())
            return element_type, confidence
        except:
            # Default fallback
            return "other", 0.5

    def process_table(self, element: Dict) -> Dict:
        """
        Process a table visual element to extract structured data.
        
        Args:
            element (Dict): Visual element data including image
            
        Returns:
            Dict: Knowledge graph segment extracted from the table
        """
        ontology_desc = self.ontology.format_for_prompt()
        
        prompt = f"""
        Extract the data from this financial table.
        
        First, extract the raw table structure as a JSON object with headers and rows.
        Then, identify entities and relationships according to this ontology:
        
        {ontology_desc}
        
        OCR text (if helpful):
        {element.get('ocr_text', '')}
        
        Format your response as JSON:
        {{
            "table_data": {{
                "headers": ["Column1", "Column2", ...],
                "rows": [
                    ["Value1", "Value2", ...],
                    ...
                ]
            }},
            "entities": [
                {{"id": "e1", "type": "pekg:Company", "name": "ABC Capital"}},
                ...
            ],
            "relationships": [
                {{"source": "e1", "target": "e2", "type": "pekg:receivedInvestment"}},
                ...
            ]
        }}
        """
        
        response = self.llm.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a financial data extraction assistant specializing in tables."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{element['image_base64']}"}}
                ]}
            ],
            temperature=0.1,
            max_tokens=8000
        )
        
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```json"):
            content = content.lstrip("```json").rstrip("```").strip()
        
        try:
            result = json.loads(content)
            # Add element metadata to result
            result["element_id"] = element["id"]
            result["element_type"] = "table"
            return result
        except Exception as e:
            print(f"Error parsing table extraction result: {e}")
            return {
                "element_id": element["id"],
                "element_type": "table",
                "entities": [],
                "relationships": []
            }

    def process_chart(self, element: Dict) -> Dict:
        """
        Process a chart/graph visual element to extract trends and data points.
        
        Args:
            element (Dict): Visual element data including image
            
        Returns:
            Dict: Knowledge graph segment extracted from the chart
        """
        ontology_desc = self.ontology.format_for_prompt()
        
        prompt = f"""
        Extract data from this financial chart/graph.
        
        Identify:
        1. The type of chart (line, bar, pie, etc.)
        2. The title and axis labels
        3. Key data points and trends
        4. Any entities and relationships according to this ontology:
        
        {ontology_desc}
        
        OCR text (if helpful):
        {element.get('ocr_text', '')}
        
        Format your response as JSON:
        {{
            "chart_type": "line|bar|pie|scatter|other",
            "title": "Chart title",
            "axes": {{
                "x": "X-axis label",
                "y": "Y-axis label"
            }},
            "data_points": [
                {{"x": "Label1", "y": 10.5}},
                ...
            ],
            "trends": [
                "Trend description 1",
                ...
            ],
            "entities": [
                {{"id": "e1", "type": "pekg:Company", "name": "ABC Capital"}},
                ...
            ],
            "relationships": [
                {{"source": "e1", "target": "e2", "type": "pekg:receivedInvestment"}},
                ...
            ]
        }}
        """
        
        response = self.llm.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a financial data extraction assistant specializing in charts and graphs."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{element['image_base64']}"}}
                ]}
            ],
            temperature=0.1,
            max_tokens=8000
        )
        
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```json"):
            content = content.lstrip("```json").rstrip("```").strip()
        
        try:
            result = json.loads(content)
            # Add element metadata to result
            result["element_id"] = element["id"]
            result["element_type"] = "chart"
            return result
        except Exception as e:
            print(f"Error parsing chart extraction result: {e}")
            return {
                "element_id": element["id"],
                "element_type": "chart",
                "entities": [],
                "relationships": []
            }

    def process_generic_visual(self, element: Dict) -> Dict:
        """
        Process any other visual element type.
        
        Args:
            element (Dict): Visual element data including image
            
        Returns:
            Dict: Knowledge graph segment extracted from the visual element
        """
        ontology_desc = self.ontology.format_for_prompt()
        
        prompt = f"""
        Extract information from this visual element in a financial document.
        
        Identify any entities and relationships according to this ontology:
        
        {ontology_desc}
        
        OCR text (if helpful):
        {element.get('ocr_text', '')}
        
        Format your response as JSON:
        {{
            "description": "Description of the visual element",
            "key_information": [
                "Key point 1",
                ...
            ],
            "entities": [
                {{"id": "e1", "type": "pekg:Company", "name": "ABC Capital"}},
                ...
            ],
            "relationships": [
                {{"source": "e1", "target": "e2", "type": "pekg:receivedInvestment"}},
                ...
            ]
        }}
        """
        
        response = self.llm.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a financial data extraction assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{element['image_base64']}"}}
                ]}
            ],
            temperature=0.1,
            max_tokens=8000
        )
        
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```json"):
            content = content.lstrip("```json").rstrip("```").strip()
        
        try:
            result = json.loads(content)
            # Add element metadata to result
            result["element_id"] = element["id"]
            result["element_type"] = "other"
            return result
        except Exception as e:
            print(f"Error parsing generic visual extraction result: {e}")
            return {
                "element_id": element["id"],
                "element_type": "other",
                "entities": [],
                "relationships": []
            }

    def analyze_spatial_relationships(self, visual_elements: List[Dict]) -> List[Dict]:
        """
        Analyze spatial relationships between visual elements on a page.
        
        Args:
            visual_elements (List[Dict]): List of visual elements with position data
            
        Returns:
            List[Dict]: List of spatial relationships between elements
        """
        relationships = []
        
        # Sort elements by position
        elements_sorted_y = sorted(visual_elements, key=lambda x: x["position"]["y"])
        elements_sorted_x = sorted(visual_elements, key=lambda x: x["position"]["x"])
        
        # Check for vertical relationships (above/below)
        for i in range(len(elements_sorted_y) - 1):
            e1 = elements_sorted_y[i]
            e2 = elements_sorted_y[i + 1]
            
            y1_bottom = e1["position"]["y"] + e1["position"]["height"]
            y2_top = e2["position"]["y"]
            
            # If elements are close vertically
            if y2_top - y1_bottom < 50:  # Threshold for "follows" relationship
                relationships.append({
                    "source": e1["id"],
                    "target": e2["id"],
                    "type": "pekg:visuallyPrecedes"
                })
        
        # Check for horizontal relationships (side by side)
        for i in range(len(elements_sorted_x) - 1):
            e1 = elements_sorted_x[i]
            e2 = elements_sorted_x[i + 1]
            
            x1_right = e1["position"]["x"] + e1["position"]["width"]
            x2_left = e2["position"]["x"]
            
            # If elements are close horizontally and at similar vertical position
            if (x2_left - x1_right < 50 and  # Horizontal threshold
                abs(e1["position"]["y"] - e2["position"]["y"]) < 100):  # Vertical alignment threshold
                relationships.append({
                    "source": e1["id"],
                    "target": e2["id"],
                    "type": "pekg:visuallyAdjacentTo"
                })
        
        return relationships

    def extract_relationship_from_elements(self, processed_elements: List[Dict]) -> List[Dict]:
        """
        Extract relationships between entities across different visual elements.
        
        Args:
            processed_elements (List[Dict]): List of processed visual elements with entities
            
        Returns:
            List[Dict]: List of relationships between entities across elements
        """
        # Create a prompt that includes all entities across elements
        all_entities = []
        for element in processed_elements:
            if "entities" in element:
                for entity in element.get("entities", []):
                    all_entities.append({
                        "id": entity["id"],
                        "type": entity["type"],
                        "name": entity.get("name", ""),
                        "element_id": element["element_id"]
                    })
        
        if not all_entities:
            return []
        
        ontology_desc = self.ontology.format_for_prompt()
        entities_json = json.dumps(all_entities, indent=2)
        
        prompt = f"""
        Identify relationships between these entities from different visual elements according to our ontology:
        
        {ontology_desc}
        
        Entities:
        {entities_json}
        
        Format your response as a JSON array of relationships:
        [
            {{"source": "entity_id1", "target": "entity_id2", "type": "pekg:relationshipType"}},
            ...
        ]
        
        Only include relationships that are likely to exist based on the entity types and the ontology.
        """
        
        response = self.llm.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a financial knowledge graph assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```json"):
            content = content.lstrip("```json").rstrip("```").strip()
        
        try:
            return json.loads(content)
        except Exception as e:
            print(f"Error parsing cross-element relationships: {e}")
            return []

    def analyze_page_text_with_llm(self, text: str, previous_graph: Dict = None) -> Dict:
        """
        Analyze the provided text using the LLM to extract a knowledge graph.
        Args:
            text (str): The text to be analyzed.
            previous_graph (Dict, optional): The merged graph from previous pages to provide context.
        Returns:
            dict: The extracted knowledge graph in JSON format.
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
        print(f"Processing page {page_data['page_num']}...")
        
        # Extract text knowledge graph
        text_kg = self.analyze_page_text_with_llm(page_data["text"], previous_graph)
        
        # Extract and process visual elements
        visual_elements = self.extract_visual_elements(page_data)
        print(f"Extracted {len(visual_elements)} visual elements from page {page_data['page_num']}")
        
        processed_elements = []
        for element in visual_elements:
            # Classify the visual element
            element_type, confidence = self.classify_visual_element(element)
            element["element_type"] = element_type
            
            # Only process if we have high confidence
            if confidence < self.element_confidence_threshold:
                continue
                
            # Process based on element type
            if element_type == "table":
                processed_element = self.process_table(element)
            elif element_type in ["chart", "chart/graph", "graph"]:
                processed_element = self.process_chart(element)
            else:
                processed_element = self.process_generic_visual(element)
                
            processed_elements.append(processed_element)
        
        # Extract spatial relationships between visual elements
        spatial_relationships = self.analyze_spatial_relationships(visual_elements)
        
        # Extract cross-element entity relationships
        cross_element_relationships = self.extract_relationship_from_elements(processed_elements)
        
        # Collect all entities and relationships
        graphs_to_merge = [text_kg]
        
        # Add each visual element's knowledge graph
        for element in processed_elements:
            element_kg = {
                "entities": element.get("entities", []),
                "relationships": element.get("relationships", [])
            }
            graphs_to_merge.append(element_kg)
        
        # Add spatial relationships graph
        if spatial_relationships:
            spatial_kg = {
                "entities": [],  # No new entities
                "relationships": spatial_relationships
            }
            graphs_to_merge.append(spatial_kg)
        
        # Add cross-element relationships graph
        if cross_element_relationships:
            cross_element_kg = {
                "entities": [],  # No new entities
                "relationships": cross_element_relationships
            }
            graphs_to_merge.append(cross_element_kg)
        
        # Merge all knowledge graphs
        if self.construction_mode == "iterative":
            page_kg = merge_graphs({}, graphs_to_merge)
        else:
            page_kg = merge_multiple_knowledge_graphs(graphs_to_merge)
        
        print(f"Page {page_data['page_num']} knowledge graph: {len(page_kg['entities'])} entities, {len(page_kg['relationships'])} relationships")
        return page_kg