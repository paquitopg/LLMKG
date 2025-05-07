import os
import pymupdf
import json

from typing import List, Dict
from llm_client import AzureOpenAIClient
from KG_visualizer import KnowledgeGraphVisualizer
from utils.pdf_utils import PDFProcessor
from pathlib import Path
from ontology.loader import PEKGOntology

from utils.kg_utils import merge_knowledge_graphs
from dotenv import load_dotenv

load_dotenv()

class FinancialKGBuilder:
    """
    A class to build a financial knowledge graph from text using Azure OpenAI.
    It extracts entities and relationships based on a predefined ontology, 
    iteratively building a graph page by page with merged subgraphs.
    Alternatively, it can build a knowledge graph from a whole document in one go.
    The class also provides functionality to visualize the knowledge graph using PyVis.
    """
    
    def __init__(self, model_name, deployment_name, project_name, construction_mode, ontology_path: str =  Path(__file__).resolve().parent / "ontology" / "pekg_ontology.yaml"):
        """
        Initialize the FinancialKGBuilder with the model name and deployment name.
        Args:
            model_name (str): The name of the model to be used for extraction.
            deployment_name (str): The name of the deployment in Azure OpenAI.
            pdf_path (str): Path to the PDF file to be processed.
            construction_mode (str) : The construction mode ("iterative" or "onego").
            ontology_path (str): Path to the ontology file.
        """
        
        self.model_name = model_name
        self.project_name = project_name
        self.llm = AzureOpenAIClient(model_name=model_name)
        self.deployment_name = deployment_name
        self.ontology = PEKGOntology(ontology_path)
        self.pdf_path = Path(__file__).resolve().parents[3] / "pages" / project_name / f"Project_{project_name}_Teaser.pdf"
        self.vizualizer = KnowledgeGraphVisualizer()
        self.pdf_processor = PDFProcessor(self.pdf_path)

        if construction_mode not in ["iterative", "onego"]:
            raise ValueError("construction_mode must be either 'iterative' or 'onego'")
        self.construction_mode = construction_mode

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
        Args:
            text (str): The text to be analyzed.
            previous_graph (dict, optional): The merged graph from previous pages to provide context.
        Returns:
            dict: The extracted knowledge graph in JSON format.
        """
        prompt = self.build_prompt(text, previous_graph)

        response = self.llm.client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "system", "content": "You are a financial information extraction assistant."
                "Your task is to extract a knowledge graph from the financial text provided."},
                      {"role": "user", "content": prompt}],
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
            return {}

    def build_knowledge_graph_from_pdf(self, dump: bool = False) -> Dict:
        """
        Build a knowledge graph iteratively from the pages of a PDF.
        Each page's subgraph is merged with the context of previous pages.
        Args:
            file_path (str): The path to the PDF file.
            dump (bool, optional): Flag to indicate if the knowledge subgraphs should be saved.
        Returns:
            dict: The final merged knowledge graph.
        """

        if self.construction_mode == "onego":
            text = self.pdf_processor.extract_text()
            merged_graph = self.analyze_text_with_llm(text)

        else : 
            pages_text = self.pdf_processor.extract_text_as_list()
            merged_graph = {}

            for i, page_text in enumerate(pages_text):
                print(f"Processing page {i+1}...")
                page_graph = self.analyze_text_with_llm(page_text, merged_graph)
                merged_graph = merge_knowledge_graphs(merged_graph, page_graph)

                if dump:
                    entity_ids = {entity['id'] for entity in page_graph.get("entities", [])}
                    filtered_relationships = [
                        rel for rel in page_graph.get("relationships", [])
                        if rel["source"] in entity_ids and rel["target"] in entity_ids
                    ]   
                    page_graph["relationships"] = filtered_relationships
            
                    output_file = Path(__file__).resolve().parents[3] / "outputs" / self.project_name / "pages" / f"knowledge_graph_page_{i+1}_{self.model_name}_iterative.json"
                    with open(output_file, "w") as f:
                        json.dump(page_graph, f, indent=2)
                    
                    output_file = str(Path(__file__).resolve().parents[3] / "outputs" / self.project_name / "pages" / f"knowledge_graph_page_{i + 1}_{self.model_name}_iterative.html")
                    self.vizualizer.export_interactive_html(page_graph, output_file)

        print("Knowledge graph building process completed.")

        return merged_graph
    
    def save_knowledge_graph(self, data: dict):
        """
        Save the knowledge graph data to a JSON file.
        Save the knowledge graph data to an HTML file. 
        Args:
            data (dict): The knowledge graph data to be saved.
            project_name (str): The name of the project for file naming.
        """
        json_output_file = Path(__file__).resolve().parents[3] / "outputs" / self.project_name / f"knowledge_graph_{self.project_name}_{self.model_name}_{self.construction_mode}.json"
        with open(json_output_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Knowledge graph saved to {json_output_file}")
        
        html_output_file = str(Path(__file__).resolve().parents[3] / "outputs" / self.project_name / f"knowledge_graph_{self.project_name}_{self.model_name}_{self.construction_mode}.html")
        self.vizualizer.export_interactive_html(data, html_output_file)
        print(f"Knowledge graph visualization saved to {html_output_file}")