import os
import pymupdf
import json

from typing import List, Dict
from openai import AzureOpenAI
from KG_visualizer import KnowledgeGraphVisualizer
from dotenv import load_dotenv
from pathlib import Path
from ontology.loader import PEKGOntology

load_dotenv()

class FinancialKGBuilder:
    """
    A class to build a financial knowledge graph from text using Azure OpenAI.
    It extracts entities and relationships based on a predefined ontology, 
    iteratively building a graph page by page with merged subgraphs.
    """
    
    def __init__(self, model_name, deployment_name, ontology_path: str = Path(__file__).resolve().parent / "ontology" / "pekg_ontology.yaml"):
        """
        Initialize the FinancialKGBuilder with the model name and deployment name.
        Args:
            model_name (str): The name of the model to be used for extraction.
            deployment_name (str): The name of the deployment in Azure OpenAI.
            ontology_path (str): Path to the ontology file.
        """
        self.model_name = model_name
        self.client = self.make_client(self.model_name)
        self.deployment_name = deployment_name
        self.ontology = PEKGOntology(ontology_path)

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

    def extract_text_from_pdf(self, file_path: str) -> List[str]:
        """
        Extract text from a PDF file using PyMuPDF.
        Args:
            file_path (str): Path to the PDF file.
        Returns:
            List[str]: A list of texts, one for each page in the PDF.
        """
        doc = pymupdf.open(file_path)
        return [page.get_text() for page in doc]

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

        response = self.client.chat.completions.create(
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

    def build_knowledge_graph_from_pdf(self, file_path: str) -> Dict:
        """
        Build a knowledge graph iteratively from the pages of a PDF.
        Each page's subgraph is merged with the context of previous pages.
        Args:
            file_path (str): The path to the PDF file.
        Returns:
            dict: The final merged knowledge graph.
        """
        pages_text = self.extract_text_from_pdf(file_path)
        merged_graph = {}

        for i, page_text in enumerate(pages_text):
            print(f"Processing page {i+1}...")
            page_graph = self.analyze_text_with_llm(page_text, merged_graph)
            merged_graph = self.merge_graphs(merged_graph, page_graph)

        return merged_graph

    def merge_graphs(self, graph1: Dict, graph2: Dict) -> Dict:
        """
        Merge two knowledge graphs.
        Args:
            graph1 (dict): The first knowledge graph.
            graph2 (dict): The second knowledge graph.
        Returns:
            dict: The merged knowledge graph.
        """
        entities = {**graph1.get('entities', {}), **graph2.get('entities', {})}
        relationships = {**graph1.get('relationships', {}), **graph2.get('relationships', {})}

        return {
            "entities": list(entities.values()),
            "relationships": list(relationships.values())
        }

    def save_knowledge_graph(self, data: dict, project_name: str):
        """
        Save the knowledge graph data to a JSON file.
        Args:
            data (dict): The knowledge graph data to be saved.
            project_name (str): The name of the project for file naming.
        """
        output_file: str = Path(__file__).resolve().parents[1] / "examples" / f"knowledge_graph_{project_name}_{self.model_name}.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)