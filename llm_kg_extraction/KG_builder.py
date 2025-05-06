import os
import pymupdf
import json

from typing import List, Dict, Tuple
from llm_client import AzureOpenAIClient
from KG_visualizer import KnowledgeGraphVisualizer
from dotenv import load_dotenv
from pathlib import Path
from ontology.loader import PEKGOntology

load_dotenv()

class FinancialKGBuilder:
    """
    A class to build a financial knowledge graph from text using Azure OpenAI.
    It extracts entities and relationships based on a predefined ontology.
    """
    
    def __init__(self, model_name, deployment_name, ontology_path: str =  Path(__file__).resolve().parent / "ontology" / "pekg_ontology.yaml"):
        """
        Initialize the FinancialKGBuilder with the model name and deployment name.
        Args:
            model_name (str): The name of the model to be used for extraction.
            deployment_name (str): The name of the deployment in Azure OpenAI.
            ontology_path (str): Path to the ontology file.
        """
        
        self.model_name = model_name

        client = AzureOpenAIClient(model_name=model_name)
        self.client = client.make_client()
        
        self.deployment_name = deployment_name
        self.ontology = PEKGOntology(ontology_path)

    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file using PyMuPDF.
        Args:
            file_path (str): Path to the PDF file.
        Returns:
            str: Extracted text from the PDF file.
        """

        doc = pymupdf.open(file_path)
        return "\n".join([page.get_text() for page in doc])

    def build_prompt(self, text: str) -> str:
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


    def analyze_text_with_llm(self, text: str) -> Dict:
        """
        Analyze the provided text using the LLM to extract a knowledge graph.
        Args:
            text (str): The text to be analyzed.
        Returns:
            dict: The extracted knowledge graph in JSON format.
        """

        prompt = self.build_prompt(text)

        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a financial information extraction assistant."
                "Your task is to extract a knowledge graph from the financial text provided."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10000
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.lstrip("```json").rstrip("```").strip()

        print("LLM response:", content)

        try:
            return json.loads(content)
        except Exception as e:
            print("Error parsing LLM response:", e)
            return {}
        
    def build_knowledge_graph_from_pdf(self, pdf_path: str, dump) -> Dict:
        """
        Build the knowledge graph from a PDF file.
        Args:
            pdf_path (str): Path to the PDF file.
        Returns:
            dict: The extracted knowledge graph in JSON format.
        """

        text = self.extract_text_from_pdf(pdf_path)
        knowledge_graph = self.analyze_text_with_llm(text)
        
        return knowledge_graph
    
    def save_knowledge_graph(self, data: dict, project_name: str):
        """
        Save the knowledge graph data to a JSON file.
        Args:
            data (dict): The knowledge graph data to be saved.
            project_name (str): The name of the project for file naming.
        """

        output_file: str = Path(__file__).resolve().parents[3] / "outputs" / project_name / f"knowledge_graph_{project_name}_{self.model_name}.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)