import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from KG_builder import FinancialKGBuilder
from KG_visualizer import KnowledgeGraphVisualizer
import sys
from pathlib import Path
import json

visualizer = KnowledgeGraphVisualizer()

# get the path to the outputs folder 
folder_path = Path(__file__).resolve().parent.parent.parents[1] / "outputs"
print(folder_path)

for i in range(1,19):
    data_path = folder_path / f"knowledge_graph_page_{i}_gpt-4.1_iterative.json"
    with open(data_path, "r") as f:
        data = json.load(f)
    output_path = folder_path / f"knowledge_graph_page_{i}_gpt-4.1_iterative.html"
    visualizer.export_interactive_html(kg_data=data, output_path= str(output_path))