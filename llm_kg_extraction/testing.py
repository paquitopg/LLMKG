import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from REPOS.llm_kg_extraction.llm_kg_extraction.utils.pdf_utils import PDFProcessor
import sys
from pathlib import Path
import json

folder_path = str(Path(__file__).resolve().parent.parent.parents[1] / "pages" / "DECK" / "Project_DECK_TEASER.pdf")

processor = PDFProcessor(folder_path)

# get the path to the outputs folder 
output_path = Path(__file__).resolve().parent.parent.parents[1] / "pages" / "DECK" / "asPDF"
print(output_path)

processor.extract_pages_as_pdfs(output_path)