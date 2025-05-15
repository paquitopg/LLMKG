# Updated company_identifier.py
import os
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class CompanyIdentifier:
    """
    Identifies and distinguishes between the main company, advisory firms, and project codenames
    in financial documents by analyzing the first and last few pages of the PDF.
    """
    
    def __init__(self, llm_client, pdf_processor, pages_to_analyze=6):
        """
        Initialize the CompanyIdentifier.
        
        Args:
            llm_client: The LLM client to use for company identification.
            pdf_processor: The PDF processor to extract text from PDFs.
            pages_to_analyze (int): Number of pages to analyze from the beginning and end of the document.
        """
        self.llm_client = llm_client
        self.pdf_processor = pdf_processor
        self.pages_to_analyze = pages_to_analyze
    
    def identify_companies(self, project_name: str) -> Dict:
        """
        Identify the main company, advisory firms, and project codename from the PDF
        by analyzing both the first and last few pages.
        
        Args:
            project_name (str): The codename of the project.
            
        Returns:
            Dict: Dictionary containing identified companies with their roles.
                {
                    "target_company": {"name": "Company Name", "description": "..."},
                    "advisory_firms": [{"name": "Advisory Firm", "role": "..."}],
                    "project_codename": "Project Name"
                }
        """
        # Get the total number of pages in the document
        total_pages = len(self.pdf_processor.doc)
        
        # Extract text from the first few pages
        first_pages_text = []
        for i in range(min(self.pages_to_analyze, total_pages)):
            first_pages_text.append(self.pdf_processor.extract_page_text(i))
        
        # Extract text from the last few pages
        last_pages_text = []
        if self.pages_to_analyze < total_pages:
            for i in range(total_pages - min(2, total_pages), total_pages):
                last_pages_text.append(self.pdf_processor.extract_page_text(i))
        else: 
            last_pages_text.append("")
        
        # Combine the text with clear section markers
        combined_text = "=== FIRST PAGES ===\n"
        combined_text += "\n".join(first_pages_text)
        combined_text += "\n\n=== LAST PAGES ===\n"
        combined_text += "\n".join(last_pages_text)
        
        # Create prompt for company identification
        prompt = self._build_company_identification_prompt(combined_text, project_name)
        
        # Get LLM response
        response = self.llm_client.client.chat.completions.create(
            model=self.llm_client.model_name,
            messages=[
                {"role": "system", "content": "You are a financial document analysis expert specializing in identifying company roles."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse the JSON response
        if content.startswith("```json"):
            content = content.lstrip("```json").rstrip("```").strip()
        
        try:
            companies_info = json.loads(content)
            return companies_info
        except Exception as e:
            print(f"Error parsing company identification response: {e}")
            # Return a default structure if parsing fails
            return {
                "target_company": {"name": project_name, "description": "Default target company (parsing failed)"},
                "advisory_firms": [],
                "project_codename": project_name
            }
    
    def _build_company_identification_prompt(self, text: str, project_name: str) -> str:
        """
        Build the prompt for company identification.
        
        Args:
            text (str): The text from the first and last few pages of the PDF.
            project_name (str): The codename of the project.
            
        Returns:
            str: The formatted prompt for the LLM.
        """
        prompt = f"""
        You are a financial document analysis expert. Your task is to identify and distinguish between different companies mentioned in this financial document.

        I'll provide you with the text from the first AND last few pages of a financial document. This is from a project codenamed "{project_name}".

        In financial documents, especially teasers or pitch decks:
        1. A target company - The main company being described, analyzed, or offered for investment/acquisition
        2. Advisory firms - Companies like investment banks (e.g., Rothschild, BlackRock, Goldman Sachs) that prepared the document
           - Advisory firms are often mentioned in the FIRST few pages (headers, cover pages) 
           - OR in the LAST few pages (disclaimers, contact information)
        3. Project codename - A code name used for confidentiality (which in this case is "{project_name}")

        Read the text carefully and identify:
        - The target company (the main subject of the document)
        - Any advisory firms (who prepared the document)
        - Confirm if the project codename appears in the text

        ### TEXT ###
        ```
        {text}
        ```

        ### INSTRUCTIONS ###
        Analyze the text and identify the companies mentioned. Pay special attention to:
        - Headers, footers, and cover pages often mention the advisory firm
        - Disclaimers and contact information at the end of the document often mention advisory firms
        - The target company is usually described in detail with financial data, business description, etc.
        - Look for phrases like "prepared for", "prepared by", "advised by", "exclusively mandated by", "contact information", "disclaimer"
        - The project codename might appear in headers, footers, or file names
        - Analyze BOTH the first pages and last pages sections to find all relevant companies

        IMPORTANT: Advisory firms are most likely to be found in the first few pages (headers, logos, introduction) OR in the last few pages (disclaimers, contact information).

        Return your analysis in this JSON format:
        ```json
        {{
            "target_company": {{
                "name": "Name of the main company being analyzed/offered",
                "description": "Brief description of why you believe this is the target company"
            }},
            "advisory_firms": [
                {{
                    "name": "Name of advisory firm",
                    "role": "Their role (e.g., 'Sell-side advisor', 'Buy-side advisor')"
                }}
            ],
            "project_codename": "The identified project codename (typically '{project_name}')"
        }}
        ```

        If you cannot identify any of these elements with confidence, provide your best guess and explain your reasoning.
        Make sure to return ONLY valid JSON without any additional text.
        """
        
        return prompt