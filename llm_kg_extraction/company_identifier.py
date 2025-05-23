import os
import json
from typing import Dict, Optional

# Used for google.generativeai SDK if llm_provider is "vertexai"
from google.generativeai.types import GenerationConfig 
# HarmCategory and HarmBlockThreshold are not needed if not overriding safety settings

class CompanyIdentifier:
    """
    Identifies and distinguishes between the main company, advisory firms, and project codenames
    in financial documents by analyzing the first and last few pages of the PDF.
    Works with Azure OpenAI or Google Vertex AI (via google.generativeai SDK).
    """

    def __init__(self, 
                 llm_client_wrapper, 
                 llm_provider: str, 
                 pdf_processor, 
                 azure_deployment_name: Optional[str] = None, 
                 pages_to_analyze: int = 6
                ):
        """
        Initialize the CompanyIdentifier.
        
        Args:
            llm_client_wrapper: The LLM client wrapper (AzureOpenAIClient or VertexAIClient instance).
            llm_provider (str): The LLM provider ("azure" or "vertexai").
            pdf_processor: The PDF processor to extract text from PDFs.
            azure_deployment_name (str, optional): The deployment name for Azure OpenAI.
                                                   Required if llm_provider is "azure".
            pages_to_analyze (int): Number of pages to analyze from the beginning of the document.
        """
        self.llm_client_wrapper = llm_client_wrapper
        self.llm_provider = llm_provider.lower()
        self.pdf_processor = pdf_processor
        self.azure_deployment_name = azure_deployment_name 
        self.pages_to_analyze = pages_to_analyze

        if self.llm_provider == "azure" and not self.azure_deployment_name:
            raise ValueError("azure_deployment_name is required when llm_provider is 'azure'.")
            
    def identify_companies(self, project_name: str) -> Dict:
        """
        Identify the main company, advisory firms, and project codename from the PDF.
        
        Args:
            project_name (str): The codename of the project.
            
        Returns:
            Dict: Dictionary containing identified companies with their roles.
        """
        default_response = {
            "target_company": {"name": project_name, "description": "Identification failed or PDF not accessible."},
            "advisory_firms": [],
            "project_codename": project_name
        }

        if not self.pdf_processor:
            print("Error: PDF Processor not available in CompanyIdentifier.")
            default_response["target_company"]["description"] = "PDF Processor not available."
            return default_response

        try:
            total_pages = len(self.pdf_processor.doc)
            if total_pages == 0:
                print("Warning: PDF document has 0 pages in CompanyIdentifier.")
                default_response["target_company"]["description"] = "PDF document has 0 pages."
                return default_response
        except Exception as e:
            print(f"Error accessing PDF document for company identification: {e}")
            default_response["target_company"]["description"] = f"Failed to access PDF: {e}"
            return default_response

        first_pages_text_list = []
        for i in range(min(self.pages_to_analyze, total_pages)):
            first_pages_text_list.append(self.pdf_processor.extract_page_text(i))
        
        last_pages_text_list = []
        num_last_pages_to_get = 2 
        first_page_indices = set(range(min(self.pages_to_analyze, total_pages)))
        distinct_last_pages_indices = []
        for i in range(1, num_last_pages_to_get + 1):
            page_idx = total_pages - i
            if page_idx >= 0 and page_idx not in first_page_indices:
                distinct_last_pages_indices.append(page_idx)
        last_pages_text_list = [self.pdf_processor.extract_page_text(idx) for idx in sorted(list(set(distinct_last_pages_indices)))]

        combined_text = "=== TEXT FROM FIRST PAGES ===\n"
        combined_text += "\n\n---\nPAGE BREAK\n---\n\n".join(first_pages_text_list) if first_pages_text_list else "No text extracted from first pages."
        if last_pages_text_list:
            combined_text += "\n\n=== TEXT FROM LAST PAGES ===\n"
            combined_text += "\n\n---\nPAGE BREAK\n---\n\n".join(last_pages_text_list)
        else:
            combined_text += "\n\n(No distinct text extracted from last pages or document too short)"
        
        prompt = self._build_company_identification_prompt(combined_text, project_name)
        
        content = ""
        sdk_client = self.llm_client_wrapper.client 

        try:
            if self.llm_provider == "azure":
                messages=[
                    {"role": "system", "content": "You are a financial document analysis expert specializing in identifying company roles."},
                    {"role": "user", "content": prompt}
                ]
                response = sdk_client.chat.completions.create(
                    model=self.azure_deployment_name, 
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2000 
                )
                content = response.choices[0].message.content.strip()

            elif self.llm_provider == "vertexai": 
                generation_config_dict = {
                    "temperature": 0.1,
                    "max_output_tokens": 2000, 
                    "response_mime_type": "application/json",
                }
                response = sdk_client.generate_content( 
                    contents=[prompt], 
                    generation_config=generation_config_dict
                )
                is_blocked_ci = (hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason) or \
                                (not hasattr(response, 'candidates') or not response.candidates) or \
                                (hasattr(response, 'candidates') and response.candidates and response.candidates[0].finish_reason != 1) # 1=STOP
                
                if is_blocked_ci:
                    print(f"Warning (CompanyIdentifier): Vertex AI response may be blocked or incomplete. Prompt Feedback: {getattr(response, 'prompt_feedback', 'N/A')}, Finish Reason: {getattr(response.candidates[0], 'finish_reason', 'N/A') if hasattr(response, 'candidates') and response.candidates else 'N/A'}")
                    content = ""
                elif hasattr(response, 'text'):
                    content = response.text.strip()
                else:
                    print("Warning (CompanyIdentifier): Vertex AI response does not have a .text attribute.")
                    content = ""
            else:
                print(f"Error: Unsupported llm_provider '{self.llm_provider}' in CompanyIdentifier.")
                default_response["target_company"]["description"] = f"Unsupported LLM provider: {self.llm_provider}"
                return default_response

            if not content.strip():
                print(f"LLM ({self.llm_provider}) returned empty content for company identification.")
                default_response["target_company"]["description"] = "LLM returned empty content for company ID."
                return default_response

            if content.startswith("```json"):
                content = content.lstrip("```json").rstrip("```").strip()
            
            companies_info = json.loads(content)
            return companies_info

        except json.JSONDecodeError as e:
            print(f"Error parsing company identification JSON response ({self.llm_provider}): {e}")
            print(f"Raw LLM content that failed parsing for company ID:\n---\n{content}\n---")
            default_response["target_company"]["description"] = f"Default (JSON parsing failed: {e})"
            return default_response
        except AttributeError as ae: 
             print(f"AttributeError accessing LLM response for company ID ({self.llm_provider}): {ae}")
             if 'response' in locals(): 
                 try: print(f"Response object (or parts): {str(response)[:500]}") # Log snippet
                 except Exception as resp_log_e: print(f"Could not log response object: {resp_log_e}")
             default_response["target_company"]["description"] = f"LLM response attribute error: {ae}"
             return default_response
        except Exception as e:
            print(f"Error during company identification LLM call ({self.llm_provider}): {type(e).__name__} - {e}")
            default_response["target_company"]["description"] = f"Default target company (LLM call failed: {e})"
            return default_response

    def _build_company_identification_prompt(self, text: str, project_name: str) -> str:
        prompt = f"""
        You are a financial document analysis expert. Your task is to identify and distinguish between different companies mentioned in this financial document.

        This is from a project codenamed "{project_name}".
        The provided text is structured with "=== TEXT FROM FIRST PAGES ===" and "=== TEXT FROM LAST PAGES ===".

        In financial documents, especially teasers or pitch decks:
        1. A target company - The main company being described, analyzed, or offered for investment/acquisition. This is the primary subject.
        2. Advisory firms - Companies like investment banks (e.g., Rothschild, BlackRock, Goldman Sachs) that prepared or are advising on the document/transaction. Advisory firms are often mentioned in headers, footers, cover pages, or disclaimers/contact sections on the first or last pages.
        3. Project codename - A confidential name for the project (in this case, "{project_name}").

        Read the provided text carefully from both sections (first and last pages) and identify:
        - The target company: The main company being discussed or offered.
        - Any advisory firms: Companies involved in preparing the document or advising.
        - Confirm if the project codename "{project_name}" appears (it should, as it's given).

        ### TEXT FROM DOCUMENT (FIRST AND LAST PAGES) ###
        ```
        {text}
        ```

        ### INSTRUCTIONS ###
        Analyze the text and identify the companies mentioned. Pay special attention to:
        - The target company is usually the main subject, described in detail with its business, financials, etc. (though this excerpt might only show mentions).
        - The project codename ("{project_name}") might appear in headers or footers.
        - Analyze BOTH the "FIRST PAGES" and "LAST PAGES" sections of the provided text.

        IMPORTANT: Advisory firms are most likely to be found in the first few pages (headers, logos, introduction) OR in the last few pages (disclaimers, contact information). Do not confuse an advisory firm with the target company.
        Be CONCISE in your descriptions.

        Return your analysis in this JSON format:
        ```json
        {{
            "target_company": {{
                "name": "Name of the main company being analyzed/offered",
                "description": "Concise description of the company and its business (e.g., 'A leading tech company specializing in AI solutions.')"
            }},
            "advisory_firms": [
                {{
                    "name": "Name of advisory firm",
                    "role": "Their concise role (e.g., 'Sell-side advisor', 'Document preparer')"
                }}
            ],
            "project_codename": "The identified project codename (should be '{project_name}')"
        }}
        ```

        If you cannot identify any of these elements with confidence from the provided text, provide your best guess and explain your reasoning CONCISELY in the "description" fields.
        Make sure to return ONLY a valid JSON object without any additional text before or after the JSON.
        """
        return prompt
