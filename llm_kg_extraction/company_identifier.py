import os
import json
import base64 # Added for base64 decoding
from typing import Dict, Optional, List, Any

# Used for google.generativeai SDK if llm_provider is "vertexai"
from google.generativeai.types import GenerationConfig 
# REMOVED: from google.generativeai.types import Part 

# HarmCategory and HarmBlockThreshold are not needed if not overriding safety settings

class CompanyIdentifier:
    """
    Identifies and distinguishes between the main company, advisory firms, and project codenames
    in financial documents. It first attempts textual analysis and falls back to multimodal analysis
    if key entities are not identified.
    Works with Azure OpenAI or Google Vertex AI (via google.generativeai SDK).
    Uses pdf_utils.PDFProcessor for multimodal image data.
    """

    def __init__(self,
                 llm_client_wrapper,
                 llm_provider: str,
                 pdf_processor, # Instance of PDFProcessor from pdf_utils.py
                 azure_deployment_name: Optional[str] = None,
                 pages_to_analyze: int = 3
                ):
        self.llm_client_wrapper = llm_client_wrapper
        self.llm_provider = llm_provider.lower()
        self.pdf_processor = pdf_processor # Expected to be an instance of PDFProcessor
        self.azure_deployment_name = azure_deployment_name
        self.pages_to_analyze = pages_to_analyze
        self.num_last_pages_to_get = 2 

        if self.llm_provider == "azure" and not self.azure_deployment_name:
            raise ValueError("azure_deployment_name is required when llm_provider is 'azure'.")

    def _is_target_company_valid(self, target_info: Optional[Dict]) -> bool:
        if not target_info or not isinstance(target_info, dict):
            return False
        name = target_info.get("name", "")
        description = target_info.get("description", "").lower()
        if not name:
            return False
        failure_keywords = [
            "identification failed", "not accessible", "not available", "0 pages",
            "failed to access pdf", "unsupported llm provider", "llm returned empty content",
            "json parsing failed", "llm response attribute error", "llm call failed",
            "textual analysis failed"
        ]
        for keyword in failure_keywords:
            if keyword in description:
                return False
        return True

    def _are_advisory_firms_valid(self, advisory_firms: Optional[List]) -> bool:
        return advisory_firms and isinstance(advisory_firms, list) and len(advisory_firms) > 0

    def _parse_llm_response(self, content: str, llm_provider_for_log: str) -> Optional[Dict]:
        if not content or not content.strip():
            print(f"LLM ({llm_provider_for_log}) returned empty content.")
            return None
        if content.startswith("```json"):
            content = content.lstrip("```json").rstrip("```").strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM JSON response ({llm_provider_for_log}): {e}")
            print(f"Raw LLM content that failed parsing:\n---\n{content}\n---")
            return None

    def _perform_textual_analysis(self, combined_text: str, project_name: str) -> Optional[Dict]:
        prompt = self._build_company_identification_prompt(combined_text, project_name)
        sdk_client = self.llm_client_wrapper.client
        content = ""
        try:
            if self.llm_provider == "azure":
                messages=[
                    {"role": "system", "content": "You are a financial document analysis expert specializing in identifying company roles from text."},
                    {"role": "user", "content": prompt}
                ]
                response = sdk_client.chat.completions.create(
                    model=self.azure_deployment_name, messages=messages, temperature=0.1, max_tokens=2000
                )
                content = response.choices[0].message.content.strip() if response.choices else ""
            elif self.llm_provider == "vertexai":
                generation_config_dict = {"temperature": 0.1, "max_output_tokens": 2000, "response_mime_type": "application/json"}
                response = sdk_client.generate_content(contents=[prompt], generation_config=generation_config_dict)
                is_blocked_ci = (hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason) or \
                                (not hasattr(response, 'candidates') or not response.candidates) or \
                                (hasattr(response, 'candidates') and response.candidates and response.candidates[0].finish_reason != 1)
                if is_blocked_ci:
                    print(f"Warning (CompanyIdentifier Textual): Vertex AI response may be blocked or incomplete. Details: {getattr(response, 'prompt_feedback', 'N/A')}, FinishReason: {getattr(response.candidates[0], 'finish_reason', 'N/A') if hasattr(response, 'candidates') and response.candidates else 'N/A'}")
                    content = ""
                elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    json_part = response.candidates[0].content.parts[0].text
                    content = json_part.strip() if json_part else ""
                elif hasattr(response, 'text'): content = response.text.strip()
                else:
                    print("Warning (CompanyIdentifier Textual): Vertex AI response structure not as expected for JSON output.")
                    content = ""
            return self._parse_llm_response(content, f"{self.llm_provider} (textual)")
        except AttributeError as ae:
             print(f"AttributeError during textual LLM call ({self.llm_provider}): {ae}")
             return None
        except Exception as e:
            print(f"Error during textual LLM call ({self.llm_provider}): {type(e).__name__} - {e}")
            return None

    def _perform_multimodal_analysis(self, project_name: str, page_indices_to_analyze: List[int]) -> Optional[Dict]:
        """
        Performs company identification using multimodal analysis.
        Uses pdf_utils.PDFProcessor to get image data for Vertex AI.
        """
        if self.llm_provider != "vertexai":
            print(f"Multimodal analysis is currently configured primarily for Vertex AI. Skipping for {self.llm_provider}.")
            return None

        # Check if the pdf_processor has the method from pdf_utils.py
        if not hasattr(self.pdf_processor, 'extract_page_from_pdf'):
            print("Error: PDF Processor does not support 'extract_page_from_pdf' method required for multimodal analysis.")
            return None

        print(f"Attempting multimodal analysis for project: {project_name} using pdf_utils.PDFProcessor")
        sdk_client = self.llm_client_wrapper.client
        
        vertex_parts_list = []
        multimodal_prompt_text = self._build_multimodal_company_identification_prompt(project_name)
        vertex_parts_list.append({'text': multimodal_prompt_text}) # Text prompt part

        for page_idx in page_indices_to_analyze: # page_idx is 0-indexed
            try:
                # self.pdf_processor is an instance of PDFProcessor from pdf_utils.py
                # extract_page_from_pdf expects a 0-indexed page number for doc[page_number] access,
                # as per its implementation in pdf_utils.py
                page_data_dict = self.pdf_processor.extract_page_from_pdf(page_idx)
                
                if page_data_dict and page_data_dict.get("image_base64"):
                    image_base64 = page_data_dict["image_base64"]
                    image_bytes = base64.b64decode(image_base64) # Decode base64 to bytes
                    # Add image part as a dictionary, similar to merged_kg_builder3.py
                    vertex_parts_list.append({'inline_data': {'mime_type': 'image/png', 'data': image_bytes}})
                else:
                    print(f"Warning: Could not extract image_base64 for page index {page_idx} (page {page_idx + 1}).")
            except Exception as e:
                print(f"Error extracting image for page index {page_idx} (page {page_idx + 1}) using pdf_processor: {e}")
        
        if len(vertex_parts_list) <= 1: # Only text prompt, no images successfully added
            print("No images could be prepared for multimodal analysis. Aborting multimodal step.")
            return None

        # Construct the final contents structure for Vertex AI, as seen in merged_kg_builder3.py
        contents_for_llm = [{'parts': vertex_parts_list}]

        try:
            generation_config_dict = {"temperature": 0.1, "max_output_tokens": 2000, "response_mime_type": "application/json"}
            response = sdk_client.generate_content(contents=contents_for_llm, generation_config=generation_config_dict)
            
            content = ""
            is_blocked_ci = (hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason) or \
                            (not hasattr(response, 'candidates') or not response.candidates) or \
                            (hasattr(response, 'candidates') and response.candidates and response.candidates[0].finish_reason != 1)
            
            if is_blocked_ci:
                print(f"Warning (CompanyIdentifier Multimodal): Vertex AI response may be blocked or incomplete. Details: {getattr(response, 'prompt_feedback', 'N/A')}, FinishReason: {getattr(response.candidates[0], 'finish_reason', 'N/A') if hasattr(response, 'candidates') and response.candidates else 'N/A'}")
            elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                json_part = response.candidates[0].content.parts[0].text
                content = json_part.strip() if json_part else ""
            else:
                 print("Warning (CompanyIdentifier Multimodal): Vertex AI response structure not as expected for JSON output.")
            return self._parse_llm_response(content, f"{self.llm_provider} (multimodal)")
        except AttributeError as ae:
             print(f"AttributeError during multimodal LLM call ({self.llm_provider}): {ae}")
             return None
        except Exception as e:
            print(f"Error during multimodal LLM call ({self.llm_provider}): {type(e).__name__} - {e}")
            return None

    def identify_companies(self, project_name: str) -> Dict:
        default_response_template = {
            "target_company": {"name": project_name, "description": "Identification failed or PDF not accessible."},
            "advisory_firms": [], "project_codename": project_name
        }
        current_default_response = default_response_template.copy()
        current_default_response["target_company"] = default_response_template["target_company"].copy()
        current_default_response["advisory_firms"] = list(default_response_template["advisory_firms"])

        if not self.pdf_processor:
            print("Error: PDF Processor not available in CompanyIdentifier.")
            current_default_response["target_company"]["description"] = "PDF Processor not available."
            return current_default_response
        try:
            # Assuming pdf_processor.doc is accessible and gives page count,
            # consistent with how PDFProcessor from pdf_utils.py initializes self.doc
            total_pages = len(self.pdf_processor.doc) 
            if total_pages == 0:
                current_default_response["target_company"]["description"] = "PDF document has 0 pages."
                return current_default_response
        except Exception as e:
            current_default_response["target_company"]["description"] = f"Failed to access PDF: {e}"
            return current_default_response

        first_pages_text_list = [self.pdf_processor.extract_page_text(i) for i in range(min(self.pages_to_analyze, total_pages))]
        first_page_indices_set = set(range(min(self.pages_to_analyze, total_pages)))
        distinct_last_pages_indices = []
        for i in range(1, self.num_last_pages_to_get + 1):
            page_idx = total_pages - i
            if page_idx >= 0 and page_idx not in first_page_indices_set: distinct_last_pages_indices.append(page_idx)
        last_pages_text_list = [self.pdf_processor.extract_page_text(idx) for idx in sorted(list(set(distinct_last_pages_indices)))]
        combined_text = "=== TEXT FROM FIRST PAGES ===\n" + ("\n\n---\nPAGE BREAK\n---\n\n".join(filter(None,first_pages_text_list)) if any(first_pages_text_list) else "No text extracted from first pages.")
        if last_pages_text_list: combined_text += "\n\n=== TEXT FROM LAST PAGES ===\n" + "\n\n---\nPAGE BREAK\n---\n\n".join(filter(None,last_pages_text_list))
        else: combined_text += "\n\n(No distinct text extracted from last pages or document too short)"
        
        companies_info_text = self._perform_textual_analysis(combined_text, project_name)
        final_companies_info = companies_info_text if companies_info_text else current_default_response.copy()
        if not companies_info_text:
             final_companies_info["target_company"] = final_companies_info.get("target_company", {}).copy()
             final_companies_info["target_company"]["name"] = project_name
             final_companies_info["target_company"]["description"] = "Textual analysis failed (e.g., LLM error, parsing issue)."
             final_companies_info["advisory_firms"] = []
             final_companies_info["project_codename"] = project_name

        target_valid_text = self._is_target_company_valid(final_companies_info.get("target_company"))
        advisors_valid_text = self._are_advisory_firms_valid(final_companies_info.get("advisory_firms"))

        if not (target_valid_text and advisors_valid_text):
            print(f"Textual analysis results insufficient. Target valid: {target_valid_text}, Advisors valid: {advisors_valid_text}. Considering multimodal fallback.")
            page_indices_for_multimodal = sorted(list(set(list(range(min(self.pages_to_analyze, total_pages))) + distinct_last_pages_indices)))
            companies_info_multi = self._perform_multimodal_analysis(project_name, page_indices_for_multimodal)
            if companies_info_multi:
                print("Multimodal analysis provided results. Merging.")
                multi_target = companies_info_multi.get("target_company")
                multi_advisors = companies_info_multi.get("advisory_firms")
                if not target_valid_text and self._is_target_company_valid(multi_target):
                    final_companies_info["target_company"] = multi_target
                    print("Updated target company from multimodal analysis.")
                elif target_valid_text and self._is_target_company_valid(multi_target) and \
                     final_companies_info.get("target_company",{}).get("name", "") == project_name and \
                     "identification failed" in final_companies_info.get("target_company",{}).get("description","").lower() :
                    final_companies_info["target_company"] = multi_target
                    print("Replaced default-like textual target with multimodal target.")
                if not advisors_valid_text and self._are_advisory_firms_valid(multi_advisors):
                    final_companies_info["advisory_firms"] = multi_advisors
                    print("Updated advisory firms from multimodal analysis.")
                if companies_info_multi.get("project_codename") == project_name:
                     final_companies_info["project_codename"] = project_name
        final_companies_info["project_codename"] = project_name
        if "target_company" not in final_companies_info or not final_companies_info["target_company"]:
            final_companies_info["target_company"] = {"name": project_name, "description": "Complete identification failure."}
        if "advisory_firms" not in final_companies_info: final_companies_info["advisory_firms"] = []
        return final_companies_info

    def _build_company_identification_prompt(self, text: str, project_name: str) -> str:
        # This prompt remains largely the same for textual analysis
        prompt = f"""
        You are a financial document analysis expert. Your task is to identify and distinguish between different companies mentioned in this financial document's text.
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

        If you cannot confidently identify the target company's actual name from the provided text, use "{project_name}" as the name and clearly state in the "description" field that the specific target company name could not be identified and why (e.g., "Target company name not explicitly found, using project codename as placeholder.").
        If no advisory firms are found, return an empty list for "advisory_firms".
        Make sure to return ONLY a valid JSON object without any additional text before or after the JSON.
        """
        return prompt

    def _build_multimodal_company_identification_prompt(self, project_name: str) -> str:
        # This prompt is for multimodal analysis (images + this text prompt)
        prompt = f"""
        You are a financial document analysis expert. Your task is to identify and distinguish between different companies by analyzing the provided images of document pages along with this textual context.
        This is from a project codenamed "{project_name}".
        The input consists of images from the first few and last few pages of a financial document AND this text prompt.

        In financial documents, especially teasers or pitch decks:
        1. A target company - The main company being described, analyzed, or offered for investment/acquisition. This is the primary subject. Look for prominent company names, descriptions of business in the images.
        2. Advisory firms - Companies like investment banks (e.g., Rothschild, BlackRock, Goldman Sachs) that prepared or are advising on the document/transaction. Look for logos, names in headers, footers, cover pages, or disclaimers/contact sections in the images.
        3. Project codename - A confidential name for the project (in this case, "{project_name}"). This might appear in headers/footers in the images.

        Analyze the provided images carefully and identify:
        - The target company: The main company being discussed or offered.
        - Any advisory firms: Companies involved in preparing the document or advising.
        - Confirm if the project codename "{project_name}" appears.

        ### INSTRUCTIONS ###
        Analyze the visual content of the images (text, logos, layout) to identify the companies. Pay special attention to:
        - The target company is usually the main subject.
        - Advisory firms are often mentioned with their logos or in smaller print in typical sections like the cover, introduction, or final pages. Do not confuse an advisory firm with the target company.
        Be CONCISE in your descriptions.

        Return your analysis in this JSON format:
        ```json
        {{
            "target_company": {{
                "name": "Name of the main company being analyzed/offered",
                "description": "Concise description of the company and its business (e.g., 'A leading tech company specializing in AI solutions.') based on image content."
            }},
            "advisory_firms": [
                {{
                    "name": "Name of advisory firm",
                    "role": "Their concise role (e.g., 'Sell-side advisor', 'Document preparer based on logo/mention')"
                }}
            ],
            "project_codename": "The identified project codename (should be '{project_name}')"
        }}
        ```

        If you cannot confidently identify the target company's actual name from the provided images, use "{project_name}" as the name and clearly state in the "description" field that the specific target company name could not be identified and why (e.g., "Target company name not explicitly found in images, using project codename as placeholder.").
        If no advisory firms are found, return an empty list for "advisory_firms".
        Make sure to return ONLY a valid JSON object without any additional text before or after the JSON.
        """
        return prompt