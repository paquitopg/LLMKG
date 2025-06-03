import json
import base64
from typing import Dict, Optional, List, Any

from .base_context_identifier import BaseContextIdentifier

from llm_integrations.base_llm_wrapper import BaseLLMWrapper
from llm_integrations.azure_llm import AzureLLM # For type checking and specific params
from llm_integrations.vertex_llm import VertexLLM # For type checking

from _1_document_ingestion.pdf_parser import PDFParser # Assuming this is the class name and path


class FinancialTeaserContextIdentifier(BaseContextIdentifier):
    """
    Identifies context specifically for financial teaser documents.
    This includes the target company, advisory firms, and project codename.
    """

    def __init__(self,
                 llm_client: BaseLLMWrapper,
                 project_name: str, # project_name is often used as the codename in teasers
                 pages_to_analyze_first: int = 6, # Number of first pages to analyze
                 pages_to_analyze_last: int = 3 # Number of last pages to analyze
                ):
        """ 
        Initializes the FinancialTeaserContextIdentifier.

        Args:
            llm_client (BaseLLMWrapper): An instance of an LLM client wrapper.
            project_name (str): The overall project name, often used as a fallback or codename.
            pages_to_analyze_first (int): Number of initial pages to focus on.
            pages_to_analyze_last (int): Number of final pages to focus on.
        """
        self.llm_client = llm_client
        self.project_name = project_name
        self.pages_to_analyze_first = pages_to_analyze_first
        self.pages_to_analyze_last = pages_to_analyze_last


    def _is_target_company_valid(self, target_info: Optional[Dict[str, Any]]) -> bool:
        if not target_info or not isinstance(target_info, dict):
            return False
        name = target_info.get("name", "")
        description = target_info.get("description", "").lower()
        if not name or name.lower() == "unknown" or name == self.project_name: # Consider project_name as not a valid identified target name
            return False
        
        failure_keywords = [
            "identification failed", "not accessible", "not available", "0 pages",
            "failed to access pdf", "llm returned empty content",
            "json parsing failed", "llm call failed",
            "textual analysis failed", "could not be identified"
        ]
        for keyword in failure_keywords:
            if keyword in description:
                return False
        # If description contains only the project name or placeholder, it might not be valid.
        if description == f"target company (defaulted to project name: {self.project_name.lower()})":
            return False
        if description == f"target company name not explicitly found, using project codename as placeholder.":
            return False
            
        return True

    def _are_advisory_firms_valid(self, advisory_firms: Optional[List[Dict[str, Any]]]) -> bool:
        return advisory_firms and isinstance(advisory_firms, list) and len(advisory_firms) > 0

    def _parse_llm_response(self, content: Optional[str], llm_provider_for_log: str) -> Optional[Dict[str, Any]]:
        if not content or not content.strip():
            print(f"LLM ({llm_provider_for_log}) returned empty content for context identification.")
            return None
        
        clean_content = content
        if content.startswith("```json"):
            clean_content = content.lstrip("```json").rstrip("```").strip()
        elif content.startswith("```"): # Handle cases like ```\njson...
            clean_content = content.lstrip("```").rstrip("```").strip()
            if clean_content.startswith("json"): # remove leading 'json'
                 clean_content = clean_content.lstrip("json").strip()


        try:
            return json.loads(clean_content)
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM JSON response ({llm_provider_for_log}) for context identification: {e}")
            print(f"Raw LLM content that failed parsing:\n---\n{content[:500]}\n---") # Log first 500 chars
            return None

    def _perform_textual_analysis(self, combined_text: str) -> Optional[Dict[str, Any]]:
        prompt = self._build_company_identification_prompt(combined_text, self.project_name)
        
        # Default LLM call parameters for textual analysis
        # The LLM Wrappers handle provider-specifics of how JSON is requested.
        # For Azure, the system prompt in the wrapper can hint at JSON.
        # For Vertex, response_mime_type="application/json" is key.
        
        llm_response_content: Optional[str] = None
        llm_provider_name = self.llm_client.__class__.__name__

        try:
            if isinstance(self.llm_client, AzureLLM):
                messages = [
                    {"role": "system", "content": "You are a financial document analysis expert specializing in identifying company roles from text. Respond in JSON format."},
                    {"role": "user", "content": prompt}
                ]
                llm_response_content = self.llm_client.chat_completion(
                    messages=messages,
                    temperature=0.1
                )
            elif isinstance(self.llm_client, VertexLLM):
                # VertexLLM's generate_content can take a direct prompt string
                llm_response_content = self.llm_client.generate_content(
                    prompt=prompt, # The prompt already asks for JSON
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            else: # Fallback or other LLM types
                 messages = [
                    {"role": "system", "content": "You are a financial document analysis expert. Respond in JSON format."},
                    {"role": "user", "content": prompt}
                ]
                 llm_response_content = self.llm_client.chat_completion(messages=messages, temperature=0.1)


            return self._parse_llm_response(llm_response_content, f"{llm_provider_name} (textual context ID)")
        except Exception as e:
            print(f"Error during textual LLM call for context ID ({llm_provider_name}): {type(e).__name__} - {e}")
            return None

    def _perform_multimodal_analysis(self, pdf_parser: PDFParser, page_indices_to_analyze: List[int]) -> Optional[Dict[str, Any]]:
        """
        Performs company identification using multimodal analysis.
        """
        llm_provider_name = self.llm_client.__class__.__name__
        print(f"Attempting multimodal context analysis for project: {self.project_name} using {llm_provider_name}")

        multimodal_prompt_text = self._build_multimodal_company_identification_prompt(self.project_name)
        
        llm_input_parts: List[Any] = [{'type': 'text', 'text': multimodal_prompt_text}] # OpenAI/Azure style
        
        # For Vertex, we'll convert this in the wrapper or prepare vertex_parts here
        vertex_specific_parts: List[Any] = [multimodal_prompt_text] # Start with text for Vertex

        images_added = False
        for page_idx in page_indices_to_analyze:
            try:
                # PDFParser's extract_page_from_pdf should return a dict with 'image_base64' and 'text'
                page_data = pdf_parser.extract_page_from_pdf(page_idx) # 0-indexed
                
                if page_data and page_data.get("image_base64"):
                    image_base64_str = page_data["image_base64"]
                    # For AzureLLM via chat_completion expecting OpenAI format
                    llm_input_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64_str}"}
                    })
                    # For VertexLLM via generate_content
                    try:
                        image_bytes = base64.b64decode(image_base64_str)
                        vertex_specific_parts.append({'inline_data': {'mime_type': 'image/png', 'data': image_bytes}})
                        images_added = True
                    except Exception as e_dec:
                        print(f"Warning: Could not decode base64 image for page index {page_idx}: {e_dec}")

                else:
                    print(f"Warning: Could not extract image_base64 for page index {page_idx} (page {page_idx + 1}).")
            except Exception as e:
                print(f"Error extracting image for page index {page_idx} (page {page_idx + 1}): {e}")
        
        if not images_added and isinstance(self.llm_client, VertexLLM): # Only proceed if images were actually added for Vertex multimodal
             print("No images could be prepared for Vertex multimodal analysis. Aborting multimodal step for Vertex.")
             return None
        if len(llm_input_parts) <= 1 and isinstance(self.llm_client, AzureLLM): # Only text prompt
             print("No images could be added for Azure multimodal analysis. Aborting multimodal step for Azure.")
             return None


        llm_response_content: Optional[str] = None
        try:
            if isinstance(self.llm_client, AzureLLM):
                # AzureLLM's chat_completion expects a list of messages.
                # The 'user' content can be a list of parts (text & image_url).
                messages = [
                    {"role": "system", "content": "You are a financial document analysis expert. Analyze the provided text and images to identify company roles. Respond in JSON format."},
                    {"role": "user", "content": llm_input_parts}
                ]
                llm_response_content = self.llm_client.chat_completion(
                    messages=messages,
                    temperature=0.1
                )
            elif isinstance(self.llm_client, VertexLLM):
                # VertexLLM's generate_content can take the list of vertex_specific_parts directly.
                # The wrapper handles structuring it as {'parts': vertex_specific_parts}
                llm_response_content = self.llm_client.generate_content(
                    prompt=vertex_specific_parts, # This list contains text and inline_data dicts
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            else:
                print(f"Multimodal analysis for context ID not specifically implemented for LLM type: {llm_provider_name}")
                return None # Or attempt a text-only fallback if desired

            return self._parse_llm_response(llm_response_content, f"{llm_provider_name} (multimodal context ID)")
        except Exception as e:
            print(f"Error during multimodal LLM call for context ID ({llm_provider_name}): {type(e).__name__} - {e}")
            return None


    def identify_context(self,
                         document_path: Optional[str] = None,
                         document_content_parts: Optional[Dict[str, Any]] = None, # Not used in this impl, relies on path
                         doc_type_hint: Optional[str] = "financial_teaser"
                        ) -> Dict[str, Any]:
        """
        Identifies context for a financial teaser document.

        Args:
            document_path (Optional[str]): Path to the PDF document.
            document_content_parts (Optional[Dict[str, Any]]): Pre-extracted parts (not directly used, uses path).
            doc_type_hint (Optional[str]): Expected to be "financial_teaser".

        Returns:
            Dict[str, Any]: Identified context including target company, advisors, and codename.
        """
        if not document_path:
            raise ValueError("document_path is required for FinancialTeaserContextIdentifier.")

        pdf_parser = PDFParser(pdf_path=document_path) # Initialize parser with the path
        
        # Default response structure, project_name is used as codename and fallback target name
        identified_info: Dict[str, Any] = {
            "identified_document_type": "financial_teaser", # or doc_type_hint
            "target_company": {"name": self.project_name, "description": "Identification pending or failed."},
            "advisory_firms": [],
            "project_codename": self.project_name
        }

        try:
            total_pages = len(pdf_parser.doc) # PyMuPDF document
            if total_pages == 0:
                identified_info["target_company"]["description"] = "PDF document has 0 pages."
                return identified_info
        except Exception as e:
            identified_info["target_company"]["description"] = f"Failed to access PDF or get page count: {e}"
            return identified_info

        # Prepare text from first and last pages for textual analysis
        first_pages_text_list = [pdf_parser.extract_page_text(i) for i in range(min(self.pages_to_analyze_first, total_pages))]
        
        # Determine indices for distinct last pages
        first_page_indices_set = set(range(min(self.pages_to_analyze_first, total_pages)))
        distinct_last_pages_indices: List[int] = []
        for i in range(1, self.pages_to_analyze_last + 1):
            page_idx = total_pages - i
            if page_idx >= 0 and page_idx not in first_page_indices_set:
                distinct_last_pages_indices.append(page_idx)
        
        last_pages_text_list = [pdf_parser.extract_page_text(idx) for idx in sorted(list(set(distinct_last_pages_indices)))]
        
        combined_text = "=== TEXT FROM FIRST PAGES ===\n" + \
                        ("\n\n---\nPAGE BREAK\n---\n\n".join(filter(None,first_pages_text_list)) if any(first_pages_text_list) else "No text extracted from first pages.")
        if last_pages_text_list:
            combined_text += "\n\n=== TEXT FROM LAST PAGES ===\n" + \
                             "\n\n---\nPAGE BREAK\n---\n\n".join(filter(None,last_pages_text_list))
        else:
            combined_text += "\n\n(No distinct text extracted from last pages or document too short)"
        
        # Perform textual analysis
        companies_info_text = self._perform_textual_analysis(combined_text)

        if companies_info_text:
            identified_info.update(companies_info_text) # Update with results from textual analysis
            # Ensure project_codename is always set to self.project_name as per original logic
            identified_info["project_codename"] = self.project_name
        else: # Textual analysis failed or returned None
            identified_info["target_company"]["description"] = "Textual analysis for company identification failed."

        # Validate results from textual analysis
        target_valid_text = self._is_target_company_valid(identified_info.get("target_company"))
        advisors_valid_text = self._are_advisory_firms_valid(identified_info.get("advisory_firms"))

        # If textual results are insufficient, consider multimodal fallback
        # (Currently, the original CompanyIdentifier has stronger conditions for multimodal,
        # e.g. if self.llm_provider == "vertexai". Here we make it dependent on instance type of llm_client)
        if not (target_valid_text and advisors_valid_text) and isinstance(self.llm_client, (VertexLLM, AzureLLM)): # Add AzureLLM if its multimodal is robust
            print(f"Textual analysis results for context considered insufficient. Target valid: {target_valid_text}, Advisors valid: {advisors_valid_text}. Attempting multimodal fallback.")
            
            page_indices_for_multimodal = sorted(list(set(list(range(min(self.pages_to_analyze_first, total_pages))) + distinct_last_pages_indices)))
            
            if page_indices_for_multimodal:
                companies_info_multi = self._perform_multimodal_analysis(pdf_parser, page_indices_for_multimodal)
                
                if companies_info_multi:
                    print("Multimodal context analysis provided results. Merging with previous findings.")
                    multi_target = companies_info_multi.get("target_company")
                    multi_advisors = companies_info_multi.get("advisory_firms")

                    # If textual target was not valid and multimodal one is, use multimodal
                    if not target_valid_text and self._is_target_company_valid(multi_target):
                        identified_info["target_company"] = multi_target
                        print("Updated target company from multimodal context analysis.")
                    # If textual target was just a placeholder (project_name) and multimodal is better
                    elif target_valid_text and self._is_target_company_valid(multi_target) and \
                         identified_info.get("target_company",{}).get("name", "") == self.project_name :
                        identified_info["target_company"] = multi_target
                        print("Replaced placeholder textual target with multimodal target from context analysis.")
                    
                    # If textual advisors were not valid and multimodal ones are, use multimodal
                    if not advisors_valid_text and self._are_advisory_firms_valid(multi_advisors):
                        identified_info["advisory_firms"] = multi_advisors
                        print("Updated advisory firms from multimodal context analysis.")
                    
                    # Ensure project_codename is always self.project_name
                    identified_info["project_codename"] = self.project_name
            else:
                print("No pages selected for multimodal analysis.")
        
        # Final cleanup for target company if still not properly identified
        if not self._is_target_company_valid(identified_info.get("target_company")):
            identified_info["target_company"]["name"] = self.project_name
            if "identification failed" not in identified_info["target_company"]["description"].lower() and \
               "0 pages" not in identified_info["target_company"]["description"].lower():
                 identified_info["target_company"]["description"] = f"Target company name not reliably identified; using project codename '{self.project_name}' as placeholder."
        
        # Print the information 
        print(f"Identified context for project '{self.project_name}':")
        print(f"  Target Company: {identified_info['target_company']['name']}")
        print(f"  Description: {identified_info['target_company']['description']}")
        if identified_info["advisory_firms"]:
            print(f"  Advisory Firms: {', '.join([firm['name'] for firm in identified_info['advisory_firms']])}")
        else:
            print("  Advisory Firms: None identified.")
        print(f"  Project Codename: {identified_info['project_codename']}")

        return identified_info


    def _build_company_identification_prompt(self, text_content: str, project_codename: str) -> str:
        # This prompt is for textual analysis to identify companies in a financial teaser.
        return f"""
        You are a financial document analysis expert. Your task is to identify and distinguish between different companies mentioned in the provided financial document text.
        This document relates to a project codenamed "{project_codename}".
        The provided text is structured with "=== TEXT FROM FIRST PAGES ===" and "=== TEXT FROM LAST PAGES ===".

        In financial documents like teasers or pitch decks:
        1.  A "target company": The primary company being described, analyzed, or offered for investment/acquisition. This is the main subject.
        2.  "Advisory firms": Companies (e.g., investment banks like Goldman Sachs, Rothschild & Co) that prepared or are advising on the transaction/document. They are often mentioned in headers, footers, cover pages, or disclaimers on the first or last pages.
        3.  "Project codename": A confidential name for the project (in this case, "{project_codename}"). This is NOT the actual name of the target company.

        Please analyze the following text from the document's first and last pages:
        ### DOCUMENT TEXT EXCERPT ###
        ```
        {text_content}
        ```

        ### INSTRUCTIONS ###
        Based ONLY on the text provided above:
        1.  Identify the full name of the "target company". This is the company being sold or profiled.
        2.  Provide a concise description of what the target company does, if mentioned.
        3.  Identify any "advisory firms" involved. List their names and their apparent role (e.g., "Sell-side advisor", "Document preparer").
        4.  Confirm the "project_codename" (it should be "{project_codename}").

        IMPORTANT:
        - The "target company" is the main subject of the teaser, not the firms preparing it.
        - "Advisory firms" are typically found on the cover, in disclaimers, or contact sections. Do not confuse them with the target.
        - The "project_codename" ("{project_codename}") is just a reference for the deal, not a company name itself.

        Return your analysis ONLY in the following JSON format. Do not include any text before or after the JSON object:
        ```json
        {{
            "target_company": {{
                "name": "Full Name of the Target Company",
                "description": "Concise description of the target company's business (e.g., 'A leading SaaS provider for the healthcare industry.'). If not found, state 'Description not found.'"
            }},
            "advisory_firms": [
                {{
                    "name": "Name of Advisory Firm 1",
                    "role": "Their concise role (e.g., 'Financial advisor to the target company', 'Prepared this document')"
                }}
            ],
            "project_codename": "{project_codename}"
        }}
        ```
        If you cannot confidently identify the target company's actual name from the text, use "{project_codename}" as the target company name and state in its description: "Target company name not explicitly found, using project codename as placeholder."
        If no advisory firms are found, return an empty list for "advisory_firms".
        Ensure the output is a single, valid JSON object.
        """

    def _build_multimodal_company_identification_prompt(self, project_codename: str) -> str:
        # This prompt is for multimodal analysis (images + this text prompt)
        return f"""
        You are a financial document analysis expert. Your task is to identify and distinguish between different company roles by analyzing the provided images of document pages.
        These images are from a project codenamed "{project_codename}".

        In financial documents (like teasers), visual cues on pages (especially first/last) help identify:
        1.  The "target company": The main company being offered or analyzed. Look for prominent logos, company names in large fonts, or descriptions of its business.
        2.  "Advisory firms": Investment banks or consulting firms involved (e.g., Goldman Sachs, Deloitte). Look for their logos or names, often in headers, footers, on the cover page, or in disclaimer/contact sections.
        3.  The "project_codename" (e.g., "{project_codename}"): This might appear in headers/footers. It's a deal identifier, not the target company itself.

        ### INSTRUCTIONS ###
        Analyze the visual content of the provided images (text visible in images, logos, layout) to identify:
        - The "target company": The primary subject of the document.
        - Any "advisory firms": Firms assisting with the transaction or document preparation.
        - The "project_codename": Should be "{project_codename}".

        IMPORTANT:
        - Do not confuse advisory firms (often with their own logos in less prominent places) with the target company (usually the main focus of the page).
        - The project codename is NOT the target company.

        Return your analysis ONLY in the following JSON format. Do not include any text before or after the JSON object:
        ```json
        {{
            "target_company": {{
                "name": "Name of the Main Target Company (from images)",
                "description": "Concise description of the company's business if evident from images (e.g., 'Visuals suggest it is a manufacturing company.'). If not clear, state 'Business type not evident from images.'"
            }},
            "advisory_firms": [
                {{
                    "name": "Name of Advisory Firm (from logo/text in image)",
                    "role": "Their role if apparent (e.g., 'Appears as financial advisor based on placement', 'Logo present on cover')"
                }}
            ],
            "project_codename": "{project_codename}"
        }}
        ```
        If the target company's name is not clearly identifiable from the images, use "{project_codename}" as the name and note in the description: "Target company name not clearly identified in images, using project codename as placeholder."
        If no advisory firms are identifiable from the images, return an empty list for "advisory_firms".
        Ensure the output is a single, valid JSON object.
        """