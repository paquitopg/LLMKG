# File: llm_kg_extraction/_1_document_ingestion/document_classifier.py

from typing import List, Optional, Dict, Any, Union
import base64
import json

from llm_integrations.base_llm_wrapper import BaseLLMWrapper
from llm_integrations.vertex_llm import VertexLLM
from llm_integrations.azure_llm import AzureLLM
from _1_document_ingestion.pdf_parser import PDFParser # Ensure PDFParser is imported for type hinting


class DocumentClassifier:
    """
    Classifies a document into predefined categories using an LLM,
    with support for N-page text/multimodal input.
    It always generates and returns a summary of the document,
    which can be used as context for knowledge extraction.
    """

    def __init__(self, llm_client: BaseLLMWrapper, categories: List[str],
                 summary_llm_client: Optional[BaseLLMWrapper] = None):
        """
        Initializes the DocumentClassifier.

        Args:
            llm_client (BaseLLMWrapper): LLM client for primary classification (likely smaller/faster).
            categories (List[str]): Predefined document categories.
            summary_llm_client (Optional[BaseLLMWrapper]): Optional LLM client for summarization.
                                                            If None, uses llm_client for summarization too.
        """
        if not llm_client:
            raise ValueError("Primary LLM client must be provided.")
        if not categories:
            raise ValueError("A list of categories must be provided.")
        
        self.llm_client = llm_client
        # Use a potentially different (larger/more capable) model for summarization if specified
        self.summary_llm_client = summary_llm_client if summary_llm_client else llm_client
        self.categories = [category.lower() for category in categories]
        self.default_category = "unknown_document_type"
        self.classification_temperature = 0.1 # Lower temperature for more deterministic classification
        self.summary_temperature = 0.3 # Higher temperature for more creative/general summaries

    def _build_classification_prompt(self, text_content: str, input_source_description: str) -> str:
        """
        Builds a prompt for text-based document classification.
        """
        category_list_str = ", ".join(f"'{cat}'" for cat in self.categories)
        # Truncate based on typical LLM context limits for classification
        truncated_text = text_content[:15000] # Approx 3-4k tokens. Adjust based on model capability.
        prompt = f"""
Your task is to classify a document based on its content.
Content source: {input_source_description}.
Predefined Categories: {category_list_str}.

Instructions:
1. Analyze the provided "Text Content to Analyze".
2. Choose the single most appropriate category from the "Predefined Categories" list.
3. Respond with ONLY the chosen category name.
4. If the document does not clearly fit any category or you are highly uncertain, respond with '{self.default_category}'.

Text Content to Analyze:
---
{truncated_text}
---

Classification (respond with only one category name):
"""
        return prompt

    def _build_multimodal_classification_prompt(self, text_content: Optional[str] = None) -> str:
        """
        Builds the textual part of a prompt for multimodal document classification.
        """
        category_list_str = ", ".join(f"'{cat}'" for cat in self.categories)
        truncated_text_content = text_content[:8000] if text_content else "" # Shorter text for multimodal
        
        base_prompt_text = f"""
Your task is to classify a document based on the provided page content (text and image(s)).
Predefined Categories: {category_list_str}.

Instructions:
1. Analyze the provided "Text Content on Page(s)" and the associated image(s).
2. Choose the single most appropriate category from the "Predefined Categories" list.
3. Respond with ONLY the chosen category name.
4. If the document does not clearly fit any category or you are highly uncertain, respond with '{self.default_category}'.
"""
        if truncated_text_content:
            return f"{base_prompt_text}\nText Content on Page(s) (use with image(s)):\n---\n{truncated_text_content}\n---\n\nClassification:"
        else:
            return f"{base_prompt_text}\n\nAnalyze the image(s) to determine the classification:"

    def _parse_llm_response(self, response_text: Optional[str]) -> str:
        """
        Parses the LLM's response to extract a valid category.
        """
        if not response_text:
            return self.default_category
        # Clean the response: lowercase, strip whitespace, remove potential quotes, take first line
        cleaned_response = response_text.strip().lower().replace("'", "").replace('"', '').splitlines()[0]
        
        if cleaned_response in self.categories:
            return cleaned_response
        
        # Try to find a category as a substring in case of more verbose LLM responses
        for category in self.categories:
            if category in cleaned_response:
                # print(f"Classifier: Matched category '{category}' from a verbose response: '{response_text}'")
                return category
        
        # print(f"Classifier: LLM response '{response_text}' did not directly match known categories. Defaulting.")
        return self.default_category

    def _call_llm_for_classification(self, prompt_parts: Union[str, List[Any]], is_multimodal: bool = False) -> str:
        """Internal helper to make the LLM call and parse response."""
        llm_response: Optional[str] = None
        try:
            if is_multimodal:
                if isinstance(self.llm_client, VertexLLM):
                    # prompt_parts is expected to be List[Union[str, Dict]] for Vertex generate_content
                    # e.g., [{'text': 'prompt'}, {'inline_data': {'mime_type': ..., 'data': ...}}]
                    llm_response = self.llm_client.generate_content(prompt=prompt_parts, temperature=self.classification_temperature)
                elif isinstance(self.llm_client, AzureLLM):
                    # prompt_parts is expected to be List[Dict] for Azure user content
                    # e.g., [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
                    messages = [
                        {"role": "system", "content": "You are a document classification assistant analyzing text and/or images."},
                        {"role": "user", "content": prompt_parts}
                    ]
                    llm_response = self.llm_client.chat_completion(messages=messages, temperature=self.classification_temperature)
                else: # Fallback for other BaseLLMWrapper types (might not support multimodal parts well)
                    text_prompt_for_fallback = ""
                    if isinstance(prompt_parts, list): # Try to extract text part
                        for part in prompt_parts:
                            if isinstance(part, str): text_prompt_for_fallback = part; break
                            if isinstance(part, dict) and part.get("type") == "text": text_prompt_for_fallback = part.get("text", ""); break
                    elif isinstance(prompt_parts, str):
                        text_prompt_for_fallback = prompt_parts
                    if not text_prompt_for_fallback: 
                        print("Warning: No text part found for multimodal LLM fallback.")
                        return self.default_category

                    print("Warning: Multimodal classification for generic LLM client using only text parts.")
                    messages = [{"role": "system", "content": "You are a document classification assistant."}, {"role": "user", "content": text_prompt_for_fallback}]
                    llm_response = self.llm_client.chat_completion(messages=messages, temperature=self.classification_temperature)
            else: # Text-only classification
                # prompt_parts is expected to be a string (the full prompt)
                messages = [
                    {"role": "system", "content": "You are a document classification assistant."},
                    {"role": "user", "content": prompt_parts}
                ]
                llm_response = self.llm_client.chat_completion(messages=messages, temperature=self.classification_temperature)
        except Exception as e:
            print(f"Error during LLM call for classification: {e}")
            return self.default_category
            
        return self._parse_llm_response(llm_response)

    def _generate_document_summary(self, text_content: str, max_length_chars: int = 20000) -> Optional[Dict[str, Any]]:
        """
        Generates a summary of the text_content using the summary_llm_client.
        
        Returns:
            Optional[Dict[str, Any]]: Dictionary with 'summary' and optional 'main_entity' fields,
                                    or None if generation fails.
        """
        print("DocumentClassifier: Generating document summary...")
        
        prompt = f"""
        Please provide a concise and informative summary of the key topics, purpose, and overall nature of the following document text.
        The summary should be dense and present the main subjects or actors of the document. 
        IT WILL BE USED AS CONTEXT TO GUIDE KNOWLEDGE EXTRACTION, SO ONLY THE MAIN POINTS ARE NEEDED.

        I also need to know the main entity or subject of the document, if applicable.
        For instance, if the document is about a product specification, the main entity would be the product name.
        If the document is a legal contract, the main entity could be the parties involved or the contract type.

        IMPORTANT: You must respond with ONLY a valid JSON object in the following format:
        {{
            "summary": "Your detailed summary here",
            "main_entity": "Main entity name (if applicable, otherwise null)"
        }}

        Do not include any other text, explanations, or markdown formatting. Only return the JSON object.

        Document Text:
        ---
        {text_content[:max_length_chars]} 
        ---
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            raw_response = self.summary_llm_client.chat_completion(
                messages=messages, 
                temperature=self.summary_temperature
            )
            
            if not raw_response or not raw_response.strip():
                print("DocumentClassifier: Generated summary was empty or whitespace.")
                return None
            
            # Clean the response - remove potential markdown formatting
            cleaned_response = raw_response.strip()
            
            # Remove markdown json blocks if present
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]   # Remove ```
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]  # Remove trailing ```
            
            cleaned_response = cleaned_response.strip()
            
            # Try to parse as JSON
            try:
                summary_dict = json.loads(cleaned_response)
                
                # Validate the structure
                if not isinstance(summary_dict, dict):
                    print("DocumentClassifier: Response is not a dictionary.")
                    return self._create_fallback_summary(raw_response)
                
                if "summary" not in summary_dict:
                    print("DocumentClassifier: Response missing required 'summary' field.")
                    return self._create_fallback_summary(raw_response)
                
                # Ensure main_entity is present (can be null)
                if "main_entity" not in summary_dict:
                    summary_dict["main_entity"] = None
                
                print(f"DocumentClassifier: Summary generated successfully: {summary_dict['summary']}...")
                if summary_dict.get("main_entity"):
                    print(f"DocumentClassifier: Main entity identified: {summary_dict['main_entity']}")
                
                return summary_dict
                
            except json.JSONDecodeError as json_error:
                print(f"DocumentClassifier: Failed to parse JSON response: {json_error}")
                print(f"Raw response (first 200 chars): {raw_response[:200]}...")
                return self._create_fallback_summary(raw_response)
                
        except Exception as e:
            print(f"Error generating document summary: {e}")
            return None

    def classify_document(self,
                          pdf_parser: PDFParser,
                          num_pages_for_classification_text: int = 4,
                          num_pages_for_classification_image: int = 4,
                          num_pages_for_summary: int = 10
                         ) -> Dict[str, str]: # Return type changed to ensure string for summary
        """
        Attempts classification with priority (ToC -> N-page Text -> N-page Multimodal).
        Always generates and returns a document summary.

        Args:
            pdf_parser (PDFParser): An instance of the PDF parser for the document.
            num_pages_for_classification_text (int): Number of first pages to use for text classification.
            num_pages_for_classification_image (int): Number of first pages to use for multimodal classification.
            num_pages_for_summary (int): Number of pages to use for summary generation.

        Returns:
            Dict[str, str]: 
                - "identified_doc_type": The classified document type (or default_category).
                - "document_summary": A generated summary of the document.
        """
        identified_category = self.default_category
        
        # --- Classification Attempts ---

        # 1. Try Table of Contents (ToC) text first
        toc_text = pdf_parser.get_toc_text()
        if toc_text and toc_text.strip():
            print("DocumentClassifier: Attempting classification using Table of Contents.")
            prompt = self._build_classification_prompt(toc_text, "Table of Contents")
            classified_by_toc = self._call_llm_for_classification(prompt, is_multimodal=False)
            if classified_by_toc != self.default_category:
                identified_category = classified_by_toc
                print(f"DocumentClassifier: Classified as '{identified_category}' using ToC.")
            else:
                print("DocumentClassifier: Classification with ToC was insufficient.")

        # 2. If ToC failed or not available and category still default, try First N Pages Text
        if identified_category == self.default_category:
            first_n_pages_text = pdf_parser.extract_text_from_first_n_pages(num_pages_for_classification_text)
            if first_n_pages_text and first_n_pages_text.strip():
                print(f"DocumentClassifier: Attempting classification using text from first {num_pages_for_classification_text} pages.")
                prompt = self._build_classification_prompt(first_n_pages_text, f"First {num_pages_for_classification_text} Pages Text")
                classified_by_text = self._call_llm_for_classification(prompt, is_multimodal=False)
                if classified_by_text != self.default_category:
                    identified_category = classified_by_text
                    print(f"DocumentClassifier: Classified as '{identified_category}' using first N pages text.")
                else:
                    print("DocumentClassifier: Classification with first N pages text was insufficient.")
            
        # 3. If text classification was insufficient and images are available, try Multimodal
        if identified_category == self.default_category:
            first_n_pages_text_for_multimodal = pdf_parser.extract_text_from_first_n_pages(num_pages_for_classification_image) # Get text for multimodal if not already
            if isinstance(self.llm_client, (VertexLLM, AzureLLM)): # Check for multimodal capability
                first_n_pages_images_b64 = pdf_parser.extract_images_from_first_n_pages_base64(num_pages_for_classification_image)
                if first_n_pages_images_b64:
                    print(f"DocumentClassifier: Attempting multimodal classification using first {num_pages_for_classification_image} page images (and text if available).")
                    
                    llm_input_parts: List[Any] = []
                    prompt_text_part = self._build_multimodal_classification_prompt(first_n_pages_text_for_multimodal)
                    
                    if isinstance(self.llm_client, VertexLLM):
                        llm_input_parts.append({'text': prompt_text_part})
                        for img_b64 in first_n_pages_images_b64:
                            try:
                                image_bytes = base64.b64decode(img_b64)
                                llm_input_parts.append({'inline_data': {'mime_type': 'image/png', 'data': image_bytes}})
                            except Exception as e:
                                print(f"Warning: Could not decode base64 image for Vertex AI: {e}")
                                continue
                    elif isinstance(self.llm_client, AzureLLM):
                        llm_input_parts.append({"type": "text", "text": prompt_text_part})
                        for img_b64 in first_n_pages_images_b64:
                            llm_input_parts.append(
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                            )
                    
                    if len(llm_input_parts) > 1 or (len(llm_input_parts) == 1 and any('image' in str(part) for part in llm_input_parts)): # Check if images were actually added
                        classified_by_multimodal = self._call_llm_for_classification(llm_input_parts, is_multimodal=True)
                        if classified_by_multimodal != self.default_category:
                            identified_category = classified_by_multimodal
                            print(f"DocumentClassifier: Classified as '{identified_category}' using multimodal analysis.")
                        else:
                            print("DocumentClassifier: Multimodal classification was insufficient.")
                    else:
                        print("DocumentClassifier: No valid image parts prepared for multimodal call, or LLM type not specialized for multimodal.")
                else:
                    print(f"DocumentClassifier: No images extracted from first {num_pages_for_classification_image} pages for multimodal classification.")
            else:
                print(f"DocumentClassifier: LLM client type {type(self.llm_client).__name__} does not support multimodal classification.")

        # --- Always Generate Summary ---
        print(f"DocumentClassifier: Classification process finished. Identified type: '{identified_category}'.")
        print(f"DocumentClassifier: Proceeding to generate document summary from first {num_pages_for_summary} pages.")
        
        text_for_summary = pdf_parser.extract_text_from_first_n_pages(num_pages_for_summary)
        document_summary = self._generate_document_summary(text_for_summary)
        
        # Ensure summary is a string, even if None was returned
        if document_summary is None:
            document_summary = "" 

        return {
            "identified_doc_type": identified_category,
            "document_summary": document_summary
        }