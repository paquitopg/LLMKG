import os
import base64
from typing import List, Dict, Optional, Any, Union

import google.generativeai as genai
# We will not import Part or Content from .types explicitly
from google.generativeai.types import GenerationConfig # GenerationConfig is usually fine to import

from .base_llm_wrapper import BaseLLMWrapper

class VertexLLM(BaseLLMWrapper):
    """
    Wrapper for Google Large Language Models using the google.generativeai SDK,
    configured to potentially use Vertex AI as the backend via Application Default Credentials.
    This version avoids explicit creation of Part/Content objects, relying on the SDK's
    internal conversion for simpler dict/list structures.
    """

    def __init__(self, model_name: str,
                 project_id: Optional[str] = None,
                 location: Optional[str] = None,
                 **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not self.project_id:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable or project_id argument not set. "
                "This is often used by google.generativeai to correctly route to Vertex AI via ADC."
            )

        self.sdk_model: Optional[genai.GenerativeModel] = None
        try:
            effective_model_name = self.model_name
            if not effective_model_name.startswith("models/") and not effective_model_name.startswith("projects/"):
                 effective_model_name = f"models/{effective_model_name}"

            print(f"Initializing google.generativeai.GenerativeModel with model name: {effective_model_name}")
            print(f" (Relying on ADC with GOOGLE_CLOUD_PROJECT='{self.project_id}' and location='{self.location}' for Vertex AI routing if applicable)")
            
            self.sdk_model = genai.GenerativeModel(model_name=effective_model_name)
            print(f"Successfully initialized google.generativeai.GenerativeModel for {effective_model_name}")

        except Exception as e:
            import traceback
            error_message = (
                f"Failed to initialize google.generativeai.GenerativeModel for model '{self.model_name}' "
                f"(effective name: '{effective_model_name}'). Project: '{self.project_id}'. Error: {type(e).__name__} - {e}"
            )
            print(error_message)
            traceback.print_exc()

    def _prepare_sdk_compatible_parts(self, prompt_parts_data: List[Any]) -> List[Dict[str, Any]]:
        """
        Ensures parts are in a simple dict/str format that genai SDK can handle for 'parts' list.
        The main goal is to ensure inline_data 'data' is bytes.
        """
        sdk_parts_list: List[Dict[str, Any]] = [] # genai can take list of dicts like {'text':...} or {'inline_data':...}
        for part_data in prompt_parts_data:
            if isinstance(part_data, str):
                sdk_parts_list.append({'text': part_data}) # Convert str to {'text': str} for consistency if mixing
            elif isinstance(part_data, dict):
                if 'text' in part_data:
                    sdk_parts_list.append(part_data)
                elif 'inline_data' in part_data:
                    # Ensure data is bytes, as this was a common pattern
                    data_content = part_data['inline_data'].get('data')
                    if isinstance(data_content, str): # If accidentally passed as base64 string
                        try:
                            part_data['inline_data']['data'] = base64.b64decode(data_content)
                        except Exception as e_dec:
                             print(f"Warning: Could not decode base64 string in inline_data: {e_dec}")
                             continue # Skip this part if decoding fails
                    elif not isinstance(data_content, bytes):
                        print(f"Warning: inline_data 'data' is not bytes. Skipping part: {part_data}")
                        continue
                    sdk_parts_list.append(part_data)
                else:
                    print(f"Warning: Unsupported dict structure in prompt_parts_data: {part_data}")
            else:
                print(f"Warning: Unsupported data type in prompt_parts_data: {type(part_data)}")
        return sdk_parts_list

    def generate_content(self, prompt: Union[str, List[Any]], temperature: float = 0.1, response_mime_type: Optional[str] = None, **kwargs) -> Optional[str]:
        if not self.sdk_model:
            print("VertexLLM (google.generativeai): SDK model not initialized.")
            return None

        contents_argument: Any # What we pass to the SDK's 'contents' parameter

        if isinstance(prompt, str):
            # For a simple text prompt, pass it as a list of strings
            contents_argument = [prompt]
        elif isinstance(prompt, list):
            # This is the case for multimodal or complex text prompts from PageLLMProcessor/ContextIdentifier
            # The original KG_builder.py used: contents=[{'parts': vertex_parts_list}]
            # where vertex_parts_list was like: [{'text': '...'}, {'inline_data': {...}}]
            
            # Ensure the inner parts are correctly formatted dicts/strings
            processed_parts = self._prepare_sdk_compatible_parts(prompt)
            if not processed_parts:
                 print("VertexLLM (google.generativeai).generate_content: Prepared parts list is empty after processing.")
                 return None
            contents_argument = [{'parts': processed_parts}] # Maintain the structure from KG_builder.py
        else:
            print(f"VertexLLM (google.generativeai).generate_content: Invalid prompt format. Must be a string or a list of parts. Got: {type(prompt)}")
            return None
        
        config_dict: Dict[str, Any] = {"temperature": float(temperature)}
        if response_mime_type:
            config_dict["response_mime_type"] = response_mime_type
        
        max_output_tokens = kwargs.get('max_output_tokens')
        if max_output_tokens is not None: # Only add if you intend to use it (user said they don't want to)
            config_dict["max_output_tokens"] = int(max_output_tokens)
        
        generation_config_obj = GenerationConfig(**config_dict)
        
        # For debugging:
        # print(f"VertexLLM (google.generativeai) Sending - Contents Arg Type: {type(contents_argument)}")
        # if isinstance(contents_argument, list) and contents_argument and isinstance(contents_argument[0], dict) and 'parts' in contents_argument[0]:
        #     print(f"VertexLLM (google.generativeai) Sending - Inner Parts: {contents_argument[0]['parts']}")
        # else:
        #     print(f"VertexLLM (google.generativeai) Sending - Contents Arg: {contents_argument}")
        # print(f"VertexLLM (google.generativeai) Sending - Generation Config: {generation_config_obj}")

        try:
            response = self.sdk_model.generate_content(
                contents=contents_argument,
                generation_config=generation_config_obj,
            )
            
            if not response.candidates:
                block_reason_msg = "Unknown"
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                    block_reason_msg = response.prompt_feedback.block_reason.name
                print(f"VertexLLM (google.generativeai): Response has no candidates. Prompt feedback block reason: {block_reason_msg}")
                return None

            candidate = response.candidates[0]
            finish_reason_val = candidate.finish_reason
            finish_reason_name = finish_reason_val.name if hasattr(finish_reason_val, 'name') else str(finish_reason_val)

            if finish_reason_name not in ["STOP", "1"]:
                print(f"Warning: VertexLLM (google.generativeai) call finished with reason '{finish_reason_name}'. Response may be incomplete.")
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    for rating in candidate.safety_ratings:
                        print(f"  Safety Rating: {rating.category.name} - {rating.probability.name}") # type: ignore
                if finish_reason_name in ["SAFETY", "3"]:
                     print("  Response blocked due to SAFETY. Returning empty content.")
                     return "" 
            
            # Try response.text first, then specific parts.
            # The genai SDK often populates response.text correctly even for JSON mime_type
            try:
                if response.text:
                    return response.text
            except ValueError as ve: # Catches "ValueError: Accessing response.text when parts are not pure text"
                 print(f"VertexLLM (google.generativeai): ValueError accessing response.text (likely non-text parts): {ve}. Trying parts access.")
            
            if candidate.content and candidate.content.parts:
                if candidate.content.parts[0].text: # Check if text attribute exists
                     return candidate.content.parts[0].text
                else: # If no .text, might be other data types; for this wrapper, we expect text or json string
                     print(f"VertexLLM (google.generativeai): First part of candidate content has no '.text' attribute.")
                     # print(f"First part: {candidate.content.parts[0]}") # For debugging
                     return None
            else:
                print(f"VertexLLM (google.generativeai): Response candidate has no text or usable content parts. Finish Reason: {finish_reason_name}")
                return None

        except Exception as e:
            import traceback
            print(f"An unexpected error occurred with google.generativeai SDK call: {type(e).__name__} - {e}")
            traceback.print_exc()
            return None

    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.1, **kwargs) -> Optional[str]:
        if not self.sdk_model:
            print("VertexLLM (google.generativeai): SDK model not initialized.")
            return None
        
        # Convert OpenAI message format to a list of dicts/strings for self.generate_content
        # This will then be wrapped as contents=[{'parts': current_turn_sdk_parts}] by generate_content
        current_turn_sdk_parts: List[Any] = [] 
        system_prompt_text: Optional[str] = None

        # Handle system prompt: In genai, system instructions are often part of the model or specific turns.
        # For a single generate_content call, we'll prepend it if it's simple text.
        if messages and messages[0].get("role") == "system":
            system_prompt_content = messages[0].get("content")
            if system_prompt_content and isinstance(system_prompt_content, str): # Make sure it's text
                 current_turn_sdk_parts.append({'text': system_prompt_content})
            messages = messages[1:]

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "user":
                if isinstance(content, str):
                    current_turn_sdk_parts.append({'text': content})
                elif isinstance(content, list): # For Azure-style multimodal in user message
                    for item_part in content:
                        if isinstance(item_part, dict):
                            if item_part.get("type") == "text":
                                current_turn_sdk_parts.append({'text': item_part.get("text", "")})
                            elif item_part.get("type") == "image_url":
                                img_url_data = item_part.get("image_url", {}).get("url", "")
                                if img_url_data.startswith("data:image/"):
                                    try:
                                        header, encoded_data = img_url_data.split(",", 1)
                                        mime_type = header.split(":")[1].split(";")[0]
                                        image_bytes = base64.b64decode(encoded_data)
                                        current_turn_sdk_parts.append({'inline_data': {'mime_type': mime_type, 'data': image_bytes}})
                                    except Exception as e_img:
                                        print(f"Error processing base64 image for Vertex chat_completion: {e_img}")
                                else:
                                     print(f"Image URL format not directly supported in this simplified Vertex chat_completion input: {img_url_data}")
            # 'assistant'/'model' roles from history are not directly translated for a single generate_content call
            # unless constructing a full history for models supporting multi-turn Content objects.

        if not current_turn_sdk_parts:
            print("VertexLLM (google.generativeai).chat_completion: No user content parts derived from messages.")
            return None
        
        response_mime_type = kwargs.pop("response_mime_type", None)
        # If you removed max_output_tokens from being passed, this will be None
        max_output_tokens = kwargs.pop("max_output_tokens", None) 
        
        additional_gen_config_kwargs = {}
        if max_output_tokens is not None:
            additional_gen_config_kwargs["max_output_tokens"] = max_output_tokens
        
        # The `prompt` argument to `generate_content` should be the list of part dicts/strings
        return self.generate_content(
            prompt=current_turn_sdk_parts, 
            temperature=temperature,
            response_mime_type=response_mime_type,
            **additional_gen_config_kwargs
        )