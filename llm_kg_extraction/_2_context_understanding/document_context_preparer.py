from typing import Dict, Any, Optional
from ontology_management.ontology_loader import PEKGOntology
import json # Ensure json is imported

from typing import Dict, Any, Optional
from ontology_management.ontology_loader import PEKGOntology

class DocumentContextPreparer:
    """
    Prepares the document-level context for knowledge graph extraction.
    This includes the identified document type, a summary of the document,
    and the full ontology schema. This class does NOT call an LLM.
    """

    def __init__(self, ontology: PEKGOntology, use_ontology: bool = True):
        """
        Initializes the DocumentContextPreparer.

        Args:
            ontology (PEKGOntology): The loaded ontology for the project.
            use_ontology (bool): Flag indicating whether to use the ontology in context preparation.
        """
        self.ontology = ontology
        self.use_ontology = use_ontology

    def prepare_context(self, 
                        identified_doc_type: str,
                        summary: Dict[str, Any],
                        **kwargs) -> Dict[str, Any]:
        """
        Prepares and returns the context for knowledge graph extraction.
        This context combines the document type, its summary, and the ontology.

        Args:
            identified_doc_type (str): The classified document type (or 'unknown_document_type').
            document_summary (Dict[str, Any]): A summary of the document content. It has the following structure:
                {
                    "summary": str,
                    "main_entity": Optional[str]
                }

        Returns:
            Dict[str, Any]: A dictionary containing the document-level context for KG extraction.
            This dictionary will be passed to the PageLLMProcessor.
        """
        print(f"DocumentContextPreparer: Preparing context for type '{identified_doc_type}'.")

        if self.use_ontology:
            full_ontology_schema = self.ontology.format_for_prompt()
            document_context_info = {
                "identified_document_type": identified_doc_type,
                "document_summary": summary["summary"],
                "main_entity": summary["main_entity"] if "main_entity" in summary else None,
                "ontology_schema": full_ontology_schema,
                # Add any other document-level context data here if needed in the future
            }

        else:
            document_context_info = {
                "identified_document_type": identified_doc_type,
                "document_summary": summary["summary"],
                "main_entity": summary["main_entity"] if "main_entity" in summary else None,
                "ontology_schema": "",
            }
        
        print("DocumentContextPreparer: Context prepared.")
        return document_context_info