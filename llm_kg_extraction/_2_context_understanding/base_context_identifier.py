from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

# Assuming your PDF parser is in '1_document_ingestion.pdf_parser.pdf_parser'
# and it has a class named PDFParser. Adjust the import path if necessary.
# from .._1_document_ingestion.pdf_parser import PDFParser # Example relative import

class BaseContextIdentifier(ABC):
    """
    Abstract base class for document context identifiers.

    The purpose of a context identifier is to analyze a document (or its initial parts)
    to determine its type (if not already known) and extract key high-level
    entities and their roles relevant to that document type. This information
    (e.g., target company in a teaser, parties in a contract) is then used to
    provide better context to the LLM during the detailed knowledge graph extraction phase.
    """

    @abstractmethod
    def identify_context(
        self,
        # pdf_parser_instance: PDFParser, # Option 1: Pass a PDFParser instance
        document_path: Optional[str] = None, # Option 2: Pass a path
        document_content_parts: Optional[Dict[str, Any]] = None, # Option 3: Pass pre-extracted content
        doc_type_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Identifies the high-level context of a given document.

        This method should be implemented by concrete subclasses to extract
        relevant contextual information based on the document type.

        Args:
            document_path (Optional[str]): The file path to the document.
                                           Implementations might use this to initialize their own PDF parser
                                           or read the document.
            document_content_parts (Optional[Dict[str, Any]]): Pre-extracted document content.
                                           This could include things like text from the first few pages,
                                           images, or other metadata.
                                           Example: {"first_page_text": "...", "last_page_text": "..."}
            doc_type_hint (Optional[str]): An optional hint about the document's type,
                                           which might guide the identification process.

        Returns:
            Dict[str, Any]: A dictionary containing the identified context.
                            The structure of this dictionary can vary based on the
                            document type but should be standardized by the implementing class.
                            Example for a financial teaser:
                            {
                                "identified_document_type": "financial_teaser", // or the doc_type_hint
                                "target_company": {"name": "...", "description": "..."},
                                "advisory_firms": [{"name": "...", "role": "..."}],
                                "project_codename": "..."
                            }
                            Example for a contract:
                            {
                                "identified_document_type": "contract",
                                "party_A": {"name": "...", "role": "..."},
                                "party_B": {"name": "...", "role": "..."},
                                "effective_date": "YYYY-MM-DD"
                            }
        Raises:
            NotImplementedError: If the subclass does not implement this method.
            FileNotFoundError: If document_path is provided but the file is not found.
            Exception: For other processing errors.
        """
        pass