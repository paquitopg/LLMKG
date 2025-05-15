from pathlib import Path
from pypdf import PdfReader, PdfWriter
import pymupdf
import io
from io import BytesIO
import base64
from PIL import Image
from typing import List, Dict

class PDFProcessor:
    """
    A unified class to process PDF documents:
    - Extract text from pages
    - Save pages as individual text files
    - Save pages as individual PDF files
    """

    def __init__(self, pdf_path):
        self.pdf_path = Path(pdf_path)
        self.reader = PdfReader(str(self.pdf_path))
        self.doc = pymupdf.open(str(self.pdf_path))
        self.page_dpi = 300 

    def extract_text(self):
        """Extract all text from the PDF."""
        return "\n".join(page.get_text() for page in self.doc)

    def extract_page_text(self, page_number):
        """Extract text from a specific page (1-indexed)."""
        return self.doc[page_number].get_text()

    def save_page_text(self, page_number, output_path):
        """Save the text of a specific page to a .txt file."""
        content = self.extract_page_text(page_number)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def dump_all_pages_as_text(self, output_dir):
        """Save all pages as individual text files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, page in enumerate(self.doc):
            output_path = output_dir / f"page_{i + 1}.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(page.get_text())

    def extract_pages_as_pdfs(self, output_dir):
        """Save each page as a separate PDF file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, page in enumerate(self.reader.pages):
            writer = PdfWriter()
            writer.add_page(page)
            output_path = output_dir / f"page_{i + 1}.pdf"
            with open(output_path, "wb") as f:
                writer.write(f)

        return f"{len(self.reader.pages)} pages extracted to {output_dir}"
    
    def extract_text_as_list(self):
        """
        Extract text from the PDF file using PyMuPDF.
        Returns:
            List[str]: A list of texts, one for each page in the PDF.
        """
        doc = pymupdf.open(self.pdf_path)
        return [page.get_text() for page in doc]
    
    def extract_pages_from_pdf(self) -> List[Dict]: 
        """
        Extract pages from a PDF file as images using PyMuPDF.
        Returns:
            List[Dict]: List of dictionaries containing page images and metadata.
        """
        pages = []
        doc = pymupdf.open(self.pdf_path)

        for page_num, page in enumerate(doc):

            width, height = page.rect.width, page.rect.height

            matrix = pymupdf.Matrix(self.page_dpi/72, self.page_dpi/72) 
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)

            img_data = pixmap.tobytes("png")
            img = Image.open(BytesIO(img_data))
            
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            text = page.get_text()
            
            pages.append({
                "page_num": page_num + 1,
                "width": width,
                "height": height,
                "image_base64": img_base64,
                "text": text
            })
        
        return pages
    
    def extract_page_from_pdf(self, page_number: int) -> Dict:
        """
        Extract a specific page from a PDF file as an image using PyMuPDF.
        Args:
            page_number (int): Page number to extract (1-indexed).
        Returns:
            Dict: Dictionary containing the page image and metadata.
        """
        doc = pymupdf.open(self.pdf_path)
        page = doc[page_number]

        width, height = page.rect.width, page.rect.height

        matrix = pymupdf.Matrix(self.page_dpi/72, self.page_dpi/72) 
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        img_data = pixmap.tobytes("png")
        img = Image.open(BytesIO(img_data))
        
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        text = page.get_text()
        
        return {
            "page_num": page_number,
            "width": width,
            "height": height,
            "image_base64": img_base64,
            "text": text
        }