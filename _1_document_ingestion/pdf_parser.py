from pathlib import Path
from pypdf import PdfReader, PdfWriter
import pymupdf
import io
from io import BytesIO
import base64
from PIL import Image
from typing import List, Dict, Optional

class PDFParser:
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
        self.num_pages = len(self.reader.pages)

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
    
    def extract_all_pages_text(self):
        """Extract text from all pages and return as a list of dictionaries.
        Each dictionary contains page number and text content."""
        return [{"page_number": i + 1, "text": page.get_text()} for i, page in enumerate(self.doc)]

    def extract_all_pages_multimodal(self):
        """Extract text and images from all pages and return as a list of dictionaries.
        Each dictionary contains page number, text content, and image data."""
        return [{"page_number": i + 1, "text": page.get_text(), "image_base64": self.extract_page_image_base64(i)} for i, page in enumerate(self.doc)]

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
                "page_number": page_num + 1,
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
            "page_number": page_number,
            "width": width,
            "height": height,
            "image_base64": img_base64,
            "text": text
        }
    
    def extract_text_from_first_n_pages(self, n: int = 4) -> Optional[str]:
        """
        Extracts concatenated text from the first N pages.

        Args:
            n (int): Number of first pages to extract text from.

        Returns:
            Optional[str]: Concatenated text string, or None if no pages.
        """
        if n <= 0:
            return None
        
        pages_to_extract = min(n, self.num_pages)
        if pages_to_extract == 0:
            return None
            
        all_text = []
        for page_num in range(pages_to_extract):
            page_text = self.extract_page_text(page_num)
            if page_text:
                all_text.append(page_text)
        
        return "\n\n--- Page Break ---\n\n".join(all_text) if all_text else None
    
    def extract_page_image_base64(self, page_number_zero_indexed: int) -> Optional[str]:
        """
        Extracts a specific page from a PDF file as a base64 encoded PNG image.
        
        Args:
            page_number_zero_indexed (int): Page number to extract (0-indexed).
        
        Returns:
            Optional[str]: Base64 encoded PNG image string, or None if page is invalid.
        """
        if not (0 <= page_number_zero_indexed < self.num_pages):
            print(f"Warning: Page number {page_number_zero_indexed + 1} is out of range.")
            return None

        page = self.doc.load_page(page_number_zero_indexed)
        matrix = pymupdf.Matrix(self.page_dpi / 72, self.page_dpi / 72)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        
        img_byte_arr = io.BytesIO()
        # Use PIL to save as PNG and ensure format consistency if needed,
        # though pixmap.tobytes("png") is often sufficient.
        try:
            # pixmap.save(img_byte_arr, "png") # PyMuPDF 1.19+
            img_data = pixmap.tobytes("png") # For older versions or direct bytes
            img = Image.open(BytesIO(img_data))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_base64
        except Exception as e:
            print(f"Error converting page {page_number_zero_indexed + 1} to image: {e}")
            return None


    def extract_images_from_first_n_pages_base64(self, n: int = 4) -> List[str]:
        """
        Extracts images (as base64 PNG strings) from the first N pages.

        Args:
            n (int): Number of first pages to extract images from.

        Returns:
            List[str]: A list of base64 encoded image strings. Empty if pages are invalid or errors occur.
        """
        if n <= 0:
            return []
        
        pages_to_extract = min(n, self.num_pages)
        images_base64 = []
        for page_num in range(pages_to_extract):
            img_b64 = self.extract_page_image_base64(page_num)
            if img_b64:
                images_base64.append(img_b64)
        return images_base64

    def get_toc_text(self, max_depth: int = 6) -> Optional[str]:
        """
        Extracts the Table of Contents and formats it as a string.

        Args:
            max_depth (int): Maximum depth of ToC entries to include.

        Returns:
            Optional[str]: Formatted ToC string, or None if no ToC.
        """
        toc = self.doc.get_toc(simple=False) # Returns list of [lvl, title, page, dest]
        if not toc:
            return None

        formatted_toc = ["Table of Contents:"]
        for level, title, page_num, _dest_details in toc: # Ignored _dest_details for now
            if level <= max_depth:
                indent = "  " * (level - 1)
                # Page numbers from get_toc are 1-indexed
                formatted_toc.append(f"{indent}- {title} (Page {page_num})")
        
        return "\n".join(formatted_toc) if len(formatted_toc) > 1 else None
        
    def __del__(self):
        """Ensure the document is closed when the object is deleted."""
        if hasattr(self, 'doc') and self.doc:
            self.doc.close()