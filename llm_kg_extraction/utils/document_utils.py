from pathlib import Path
from pypdf import PdfReader, PdfWriter
import pymupdf
import io

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

    def extract_text(self):
        """Extract all text from the PDF."""
        return "\n".join(page.get_text() for page in self.doc)

    def extract_page_text(self, page_number):
        """Extract text from a specific page (1-indexed)."""
        if 1 <= page_number <= len(self.doc):
            return self.doc[page_number - 1].get_text()
        raise ValueError(f"Page {page_number} is out of range.")

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


    def get_pages_as_pdf_streams(self):
        """
        Return a list of PDF byte streams (BytesIO), one for each page.
        Useful for in-memory processing without writing to disk.

        Returns:
            List[io.BytesIO]: List of in-memory PDFs, each containing one page.
        """
        pdf_streams = []
        for page in self.reader.pages:
            writer = PdfWriter()
            writer.add_page(page)
            buffer = io.BytesIO()
            writer.write(buffer)
            buffer.seek(0)
            pdf_streams.append(buffer)
            print(f"Page {len(pdf_streams)} extracted as PDF stream.")
        return pdf_streams