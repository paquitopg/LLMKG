class PDFProcessor:
    def __init__(self, document_path):
        self.document_path = document_path
        self.pages = self._split_document_into_pages()

    def _split_document_into_pages(self):
        """Split a PDF into individual pages and return a list of page contents."""
        import pymupdf
        
        doc = pymupdf.open(self.document_path)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            pages.append({
                "page_number": page_num + 1,
                "content": text
            })
        
        return pages
    
    def get_page_content(self, page_number):
        """Get the content of a specific page (1-indexed)."""
        if 1 <= page_number <= len(self.pages):
            return self.pages[page_number - 1]["content"]
        raise ValueError(f"Page number {page_number} is out of range")
    
    def get_total_pages(self):
        """Get the total number of pages in the document."""
        return len(self.pages)
    
    def extract_text(self):
        """Extract text from the entire document."""
        return "\n".join([page["content"] for page in self.pages])