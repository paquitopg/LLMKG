import re
import nltk
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

# Download required NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize, word_tokenize

# Import your existing components
from _1_document_ingestion.pdf_parser import PDFParser


@dataclass
class SemanticChunk:
    """Represents a semantically meaningful chunk of document content."""
    id: str
    text: str
    start_page: int
    end_page: int
    section_context: str  # Which section this chunk belongs to
    previous_chunk_summary: str  # Context from previous chunk
    document_position: float  # 0.0 to 1.0, position in document
    chunk_type: str  # 'text', 'table', 'mixed'
    related_chunks: List[str]  # IDs of semantically related chunks
    sentence_count: int
    word_count: int
    
    def __post_init__(self):
        """Calculate metrics after initialization."""
        if self.sentence_count == 0:
            self.sentence_count = len(sent_tokenize(self.text))
        if self.word_count == 0:
            self.word_count = len(word_tokenize(self.text))


@dataclass
class DocumentSection:
    """Represents a document section detected from structure."""
    title: str
    start_page: int
    end_page: int
    level: int  # Header level (1=main section, 2=subsection, etc.)
    content_type: str  # 'text', 'table', 'mixed'
    text_content: str
    
    
class SemanticChunker:
    """
    Creates semantically meaningful chunks from PDF documents.
    Uses document structure, topic detection, and content analysis.
    """
    
    def __init__(self, 
                 max_chunk_size: int = 4000,
                 min_chunk_size: int = 500,
                 overlap_size: int = 200,
                 respect_sentence_boundaries: bool = True,
                 detect_topic_shifts: bool = True):
        """
        Initialize the SemanticChunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk (avoid tiny chunks)
            overlap_size: Number of characters to overlap between chunks
            respect_sentence_boundaries: Don't break chunks mid-sentence
            detect_topic_shifts: Use NLP to detect topic boundaries
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.respect_sentence_boundaries = respect_sentence_boundaries
        self.detect_topic_shifts = detect_topic_shifts
        
        # Compile regex patterns for efficiency
        self.section_patterns = [
            re.compile(r'^(\d+\.?\d*\.?\d*)\s+([A-Z][^.!?]*)', re.MULTILINE),  # "1.1 Section Title"
            re.compile(r'^([A-Z][A-Z\s]{3,})\s*$', re.MULTILINE),  # "SECTION TITLE"
            re.compile(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$', re.MULTILINE),  # "Section Title"
        ]
        
        self.topic_shift_indicators = [
            'furthermore', 'however', 'moreover', 'additionally', 'in addition',
            'on the other hand', 'in contrast', 'nevertheless', 'consequently',
            'therefore', 'thus', 'accordingly', 'as a result', 'meanwhile',
            'subsequently', 'finally', 'in conclusion', 'to summarize'
        ]
        
        self.logger = logging.getLogger(__name__)
    
    def create_chunks(self, pdf_parser: PDFParser) -> List[SemanticChunk]:
        """
        Create semantic chunks from a PDF document.
        
        Args:
            pdf_parser: PDFParser instance with loaded document
            
        Returns:
            List of SemanticChunk objects
        """
        self.logger.info(f"Starting semantic chunking for document with {pdf_parser.num_pages} pages")
        
        # Extract all text with page information
        pages_data = pdf_parser.extract_all_pages_text()
        
        # Detect document structure
        sections = self._detect_document_sections(pages_data)
        
        # Create initial chunks respecting structure
        initial_chunks = self._create_structure_aware_chunks(pages_data, sections)
        
        # Refine chunks based on content analysis
        refined_chunks = self._refine_chunks_by_content(initial_chunks)
        
        # Add overlap and context
        final_chunks = self._add_overlap_and_context(refined_chunks)
        
        self.logger.info(f"Created {len(final_chunks)} semantic chunks")
        return final_chunks
    
    def _detect_document_sections(self, pages_data: List[Dict[str, Any]]) -> List[DocumentSection]:
        """
        Detect document sections using headers, formatting, and content patterns.
        
        Args:
            pages_data: List of page data from PDFParser
            
        Returns:
            List of detected DocumentSection objects
        """
        sections = []
        current_section = None
        
        for page_info in pages_data:
            page_num = page_info['page_number']
            page_text = page_info['text']
            
            # Look for section headers
            detected_headers = self._find_section_headers(page_text)
            
            for header_info in detected_headers:
                # Close previous section
                if current_section:
                    current_section.end_page = page_num - 1
                    sections.append(current_section)
                
                # Start new section
                current_section = DocumentSection(
                    title=header_info['title'],
                    start_page=page_num,
                    end_page=page_num,  # Will be updated when next section starts
                    level=header_info['level'],
                    content_type='text',  # Default, could be refined
                    text_content=''
                )
            
            # Add page text to current section
            if current_section:
                current_section.text_content += f"\n{page_text}"
                current_section.end_page = page_num
        
        # Close final section
        if current_section:
            sections.append(current_section)
        
        # If no sections detected, create one section for entire document
        if not sections:
            all_text = '\n'.join([page['text'] for page in pages_data])
            sections.append(DocumentSection(
                title="Document Content",
                start_page=1,
                end_page=len(pages_data),
                level=1,
                content_type='text',
                text_content=all_text
            ))
        
        self.logger.info(f"Detected {len(sections)} document sections")
        return sections
    
    def _find_section_headers(self, text: str) -> List[Dict[str, Any]]:
        """
        Find section headers in text using pattern matching.
        
        Args:
            text: Text to search for headers
            
        Returns:
            List of header information dictionaries
        """
        headers = []
        
        for level, pattern in enumerate(self.section_patterns, 1):
            matches = pattern.finditer(text)
            for match in matches:
                headers.append({
                    'title': match.group().strip(),
                    'level': level,
                    'position': match.start()
                })
        
        # Sort by position in text
        headers.sort(key=lambda x: x['position'])
        return headers
    
    def _create_structure_aware_chunks(self, pages_data: List[Dict[str, Any]], 
                                     sections: List[DocumentSection]) -> List[Dict[str, Any]]:
        """
        Create initial chunks that respect document structure.
        
        Args:
            pages_data: List of page data
            sections: Detected document sections
            
        Returns:
            List of initial chunk dictionaries
        """
        chunks = []
        
        for section in sections:
            section_text = section.text_content.strip()
            if not section_text:
                continue
            
            # Split section into chunks if it's too large
            if len(section_text) <= self.max_chunk_size:
                # Section fits in one chunk
                chunks.append({
                    'text': section_text,
                    'start_page': section.start_page,
                    'end_page': section.end_page,
                    'section_title': section.title,
                    'section_level': section.level
                })
            else:
                # Split section into multiple chunks
                section_chunks = self._split_large_section(section)
                chunks.extend(section_chunks)
        
        return chunks
    
    def _split_large_section(self, section: DocumentSection) -> List[Dict[str, Any]]:
        """
        Split a large section into multiple chunks while preserving coherence.
        
        Args:
            section: DocumentSection that needs to be split
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        text = section.text_content
        
        # Split by paragraphs first
        paragraphs = self._split_into_paragraphs(text)
        
        current_chunk = ""
        current_start_page = section.start_page
        
        for para in paragraphs:
            # Check if adding this paragraph would exceed max size
            if len(current_chunk) + len(para) > self.max_chunk_size and current_chunk:
                # Finalize current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'start_page': current_start_page,
                    'end_page': section.end_page,  # Approximate, could be refined
                    'section_title': section.title,
                    'section_level': section.level
                })
                current_chunk = para
            else:
                current_chunk += f"\n{para}" if current_chunk else para
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'start_page': current_start_page,
                'end_page': section.end_page,
                'section_title': section.title,
                'section_level': section.level
            })
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs using various indicators.
        
        Args:
            text: Text to split
            
        Returns:
            List of paragraph strings
        """
        # Split by double newlines (most common paragraph separator)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Further split very long paragraphs by sentences if needed
        refined_paragraphs = []
        for para in paragraphs:
            if len(para) > self.max_chunk_size * 0.8:  # 80% of max chunk size
                sentences = sent_tokenize(para)
                current_para = ""
                
                for sentence in sentences:
                    if len(current_para) + len(sentence) > self.max_chunk_size * 0.8:
                        if current_para:
                            refined_paragraphs.append(current_para.strip())
                            current_para = sentence
                        else:
                            # Single sentence is too long, add it anyway
                            refined_paragraphs.append(sentence)
                    else:
                        current_para += f" {sentence}" if current_para else sentence
                
                if current_para:
                    refined_paragraphs.append(current_para.strip())
            else:
                refined_paragraphs.append(para)
        
        return [p for p in refined_paragraphs if p.strip()]
    
    def _refine_chunks_by_content(self, initial_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Refine chunks based on content analysis and topic detection.
        
        Args:
            initial_chunks: Initial chunk dictionaries
            
        Returns:
            Refined chunk dictionaries
        """
        if not self.detect_topic_shifts:
            return initial_chunks
        
        refined_chunks = []
        
        for chunk in initial_chunks:
            # Detect topic shifts within the chunk
            topic_boundaries = self._detect_topic_boundaries(chunk['text'])
            
            if not topic_boundaries:
                # No topic shifts detected, keep chunk as is
                refined_chunks.append(chunk)
            else:
                # Split chunk at topic boundaries
                split_chunks = self._split_at_boundaries(chunk, topic_boundaries)
                refined_chunks.extend(split_chunks)
        
        return refined_chunks
    
    def _detect_topic_boundaries(self, text: str) -> List[int]:
        """
        Detect topic shift boundaries within text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of character positions where topic shifts occur
        """
        boundaries = []
        sentences = sent_tokenize(text)
        current_pos = 0
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Check for topic shift indicators
            for indicator in self.topic_shift_indicators:
                if indicator in sentence_lower:
                    # Found a topic shift indicator
                    boundaries.append(current_pos)
                    break
            
            current_pos += len(sentence) + 1  # +1 for space/punctuation
        
        return boundaries
    
    def _split_at_boundaries(self, chunk: Dict[str, Any], boundaries: List[int]) -> List[Dict[str, Any]]:
        """
        Split a chunk at detected topic boundaries.
        
        Args:
            chunk: Chunk dictionary to split
            boundaries: List of character positions to split at
            
        Returns:
            List of split chunk dictionaries
        """
        text = chunk['text']
        split_chunks = []
        start_pos = 0
        
        for boundary in boundaries:
            if boundary > start_pos:
                chunk_text = text[start_pos:boundary].strip()
                if len(chunk_text) >= self.min_chunk_size:
                    new_chunk = chunk.copy()
                    new_chunk['text'] = chunk_text
                    split_chunks.append(new_chunk)
                start_pos = boundary
        
        # Add final chunk
        final_text = text[start_pos:].strip()
        if len(final_text) >= self.min_chunk_size:
            final_chunk = chunk.copy()
            final_chunk['text'] = final_text
            split_chunks.append(final_chunk)
        elif split_chunks:
            # Merge short final text with last chunk
            split_chunks[-1]['text'] += f"\n{final_text}"
        
        return split_chunks if split_chunks else [chunk]
    
    def _add_overlap_and_context(self, chunks: List[Dict[str, Any]]) -> List[SemanticChunk]:
        """
        Add overlap between chunks and create final SemanticChunk objects.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of SemanticChunk objects with overlap and context
        """
        semantic_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i+1:03d}"
            
            # Add overlap from previous chunk
            chunk_text = chunk['text']
            if i > 0 and self.overlap_size > 0:
                prev_text = chunks[i-1]['text']
                overlap_text = prev_text[-self.overlap_size:] if len(prev_text) > self.overlap_size else prev_text
                
                # Find a good sentence boundary for overlap
                if self.respect_sentence_boundaries:
                    sentences = sent_tokenize(overlap_text)
                    if sentences:
                        overlap_text = sentences[-1] if len(sentences) == 1 else " ".join(sentences[-2:])
                
                chunk_text = f"{overlap_text}\n\n{chunk_text}"
            
            # Generate previous chunk summary for context
            previous_summary = ""
            if i > 0:
                prev_chunk_text = chunks[i-1]['text']
                previous_summary = self._generate_chunk_summary(prev_chunk_text)
            
            # Calculate document position
            document_position = i / len(chunks) if len(chunks) > 1 else 0.0
            
            # Create SemanticChunk object
            semantic_chunk = SemanticChunk(
                id=chunk_id,
                text=chunk_text,
                start_page=chunk.get('start_page', 1),
                end_page=chunk.get('end_page', 1),
                section_context=chunk.get('section_title', 'Unknown Section'),
                previous_chunk_summary=previous_summary,
                document_position=document_position,
                chunk_type='text',  # Could be enhanced to detect tables, etc.
                related_chunks=[],  # Could be populated with semantic similarity
                sentence_count=0,  # Will be calculated in __post_init__
                word_count=0       # Will be calculated in __post_init__
            )
            
            semantic_chunks.append(semantic_chunk)
        
        return semantic_chunks
    
    def _generate_chunk_summary(self, text: str, max_length: int = 200) -> str:
        """
        Generate a brief summary of chunk content for context.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary string
        """
        # Simple extractive summary - take first and last sentences
        sentences = sent_tokenize(text)
        
        if not sentences:
            return ""
        
        if len(sentences) == 1:
            summary = sentences[0]
        elif len(sentences) == 2:
            summary = " ".join(sentences)
        else:
            # Take first and last sentences
            summary = f"{sentences[0]} ... {sentences[-1]}"
        
        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def get_chunk_statistics(self, chunks: List[SemanticChunk]) -> Dict[str, Any]:
        """
        Calculate statistics about the generated chunks.
        
        Args:
            chunks: List of SemanticChunk objects
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk.text) for chunk in chunks]
        word_counts = [chunk.word_count for chunk in chunks]
        sentence_counts = [chunk.sentence_count for chunk in chunks]
        
        stats = {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'avg_sentence_count': sum(sentence_counts) / len(sentence_counts),
            'total_text_length': sum(chunk_sizes),
            'sections_detected': len(set(chunk.section_context for chunk in chunks))
        }
        
        return stats