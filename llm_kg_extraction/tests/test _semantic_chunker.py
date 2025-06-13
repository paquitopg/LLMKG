# File: tests/test_semantic_chunker.py

"""
Test script for SemanticChunker functionality.
Run this to validate that the semantic chunking is working correctly.
"""

import sys
import tempfile
import json
from pathlib import Path

# Add the parent directory to the system path
sys.path.append(str(Path(__file__).parent.parent))

from _1_document_ingestion.pdf_parser import PDFParser
from _3_knowledge_extraction.document_aware_extraction import SemanticChunker, SemanticChunk


def create_test_pdf_content():
    """
    Create a sample text that simulates a complex document
    that would benefit from semantic chunking.
    """
    return """
1. EXECUTIVE SUMMARY

This document provides a comprehensive analysis of TechCorp's financial performance 
and strategic positioning. The company has shown remarkable growth over the past fiscal year.

TechCorp, founded in 2020, specializes in artificial intelligence solutions for 
enterprise clients. The company's flagship product, AI-Analytics Suite, has gained 
significant market traction.

2. FINANCIAL PERFORMANCE

2.1 Revenue Analysis

For the fiscal year 2023, TechCorp reported total revenue of $15.2 million, 
representing a 45% increase from the previous year. This growth was primarily 
driven by new client acquisitions and expansion of existing contracts.

The revenue breakdown by quarter shows consistent growth:
- Q1 2023: $3.2 million
- Q2 2023: $3.8 million  
- Q3 2023: $4.1 million
- Q4 2023: $4.1 million

2.2 Profitability Metrics

TechCorp achieved an EBITDA of $3.8 million in FY2023, compared to $1.2 million 
in the previous year. This represents an EBITDA margin of 25%, which is considered 
strong for a company in the growth phase.

Furthermore, the company has maintained healthy cash flow throughout the year. 
Operating cash flow reached $3.2 million, providing sufficient liquidity for 
planned expansion activities.

3. MARKET POSITION

The artificial intelligence market continues to expand rapidly. According to 
industry reports, the global AI market is expected to reach $190 billion by 2025.

TechCorp competes primarily with established players such as DataCorp and 
AI Solutions Inc. However, the company's focus on specialized industry verticals 
has allowed it to carve out a unique market position.

3.1 Competitive Advantages

TechCorp's main competitive advantages include:
- Proprietary machine learning algorithms
- Strong customer relationships
- Experienced management team led by CEO Sarah Johnson

The company's technology platform processes over 100 million data points daily, 
enabling real-time insights for clients across various industries.

4. STRATEGIC INITIATIVES

Looking ahead, TechCorp has outlined several strategic initiatives for 2024:

4.1 Product Development

The company plans to invest $2.5 million in R&D to enhance its AI capabilities. 
This includes development of new features for predictive analytics and 
natural language processing.

4.2 Market Expansion

TechCorp intends to expand into European markets, with plans to establish 
offices in London and Berlin by Q3 2024. This expansion is expected to increase 
the company's addressable market by approximately 40%.

Additionally, the company is exploring strategic partnerships with local 
technology firms to accelerate market penetration in these new regions.

4.3 Talent Acquisition

To support its growth objectives, TechCorp plans to increase its workforce from 
85 employees to 120 employees by the end of 2024. Key hiring priorities include:
- Senior data scientists
- Sales representatives for European markets  
- Customer success managers

5. RISK FACTORS

While TechCorp's prospects are positive, several risk factors should be considered:

5.1 Market Competition

The AI market is highly competitive, with both established technology giants and 
emerging startups vying for market share. Increased competition could pressure 
pricing and market positioning.

5.2 Technology Risks

Rapid technological change in the AI field means that TechCorp's current advantages 
could be eroded if the company fails to innovate continuously.

However, the company's strong R&D capabilities and experienced technical team 
mitigate these risks to some extent.

6. CONCLUSION

TechCorp represents a compelling investment opportunity in the growing AI market. 
The company's strong financial performance, competitive positioning, and clear 
strategic vision position it well for continued growth.

Nevertheless, potential investors should carefully consider the risk factors 
outlined in this analysis before making investment decisions.
"""


def test_basic_semantic_chunking():
    """Test basic semantic chunking functionality."""
    print("Testing basic semantic chunking...")
    
    # Create a mock PDF parser with test content
    class MockPDFParser:
        def __init__(self, content):
            self.content = content
            self.num_pages = 3  # Simulate 3 pages
            
        def extract_all_pages_text(self):
            # Split content into simulated pages
            lines = self.content.split('\n')
            page_size = len(lines) // 3
            
            pages = []
            for i in range(3):
                start_idx = i * page_size
                end_idx = (i + 1) * page_size if i < 2 else len(lines)
                page_text = '\n'.join(lines[start_idx:end_idx])
                
                pages.append({
                    'page_number': i + 1,
                    'text': page_text
                })
            
            return pages
    
    # Create test parser
    test_content = create_test_pdf_content()
    mock_parser = MockPDFParser(test_content)
    
    # Initialize semantic chunker
    chunker = SemanticChunker(
        max_chunk_size=2000,
        min_chunk_size=300,
        overlap_size=150,
        respect_sentence_boundaries=True,
        detect_topic_shifts=True
    )
    
    # Create chunks
    chunks = chunker.create_chunks(mock_parser)
    
    # Validate results
    assert len(chunks) > 0, "No chunks were created"
    assert all(isinstance(chunk, SemanticChunk) for chunk in chunks), "Invalid chunk types"
    
    print(f"✓ Created {len(chunks)} semantic chunks")
    
    # Test chunk properties
    for i, chunk in enumerate(chunks):
        assert chunk.id == f"chunk_{i+1:03d}", f"Invalid chunk ID: {chunk.id}"
        assert len(chunk.text) >= chunker.min_chunk_size or i == len(chunks)-1, f"Chunk {chunk.id} too small: {len(chunk.text)}"
        assert len(chunk.text) <= chunker.max_chunk_size + chunker.overlap_size, f"Chunk {chunk.id} too large: {len(chunk.text)}"
        assert 0 <= chunk.document_position <= 1, f"Invalid document position: {chunk.document_position}"
        
    print(f"✓ All chunk properties validated")
    
    return chunks


def test_section_detection():
    """Test section detection functionality."""
    print("\nTesting section detection...")
    
    # Create test content with clear sections
    test_content = """
1. INTRODUCTION
This is the introduction section with some content.

2. METHODOLOGY  
This section describes the methodology used in the analysis.

2.1 Data Collection
This subsection covers data collection procedures.

2.2 Analysis Framework
This subsection describes the analysis framework.

3. RESULTS
This section presents the results of the analysis.
"""
    
    class MockPDFParser:
        def __init__(self, content):
            self.content = content
            self.num_pages = 1
            
        def extract_all_pages_text(self):
            return [{'page_number': 1, 'text': self.content}]
    
    mock_parser = MockPDFParser(test_content)
    chunker = SemanticChunker(max_chunk_size=1000)
    
    # Test section detection
    pages_data = mock_parser.extract_all_pages_text()
    sections = chunker._detect_document_sections(pages_data)
    
    print(f"✓ Detected {len(sections)} sections")
    
    # Validate sections
    section_titles = [section.title for section in sections]
    print(f"  Section titles: {section_titles}")
    
    # Should detect at least the main sections
    assert len(sections) >= 1, "No sections detected"
    
    return sections


def test_chunk_overlap():
    """Test chunk overlap functionality."""
    print("\nTesting chunk overlap...")
    
    test_content = "This is sentence one. This is sentence two. This is sentence three. " * 100
    
    class MockPDFParser:
        def __init__(self, content):
            self.content = content
            self.num_pages = 1
            
        def extract_all_pages_text(self):
            return [{'page_number': 1, 'text': self.content}]

    mock_parser = MockPDFParser(test_content)
    chunker = SemanticChunker(
        max_chunk_size=500,
        overlap_size=100,
        respect_sentence_boundaries=True
    )
    
    chunks = chunker.create_chunks(mock_parser)
    
    # Test overlap
    if len(chunks) > 1:
        # Check that consecutive chunks have some overlapping content
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Get the end of current chunk and start of next chunk
            current_end = current_chunk.text[-100:]  # Last 100 chars
            next_start = next_chunk.text[:200]  # First 200 chars
            
            # There should be some overlap
            overlap_found = any(word in next_start for word in current_end.split()[-10:])
            print(f"  Overlap between chunk {i+1} and {i+2}: {'✓' if overlap_found else '✗'}")
    
    print(f"✓ Overlap testing completed")
    
    return chunks


def test_statistics():
    """Test chunk statistics calculation."""
    print("\nTesting statistics calculation...")
    
    test_content = create_test_pdf_content()
    
    class MockPDFParser:
        def __init__(self, content):
            self.content = content
            self.num_pages = 2
            
        def extract_all_pages_text(self):
            return [
                {'page_number': 1, 'text': self.content[:len(self.content)//2]},
                {'page_number': 2, 'text': self.content[len(self.content)//2:]}
            ]
    
    mock_parser = MockPDFParser(test_content)
    chunker = SemanticChunker(max_chunk_size=1500)
    
    chunks = chunker.create_chunks(mock_parser)
    stats = chunker.get_chunk_statistics(chunks)
    
    # Validate statistics
    required_stats = ['total_chunks', 'avg_chunk_size', 'min_chunk_size', 'max_chunk_size', 
                     'avg_word_count', 'avg_sentence_count', 'total_text_length', 'sections_detected']
    
    for stat in required_stats:
        assert stat in stats, f"Missing statistic: {stat}"
        assert isinstance(stats[stat], (int, float)), f"Invalid statistic type: {stat}"
    
    print(f"✓ Statistics calculated successfully:")
    for key, value in stats.items():
        print(f"  {key}: {value:.1f}" if isinstance(value, float) else f"  {key}: {value}")
    
    return stats


def test_comparison_with_page_based():
    """Compare semantic chunking with page-based processing."""
    print("\nComparing with page-based processing...")
    
    test_content = create_test_pdf_content()
    
    class MockPDFParser:
        def __init__(self, content):
            self.content = content
            self.num_pages = 4
            
        def extract_all_pages_text(self):
            # Simulate page breaks that split content awkwardly
            lines = self.content.split('\n')
            page_size = len(lines) // 4
            
            pages = []
            for i in range(4):
                start_idx = i * page_size
                end_idx = (i + 1) * page_size if i < 3 else len(lines)
                page_text = '\n'.join(lines[start_idx:end_idx])
                
                pages.append({
                    'page_number': i + 1,
                    'text': page_text
                })
            
            return pages
    
    mock_parser = MockPDFParser(test_content)
    
    # Page-based processing
    pages = mock_parser.extract_all_pages_text()
    page_sizes = [len(page['text']) for page in pages]
    
    # Semantic chunking
    chunker = SemanticChunker(max_chunk_size=1500)
    chunks = chunker.create_chunks(mock_parser)
    chunk_sizes = [len(chunk.text) for chunk in chunks]
    
    print(f"Page-based processing:")
    print(f"  {len(pages)} pages, sizes: {min(page_sizes)}-{max(page_sizes)} chars")
    print(f"  Average: {sum(page_sizes)/len(page_sizes):.1f} chars")
    
    print(f"Semantic chunking:")
    print(f"  {len(chunks)} chunks, sizes: {min(chunk_sizes)}-{max(chunk_sizes)} chars") 
    print(f"  Average: {sum(chunk_sizes)/len(chunk_sizes):.1f} chars")
    
    # Count cross-page chunks
    cross_page_chunks = [c for c in chunks if c.start_page != c.end_page]
    print(f"  {len(cross_page_chunks)} chunks span multiple pages")
    
    print("✓ Comparison completed")


def run_all_tests():
    """Run all tests for SemanticChunker."""
    print("=" * 60)
    print("SEMANTIC CHUNKER TEST SUITE")
    print("=" * 60)
    
    try:
        # Run individual tests
        chunks = test_basic_semantic_chunking()
        sections = test_section_detection()
        overlap_chunks = test_chunk_overlap()
        stats = test_statistics()
        test_comparison_with_page_based()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
        print(f"\nSemanticChunker is ready for integration!")
        print(f"Example usage in your pipeline:")
        print(f"  python main_orchestrator.py input_folder project_name vertexai \\")
        print(f"    --main_model_name 'gemini-1.5-pro' \\")
        print(f"    --processing_mode document_aware \\")
        print(f"    --chunk_size 3000 --chunk_overlap 200")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Check if NLTK data is available
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except:
        print("Downloading required NLTK data...")
        import nltk
        nltk.download('punkt')
        nltk.download('punkt_tab')
    
    # Run tests
    success = run_all_tests()
    sys.exit(0 if success else 1)