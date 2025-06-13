# File: llm_kg_extraction/_3_knowledge_extraction/document_aware_extraction/__init__.py

"""
Document-aware extraction module for processing documents with cross-page context.

This module provides tools for semantic chunking, global entity tracking,
and context-aware processing that goes beyond simple page-based extraction.
"""

from .semantic_chunker import SemanticChunker, SemanticChunk, DocumentSection

__all__ = [
    'SemanticChunker',
    'SemanticChunk', 
    'DocumentSection'
]

__version__ = '0.1.0'