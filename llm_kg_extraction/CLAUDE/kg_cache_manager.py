import os
import json
import time
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger('KGCacheManager')

class KGDiskCache:
    """
    File-based cache implementation for knowledge graph segments.
    Uses a file system structure for caching LLM responses.
    """
    
    def __init__(self, cache_dir: Path):
        """
        Initialize a disk-based cache at the specified directory.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Disk cache initialized at {cache_dir}")
    
    def _generate_key(self, prompt: str) -> str:
        """Generate a unique hash key for a prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[Dict]:
        """Retrieve a cached response for a prompt if it exists."""
        key = self._generate_key(prompt)
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    logger.debug(f"Cache hit for key {key}")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading from cache: {e}")
                return None
        return None
    
    def set(self, prompt: str, response: Dict) -> None:
        """Cache a response for a prompt."""
        key = self._generate_key(prompt)
        cache_file = self.cache_dir / f"{key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(response, f)
            logger.debug(f"Cached response for key {key}")
        except Exception as e:
            logger.warning(f"Error writing to cache: {e}")
    
    def clear(self) -> None:
        """Clear all cached responses."""
        for file in self.cache_dir.glob("*.json"):
            try:
                file.unlink()
            except Exception as e:
                logger.warning(f"Error deleting cache file {file}: {e}")
        logger.info("Cache cleared")


class KGSQLiteCache:
    """
    SQLite-based cache implementation for knowledge graph segments.
    More efficient for frequent lookups and large datasets compared to file-based caching.
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize an SQLite-based cache at the specified file.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize the database and create the cache table if it doesn't exist
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS kg_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    prompt TEXT,
                    response TEXT,
                    timestamp REAL
                )
            ''')
            conn.commit()
        
        logger.info(f"SQLite cache initialized at {db_path}")
    
    def _get_connection(self):
        """Get a connection to the SQLite database."""
        return sqlite3.connect(self.db_path)
    
    def _generate_key(self, prompt: str) -> str:
        """Generate a unique hash key for a prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[Dict]:
        """Retrieve a cached response for a prompt if it exists."""
        key = self._generate_key(prompt)
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT response FROM kg_cache WHERE prompt_hash = ?", (key,))
                result = cursor.fetchone()
                
                if result:
                    logger.debug(f"Cache hit for key {key}")
                    return json.loads(result[0])
                return None
        except Exception as e:
            logger.warning(f"Error reading from cache: {e}")
            return None
    
    def set(self, prompt: str, response: Dict) -> None:
        """Cache a response for a prompt."""
        key = self._generate_key(prompt)
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO kg_cache (prompt_hash, prompt, response, timestamp) VALUES (?, ?, ?, ?)",
                    (key, prompt, json.dumps(response), time.time())
                )
                conn.commit()
            logger.debug(f"Cached response for key {key}")
        except Exception as e:
            logger.warning(f"Error writing to cache: {e}")
    
    def clear(self) -> None:
        """Clear all cached responses."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM kg_cache")
                conn.commit()
            logger.info("Cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")
    
    def clear_older_than(self, seconds: int) -> None:
        """Clear cached responses older than the specified number of seconds."""
        cutoff_time = time.time() - seconds
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM kg_cache WHERE timestamp < ?", (cutoff_time,))
                deleted_count = cursor.rowcount
                conn.commit()
            logger.info(f"Cleared {deleted_count} cache entries older than {seconds} seconds")
        except Exception as e:
            logger.warning(f"Error clearing old cache entries: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM kg_cache")
                count = cursor.fetchone()[0]
                
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM kg_cache")
                min_time, max_time = cursor.fetchone()
                
                return {
                    "entry_count": count,
                    "oldest_entry": time.ctime(min_time) if min_time else None,
                    "newest_entry": time.ctime(max_time) if max_time else None,
                    "size_mb": os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
                }
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return {"error": str(e)}


class MemoryEfficientKGBuilder:
    """
    A specialized class for building knowledge graphs with low memory footprint.
    Uses streaming processing and memory-efficient data structures.
    """
    
    def __init__(self, cache_manager, output_manager, llm_client):
        """
        Initialize with required dependencies.
        
        Args:
            cache_manager: Cache manager for LLM responses
            output_manager: Manager for handling output files
            llm_client: Client for LLM API calls
        """
        self.cache = cache_manager
        self.output = output_manager
        self.llm = llm_client
        self.entity_index = {}  # Maps entity IDs to their data
        self.relationship_buffer = []  # Temporary storage for relationships
        self.buffer_size = 1000  # Max relationships to keep in memory
    
    def process_text_chunk(self, text: str, context: Dict = None) -> Dict:
        """
        Process a chunk of text and update the knowledge graph efficiently.
        
        Args:
            text: Text chunk to process
            context: Optional context from previous processing
            
        Returns:
            Extracted knowledge graph segment
        """
        # Use cached response if available
        cached = self.cache.get(text)
        if cached:
            return self._integrate_graph_segment(cached)
        
        # Get response from LLM
        response = self.llm.extract_knowledge_graph(text, context)
        self.cache.set(text, response)
        
        return self._integrate_graph_segment(response)
    
    def _integrate_graph_segment(self, segment: Dict) -> Dict:
        """
        Integrate a new graph segment with existing entities and relationships.
        Handles entity deduplication and relationship merging.
        
        Args:
            segment: New knowledge graph segment
            
        Returns:
            Updated segment with resolved references
        """
        # Process entities
        for entity in segment.get("entities", []):
            entity_id = entity["id"]
            
            # Check if this entity already exists
            if entity_id in self.entity_index:
                # Merge attributes from both entities
                existing = self.entity_index[entity_id]
                for key, value in entity.items():
                    if key not in existing or not existing[key]:
                        existing[key] = value
            else:
                # Add new entity to index
                self.entity_index[entity_id] = entity
        
        # Process relationships
        for rel in segment.get("relationships", []):
            self.relationship_buffer.append(rel)
        
        # If buffer exceeds limit, flush to disk
        if len(self.relationship_buffer) > self.buffer_size:
            self._flush_relationships()
        
        return segment
    
    def _flush_relationships(self) -> None:
        """Flush relationship buffer to disk."""
        if not self.relationship_buffer:
            return
            
        self.output.append_relationships(self.relationship_buffer)
        self.relationship_buffer = []
    
    def finalize(self) -> Dict:
        """
        Finalize the knowledge graph and return the result.
        
        Returns:
            Complete knowledge graph
        """
        # Flush any remaining relationships
        self._flush_relationships()
        
        # Build the final graph
        entities = list(self.entity_index.values())
        relationships = self.output.get_all_relationships()
        
        return {
            "entities": entities,
            "relationships": relationships
        }


class KGOutputManager:
    """
    Manages output files for knowledge graph components.
    Handles incremental saving of graph segments to reduce memory usage.
    """
    
    def __init__(self, output_dir: Path, project_name: str):
        """
        Initialize with output directory and project name.
        
        Args:
            output_dir: Directory for output files
            project_name: Name of the project for file naming
        """
        self.output_dir = output_dir
        self.project_name = project_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize relationship file
        self.rel_file = self.output_dir / f"{project_name}_relationships.jsonl"
        if not self.rel_file.exists():
            with open(self.rel_file, 'w') as f:
                pass  # Create empty file
    
    def append_relationships(self, relationships: list) -> None:
        """
        Append relationships to the relationship file.
        
        Args:
            relationships: List of relationship objects
        """
        with open(self.rel_file, 'a') as f:
            for rel in relationships:
                f.write(json.dumps(rel) + '\n')
    
    def get_all_relationships(self) -> list:
        """
        Read all relationships from the relationship file.
        
        Returns:
            List of all relationship objects
        """
        relationships = []
        if self.rel_file.exists():
            with open(self.rel_file, 'r') as f:
                for line in f:
                    if line.strip():
                        relationships.append(json.loads(line))
        return relationships
    
    def save_entities(self, entities: list) -> None:
        """
        Save entities to file.
        
        Args:
            entities: List of entity objects
        """
        entity_file = self.output_dir / f"{self.project_name}_entities.json"
        with open(entity_file, 'w') as f:
            json.dump(entities, f, indent=2)
    
    def save_complete_graph(self, graph: Dict) -> None:
        """
        Save the complete knowledge graph.
        
        Args:
            graph: Complete knowledge graph
        """
        graph_file = self.output_dir / f"{self.project_name}_knowledge_graph.json"
        with open(graph_file, 'w') as f:
            json.dump(graph, f, indent=2)


class ChunkedTextProcessor:
    """
    Processes large text documents in manageable chunks.
    Implements sliding window and sentence boundary detection for better chunking.
    """
    
    def __init__(self, chunk_size: int = 4000, overlap: int = 500):
        """
        Initialize with chunk size and overlap parameters.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Overlap between consecutive chunks in characters
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_size(self, text: str) -> list:
        """
        Split text into chunks of specified size with overlap.
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to find sentence boundary for cleaner chunks
            if end < len(text):
                # Look for sentence boundary within the last 20% of the chunk
                search_start = max(end - int(self.chunk_size * 0.2), start)
                sentence_end = text.rfind('. ', search_start, end)
                
                if sentence_end > search_start:
                    end = sentence_end + 1  # Keep the period
            
            chunks.append(text[start:end])
            start = end - self.overlap
        
        return chunks
    
    def chunk_document(self, pages: list) -> list:
        """
        Process a document represented as list of pages.
        
        Args:
            pages: List of page texts
            
        Returns:
            List of text chunks
        """
        chunks = []
        current_chunk = ""
        
        for page in pages:
            # If adding this page would exceed chunk size, start a new chunk
            if len(current_chunk) + len(page) > self.chunk_size:
                # If current chunk is not empty, add it to chunks
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Start a new chunk with this page
                current_chunk = page
            else:
                # Add this page to the current chunk
                if current_chunk:
                    current_chunk += " "
                current_chunk += page
        
        # Add the final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks