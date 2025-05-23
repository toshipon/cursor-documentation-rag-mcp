#!/usr/bin/env python
"""
Fallback Vector Store Implementation
This script provides a workaround for when SQLite vector search is not available
by implementing a Python-based vector search fallback mechanism.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class FallbackVectorSearch:
    """
    This class provides a fallback implementation of vector similarity search
    when the SQLite vector extension is not available.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the fallback vector search
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self.vector_cache = {}
        self.document_cache = {}
        self.initialized = False
        logger.info(f"Initialized fallback vector search for {db_path}")
    
    def initialize(self, conn):
        """
        Load all vectors from the database into memory
        
        Args:
            conn: SQLite connection
        """
        if self.initialized:
            return
            
        logger.info("Loading vectors into memory for fallback search...")
        try:
            cursor = conn.execute("""
                SELECT d.id, d.content, d.source, d.source_type, d.metadata, d.vector
                FROM documents d
            """)
            
            count = 0
            for row in cursor.fetchall():
                doc_id = row[0]
                content = row[1]
                source = row[2]
                source_type = row[3]
                metadata = json.loads(row[4])
                vector_json = row[5]
                
                # Parse the vector
                try:
                    vector = json.loads(vector_json)
                    self.vector_cache[doc_id] = np.array(vector)
                    self.document_cache[doc_id] = {
                        "id": doc_id,
                        "content": content,
                        "source": source,
                        "source_type": source_type,
                        "metadata": metadata
                    }
                    count += 1
                except Exception as e:
                    logger.warning(f"Error loading vector for doc {doc_id}: {e}")
                    
            logger.info(f"Loaded {count} vectors into memory for fallback search")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing fallback vector search: {e}")
    
    def similarity_search(self, query_vector: List[float], top_k: int = 5, 
                          filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search using in-memory vectors
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filter_criteria: Optional filtering criteria
            
        Returns:
            List of matching documents with scores
        """
        if (not self.initialized or not self.vector_cache) and self.db_path:
            # Try to initialize on demand
            import sqlite3
            try:
                conn = sqlite3.connect(self.db_path)
                self.initialize(conn)
            except Exception as e:
                logger.error(f"Error initializing on demand: {e}")
                
        if not self.initialized or not self.vector_cache:
            logger.warning("Fallback search not initialized or empty cache")
            return []
            
        logger.info(f"Performing fallback similarity search with top_k={top_k}")
        
        try:
            # Convert query vector to numpy array
            query_np = np.array(query_vector)
            
            # Calculate similarities for all vectors
            results = []
            for doc_id, vec in self.vector_cache.items():
                if vec is not None and len(vec) > 0:
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_np, vec)
                    
                    # Check if document passes filters
                    if self._passes_filter(doc_id, filter_criteria):
                        doc = self.document_cache[doc_id].copy()
                        doc["score"] = float(similarity)
                        results.append(doc)
            
            # Sort by similarity (highest first) and take top_k
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
            return results
            
        except Exception as e:
            logger.error(f"Error in fallback similarity search: {e}")
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        try:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(vec1, vec2) / (norm1 * norm2)
        except Exception:
            return 0.0
    
    def _passes_filter(self, doc_id: int, filter_criteria: Optional[Dict[str, Any]]) -> bool:
        """
        Check if a document passes the filter criteria
        
        Args:
            doc_id: Document ID
            filter_criteria: Filter criteria
            
        Returns:
            True if document passes filter, False otherwise
        """
        if not filter_criteria:
            return True
            
        doc = self.document_cache.get(doc_id)
        if not doc:
            return False
            
        # Check source type filter
        if "source_type" in filter_criteria and doc["source_type"] != filter_criteria["source_type"]:
            return False
            
        # Check source filter
        if "source" in filter_criteria and doc["source"] != filter_criteria["source"]:
            return False
            
        # Check metadata filters
        if "metadata" in filter_criteria:
            metadata = doc["metadata"]
            for k, v in filter_criteria["metadata"].items():
                if k not in metadata or metadata[k] != v:
                    return False
        
        return True
    
    def update_vector(self, doc_id: int, vector: List[float], document_data: Dict[str, Any]):
        """
        Update a vector in the cache
        
        Args:
            doc_id: Document ID
            vector: Vector data
            document_data: Document data
        """
        try:
            self.vector_cache[doc_id] = np.array(vector)
            self.document_cache[doc_id] = document_data
        except Exception as e:
            logger.error(f"Error updating vector in fallback cache: {e}")
    
    def remove_vector(self, doc_id: int):
        """
        Remove a vector from the cache
        
        Args:
            doc_id: Document ID
        """
        if doc_id in self.vector_cache:
            del self.vector_cache[doc_id]
        if doc_id in self.document_cache:
            del self.document_cache[doc_id]
