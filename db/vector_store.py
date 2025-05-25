import os
import json
import time
import sqlite3
import logging
import uuid
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Attempt to find the sqlite-vss extension
SQLITE_VSS_PATHS = [
    'vss0',                 # Common name
    'libsqlitevss.so',      # Linux
    'libsqlitevss.dylib',   # macOS
    'sqlitevss.dll'         # Windows
]

class VectorStore:
    """SQLite-based vector store with VSS extension support."""
    
    def __init__(self, db_path: str, vector_dimension: int = 512, vss_enabled: bool = True):
        """
        Initializes the VectorStore.

        Args:
            db_path: Path to the SQLite database file.
            vector_dimension: Dimension of the vectors.
            vss_enabled: Flag to enable or disable VSS extension usage.
        """
        self.db_path = db_path
        self.vector_dimension = vector_dimension
        self.vss_enabled = vss_enabled
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self._load_vss_extension()
        self._init_db()
        logger.info(f"Initialized SQLite vector store at {db_path} (VSS enabled: {self.vss_active})")

    def _load_vss_extension(self):
        """Loads the SQLite-VSS extension if enabled and available."""
        self.vss_active = False
        if not self.vss_enabled:
            logger.info("SQLite-VSS extension usage is disabled by configuration.")
            return

        try:
            # In-memory DB for testing VSS loading without affecting the main DB yet
            conn = sqlite3.connect(':memory:')
            conn.enable_load_extension(True)
            
            loaded_path = None
            for path in SQLITE_VSS_PATHS:
                try:
                    conn.load_extension(path)
                    loaded_path = path
                    break
                except sqlite3.OperationalError:
                    continue
            
            if loaded_path:
                # Check if VSS functions are available
                cursor = conn.execute("SELECT vss_version()")
                version = cursor.fetchone()[0]
                logger.info(f"SQLite-VSS extension loaded successfully (version: {version}) from path: {loaded_path}.")
                self.vss_active = True
            else:
                logger.warning(
                    "SQLite-VSS extension not found or failed to load from known paths. "
                    "Vector search will use manual cosine similarity."
                )
        except Exception as e:
            logger.error(f"Error during SQLite-VSS extension loading test: {e}. VSS will be disabled.")
        finally:
            if conn:
                conn.close()
    
    def _init_db(self):
        """Initializes database tables, including VSS virtual table if active."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable VSS for this connection if it was successfully loaded
                if self.vss_active:
                    conn.enable_load_extension(True) # Redundant if loaded via SQLITE_VSS_PATHS, but good practice
                    # Attempt to load again for this specific connection.
                    # This might be needed if the test load was on :memory:
                    loaded_this_conn = False
                    for path in SQLITE_VSS_PATHS:
                        try:
                            conn.load_extension(path)
                            loaded_this_conn = True
                            break
                        except sqlite3.OperationalError:
                            continue
                    if not loaded_this_conn and self.vss_active: # If vss_active was true from test, but failed here
                        logger.warning(f"VSS was active in test, but failed to load for main DB connection {self.db_path}. Disabling VSS features.")
                        self.vss_active = False


                # Main documents table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        vector BLOB,
                        metadata TEXT,
                        source TEXT  -- Extracted from metadata for easier querying
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source)")

                # File metadata table for incremental updates
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS file_metadata (
                        file_path TEXT PRIMARY KEY,
                        last_modified REAL NOT NULL,
                        file_hash TEXT NOT NULL,
                        last_vectorized REAL,
                        status TEXT  -- e.g., 'processed', 'failed', 'pending'
                    )
                """)
                
                if self.vss_active:
                    # VSS virtual table, depends on the actual VSS extension (faiss, etc.)
                    # Assuming standard sqlite-vss syntax for creating a virtual table.
                    # The vector column in 'documents' is named 'vector'.
                    # The number of dimensions is self.vector_dimension.
                    conn.execute(f"""
                        CREATE VIRTUAL TABLE IF NOT EXISTS vss_documents USING vss0(
                            vector({self.vector_dimension})
                        )
                    """)
                    # Note: vss_documents would typically store rowid from 'documents' and the vector.
                    # Data is inserted into vss_documents by referencing rowid of the main table.
                    # Example: INSERT INTO vss_documents(rowid, vector) VALUES (new_document_rowid, document_vector);

                logger.debug("Database tables and indices initialized.")
        except Exception as e:
            logger.error(f"Error initializing database at {self.db_path}: {e}", exc_info=True)
            raise

    def add_documents(self, chunks: List[Dict[str, Any]], vectors: List[np.ndarray]):
        """Adds documents and their vectors to the store."""
        if not chunks or not vectors:
            logger.warning("No chunks or vectors provided.")
            return
        if len(chunks) != len(vectors):
            logger.error(f"Mismatch: {len(chunks)} chunks, {len(vectors)} vectors.")
            raise ValueError("Number of chunks must match number of vectors.")

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Re-enable VSS for this connection if active
                if self.vss_active:
                    conn.enable_load_extension(True)
                    for path in SQLITE_VSS_PATHS: # Ensure VSS is loaded for this connection
                        try: conn.load_extension(path); break
                        except: pass
                
                cursor = conn.cursor()
                for chunk, vector_np in zip(chunks, vectors):
                    doc_id = str(uuid.uuid4().hex)
                    text_content = chunk.get("text", "")
                    metadata = chunk.get("metadata", {})
                    source_path = metadata.get("source", "unknown_source") # Ensure source is present

                    # Serialize vector to BLOB
                    vector_blob = vector_np.astype(np.float32).tobytes()

                    cursor.execute(
                        """
                        INSERT INTO documents (id, text, vector, metadata, source)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (doc_id, text_content, vector_blob, json.dumps(metadata), source_path)
                    )
                    
                    if self.vss_active:
                        # Insert into VSS table using the rowid of the just-inserted document
                        last_row_id = cursor.lastrowid
                        cursor.execute(
                            "INSERT INTO vss_documents (rowid, vector) VALUES (?, ?)",
                            (last_row_id, vector_blob)
                        )
                conn.commit()
            logger.info(f"Added {len(chunks)} documents to vector store.")
        except Exception as e:
            logger.error(f"Error adding documents: {e}", exc_info=True)
            raise

    def similarity_search(self, query_vector: np.ndarray, top_k: int = 5,
                          filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Performs similarity search."""
        results = []
        query_vector_blob = query_vector.astype(np.float32).tobytes()

        try:
            with sqlite3.connect(self.db_path) as conn:
                if self.vss_active:
                    conn.enable_load_extension(True) # Ensure VSS is loaded
                    for path in SQLITE_VSS_PATHS:
                        try: conn.load_extension(path); break
                        except: pass

                    # Build query for VSS
                    # Base query to find nearest neighbors
                    vss_query = """
                        SELECT d.id, d.text, d.metadata, r.distance as score
                        FROM vss_documents r
                        JOIN documents d ON d.rowid = r.rowid
                    """
                    
                    # Filtering
                    filter_clauses = []
                    filter_params = []
                    if filters:
                        for key, value in filters.items():
                            if key == "source": # Direct column
                                filter_clauses.append(f"d.source = ?")
                                filter_params.append(value)
                            else: # Assume other filters are on metadata JSON
                                filter_clauses.append(f"json_extract(d.metadata, '$.{key}') = ?")
                                filter_params.append(value)
                    
                    if filter_clauses:
                        vss_query += " WHERE " + " AND ".join(filter_clauses)
                    
                    # Add VSS search condition
                    # If there are other where clauses, VSS search might need to be a subquery or CTE
                    # For simplicity, adding it here. This might need adjustment depending on how VSS extension handles WHERE + vss_search.
                    # A common pattern is to get rowids from VSS then join and filter.
                    # Let's refine: get rowids and distances first.
                    
                    base_vss_search_sql = "SELECT rowid, distance FROM vss_documents WHERE vss_search(vector, ?) "
                    # If filters are present, it's often better to apply them AFTER the VSS search
                    # or use them in a way the VSS extension can optimize.
                    # For now, let's get top N from VSS and then filter, though this is suboptimal.
                    # A better way: SELECT ... WHERE rowid IN (SELECT rowid FROM vss_documents WHERE vss_search(...) AND <simple_conditions_on_vss_table>)
                    # And then apply JSON filters on the main table.
                    
                    # Simpler approach: Get top_k * factor from VSS, then filter and re-sort.
                    # Or, if VSS supports it, include simple filters in its query.
                    # For now, a basic VSS search, then filtering on the joined documents table.
                    
                    # This query applies filters directly, which might not be efficient with all VSS extensions.
                    # It relies on the VSS extension to handle the vss_search condition alongside other WHERE clauses.
                    vss_query += (" AND " if filter_clauses else " WHERE ") + "vss_search(d.vector, ?)" # Assuming d.vector can be used by vss_search if vss_documents is just index
                                                                                                 # This is incorrect. vss_search is on vss_documents.vector
                    
                    # Corrected VSS query structure:
                    # 1. Search in vss_documents
                    # 2. Join with documents table
                    # 3. Apply filters
                    
                    sql_query = f"""
                        SELECT d.id, d.text, d.metadata, r.distance as score
                        FROM (
                            SELECT rowid, distance 
                            FROM vss_documents
                            WHERE vss_search(vector, ?)
                            ORDER BY distance 
                            LIMIT ? 
                        ) r
                        JOIN documents d ON d.rowid = r.rowid
                    """
                    params = [query_vector_blob, top_k * 2] # Fetch more to filter later, if filters are complex

                    # If filters are present, apply them in the outer query
                    if filter_clauses:
                        sql_query += " WHERE " + " AND ".join(filter_clauses)
                        params.extend(filter_params)
                    
                    sql_query += " ORDER BY score LIMIT ?" # Final limit
                    params.append(top_k)
                    
                    logger.debug(f"Executing VSS search: {sql_query} with {len(params)} params")
                    cursor = conn.execute(sql_query, params)

                else: # Manual cosine similarity search (fallback)
                    logger.warning("VSS extension not active. Performing manual cosine similarity search.")
                    # Build manual query
                    sql_query = "SELECT id, text, vector, metadata, source FROM documents"
                    filter_params_manual = []
                    if filters:
                        filter_clauses_manual = []
                        for key, value in filters.items():
                            if key == "source":
                                filter_clauses_manual.append(f"source = ?")
                                filter_params_manual.append(value)
                            else:
                                filter_clauses_manual.append(f"json_extract(metadata, '$.{key}') = ?")
                                filter_params_manual.append(value)
                        if filter_clauses_manual:
                            sql_query += " WHERE " + " AND ".join(filter_clauses_manual)
                    
                    logger.debug(f"Executing manual search: {sql_query} with {len(filter_params_manual)} params")
                    cursor = conn.execute(sql_query, filter_params_manual)
                    
                    # Calculate similarities manually
                    all_docs = []
                    for row_id, text_content, vector_blob_db, metadata_json, _ in cursor:
                        db_vector = np.frombuffer(vector_blob_db, dtype=np.float32)
                        similarity = self._cosine_similarity(query_vector, db_vector)
                        all_docs.append({
                            "id": row_id, "text": text_content, 
                            "metadata": json.loads(metadata_json), "score": similarity
                        })
                    all_docs.sort(key=lambda x: x["score"], reverse=True)
                    return all_docs[:top_k]

                # Process results from VSS search
                for row_id, text_content, metadata_json, score in cursor:
                    results.append({
                        "id": row_id,
                        "text": text_content,
                        "metadata": json.loads(metadata_json),
                        "score": score # VSS typically returns distance, so smaller is better. Or it might be configurable.
                                       # If distance, might want to convert to similarity: 1 / (1 + distance) or exp(-distance)
                    })
                # If VSS returns distance, and we filtered AFTER fetching top_k*factor, re-sort and limit
                # The refined query already handles LIMIT top_k.
                
        except Exception as e:
            logger.error(f"Error during similarity search: {e}", exc_info=True)
        return results

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Computes cosine similarity between two numpy vectors."""
        if vec1.shape != vec2.shape: # Should not happen if dimension is fixed
            logger.error(f"Vector shape mismatch for cosine similarity: {vec1.shape} vs {vec2.shape}")
            return 0.0
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def delete_by_source(self, file_path: str) -> int:
        """Deletes documents associated with a given source file_path."""
        deleted_count = 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                if self.vss_active: # Ensure VSS is loaded for this connection
                    conn.enable_load_extension(True)
                    for path in SQLITE_VSS_PATHS:
                        try: conn.load_extension(path); break
                        except: pass
                
                # First, get rowids of documents to be deleted to remove them from VSS table
                if self.vss_active:
                    cursor_rowids = conn.execute("SELECT rowid FROM documents WHERE source = ?", (file_path,))
                    rowids_to_delete = [row[0] for row in cursor_rowids]
                    if rowids_to_delete:
                        conn.executemany("DELETE FROM vss_documents WHERE rowid = ?", [(r,) for r in rowids_to_delete])
                
                # Then, delete from the main documents table
                cursor = conn.execute("DELETE FROM documents WHERE source = ?", (file_path,))
                deleted_count = cursor.rowcount
                conn.commit()

                # Also delete from file_metadata table
                conn.execute("DELETE FROM file_metadata WHERE file_path = ?", (file_path,))
                conn.commit()
                
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} documents for source: {file_path}.")
            if self.get_file_metadata(file_path) is None: # Check if deleted from file_metadata
                 logger.info(f"Deleted metadata for source: {file_path} from file_metadata table.")
        except Exception as e:
            logger.error(f"Error deleting documents for source {file_path}: {e}", exc_info=True)
        return deleted_count

    def _get_file_hash(self, file_path: str) -> Optional[str]:
        """Computes SHA256 hash of a file."""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192): # Read in 8KB chunks
                    hasher.update(chunk)
            return hasher.hexdigest()
        except FileNotFoundError:
            logger.warning(f"File not found for hashing: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}", exc_info=True)
            return None

    def get_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Retrieves file metadata from the file_metadata table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT file_path, last_modified, file_hash, last_vectorized, status FROM file_metadata WHERE file_path = ?",
                    (file_path,)
                )
                row = cursor.fetchone()
                if row:
                    return {"file_path": row[0], "last_modified": row[1], "file_hash": row[2], 
                            "last_vectorized": row[3], "status": row[4]}
        except Exception as e:
            logger.error(f"Error getting file metadata for {file_path}: {e}", exc_info=True)
        return None

    def update_file_metadata(self, file_path: str, last_modified: float, file_hash: str, 
                             status: str = "processed", last_vectorized: Optional[float] = None):
        """Updates or inserts file metadata into the file_metadata table."""
        if last_vectorized is None:
            last_vectorized = time.time()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO file_metadata 
                    (file_path, last_modified, file_hash, last_vectorized, status)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (file_path, last_modified, file_hash, last_vectorized, status)
                )
                conn.commit()
            logger.debug(f"Updated file metadata for {file_path} with status {status}.")
        except Exception as e:
            logger.error(f"Error updating file metadata for {file_path}: {e}", exc_info=True)

    def file_needs_processing(self, file_path: str) -> bool:
        """
        Checks if a file needs processing based on its modification time and hash
        stored in the file_metadata table.
        """
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} does not exist, cannot determine if needs processing.")
            return False # Or handle as a deletion if it was previously tracked

        current_mtime = os.path.getmtime(file_path)
        stored_meta = self.get_file_metadata(file_path)

        if not stored_meta:
            logger.info(f"File {file_path} not found in metadata, needs processing.")
            return True # New file

        if stored_meta['status'] == 'failed':
            logger.info(f"File {file_path} previously failed, needs reprocessing.")
            return True

        if current_mtime > stored_meta['last_modified']:
            logger.info(f"File {file_path} modified since last check (mtime), needs processing.")
            # Optionally, verify with hash if mtime changed but content might be same
            current_hash = self._get_file_hash(file_path)
            if current_hash != stored_meta['file_hash']:
                logger.info(f"File {file_path} hash changed, needs processing.")
                return True
            else:
                # Update mtime even if hash is same, to avoid re-checking hash next time
                self.update_file_metadata(file_path, current_mtime, current_hash, stored_meta['status'], stored_meta['last_vectorized'])
                logger.info(f"File {file_path} mtime changed but hash is identical. Marked as processed with new mtime.")
                return False 
        
        logger.debug(f"File {file_path} does not need processing based on mtime and hash.")
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Retrieves statistics about the vector store."""
        try:
        stats = {"total_documents": 0, "total_files": 0, "vector_dimension": self.vector_dimension, "vss_active": self.vss_active}
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total documents in the main table
                cursor_docs = conn.execute("SELECT COUNT(*) FROM documents")
                stats["total_documents"] = cursor_docs.fetchone()[0]
                
                # Total distinct source files
                cursor_files = conn.execute("SELECT COUNT(DISTINCT source) FROM documents")
                stats["total_files"] = cursor_files.fetchone()[0]

                if self.vss_active:
                    # Potentially add VSS specific stats if available, e.g., count from vss_table
                    # cursor_vss = conn.execute("SELECT COUNT(*) FROM vss_documents")
                    # stats["vss_indexed_count"] = cursor_vss.fetchone()[0]
                    pass

        except Exception as e:
            logger.error(f"Error getting stats from vector store: {e}", exc_info=True)
        return stats
    
    def close(self):
        """Placeholder for cleanup, if any. SQLite connections are usually managed per transaction."""
        logger.info("VectorStore close called (typically no-op for SQLite connection handling).")
        pass
