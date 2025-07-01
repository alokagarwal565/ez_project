# ===============================================================
# File: src/helpers.py
# ===============================================================

import logging
import base64
import mimetypes
import numpy as np
import streamlit as st
from typing import List, Tuple
import os
import psycopg2
from config import PGVECTOR_CONN_STRING_PSYCOPG2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    try:
        vec1 = np.asarray(vec1, dtype=np.float32)
        vec2 = np.asarray(vec2, dtype=np.float32)
        if not vec1.any() or not vec2.any():
            return 0.0
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return float(dot_product / (norm_vec1 * norm_vec2))
    except Exception as e:
        logging.error(f"Error computing cosine similarity: {e}")
        return 0.0

def create_base64_download_button(file_path: str, label: str = "📄 Download source document"):
    """Create download button for source documents."""
    try:
        if not os.path.exists(file_path):
            st.info(f"Source file not found: {os.path.basename(file_path)}")
            logging.warning(f"File not found for download button: {file_path}")
            return

        with open(file_path, "rb") as f:
            bytes_data = f.read()
        
        b64 = base64.b64encode(bytes_data).decode()
        file_name = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_name)
        if mime_type is None:
            mime_type = "application/octet-stream" # Default if MIME type can't be guessed

        st.markdown(f"""
            <a href="data:{mime_type};base64,{b64}" download="{file_name}">
                <button style="
                    background-color: #4CAF50; /* Green */
                    border: none;
                    color: white;
                    padding: 8px 16px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-flex;
                    font-size: 14px;
                    margin: 4px 2px;
                    cursor: pointer;
                    border-radius: 8px;
                    align-items: center;
                    justify-content: center;
                    gap: 0.5rem;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                    transition: all 0.2s ease-in-out;
                ">
                    {label}: {file_name}
                </button>
            </a>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to generate download link for {os.path.basename(file_path)}. Error: {e}")
        logging.error(f"Error generating download link for '{file_path}': {e}", exc_info=True)
        
# ***************Indexing*******************
def create_hnsw_indexing():
    """
    Creates an HNSW index on the 'embedding' column of the 'langchain_pg_embedding' table
    for faster similarity searches. It checks if the index already exists before creating it.
    """
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(PGVECTOR_CONN_STRING_PSYCOPG2)
        conn.autocommit = True # Enable autocommit for DDL operations
        cursor = conn.cursor()

        # Check if the index already exists
        cursor.execute("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'langchain_pg_embedding'
            AND indexname = 'hnsw_embedding_idx';
        """)
        index_exists = cursor.fetchone()

        if not index_exists:
            logging.info("HNSW index not found, creating 'hnsw_embedding_idx'...")
            # Ensure the table and column exist before attempting to create an index
            # This is a basic check, full table creation is handled by init_db.py or PGVector
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                    id SERIAL PRIMARY KEY,
                    collection_id UUID,
                    embedding VECTOR, -- Type should match your pgvector setup
                    document TEXT,
                    cmetadata JSONB,
                    custom_id TEXT,
                    uuid UUID
                );
            """)
            conn.commit() # Commit table creation if it happened

            cursor.execute("""
                CREATE INDEX hnsw_embedding_idx
                ON langchain_pg_embedding
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 200);
            """)
            logging.info("✅ HNSW index created on langchain_pg_embedding.embedding")
        else:
            logging.info("HNSW index 'hnsw_embedding_idx' already exists. Skipping creation.")

    except psycopg2.errors.UndefinedColumn as e:
        logging.error(f"❌ Error creating HNSW index: Column 'embedding' not found in 'langchain_pg_embedding'. Ensure your PGVector setup is correct. Details: {e}", exc_info=True)
        st.error("Database column error: Ensure PGVector is correctly set up (e.g., `embedding` column exists).")
    except psycopg2.errors.UndefinedTable as e:
        logging.error(f"❌ Error creating HNSW index: Table 'langchain_pg_embedding' not found. Ensure your PGVector setup is correct. Details: {e}", exc_info=True)
        st.error("Database table error: Ensure PGVector is correctly set up (e.g., `langchain_pg_embedding` table exists).")
    except Exception as e:
        logging.error(f"❌ Error creating HNSW index: {e}", exc_info=True)
        st.error(f"Failed to create HNSW index: {e}. Please check your PostgreSQL connection and permissions.")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
