# ===============================================================
# File: src/init_db.py
# Description: Initializes the PostgreSQL database with tables required by PGVector.
# ===============================================================

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging

# Configure logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import database connection string from config.py
try:
    from config import PGVECTOR_CONN_STRING_PSYCOPG2
except ImportError:
    logging.error("Error: config.py not found or PGVECTOR_CONN_STRING_PSYCOPG2 not defined.")
    logging.error("Please ensure config.py is in the 'src' directory and contains the PostgreSQL connection string.")
    exit(1) # Exit if config is not found

def init_vector_db_tables():
    """
    Initializes the PostgreSQL database by creating the 'vector' extension
    and the tables required by LangChain's PGVector store.
    """
    conn = None
    cur = None
    try:
        logging.info(f"Attempting to connect to PostgreSQL: {PGVECTOR_CONN_STRING_PSYCOPG2.split('@')[-1]}")
        conn = psycopg2.connect(PGVECTOR_CONN_STRING_PSYCOPG2)
        # Set isolation level to autocommit for DDL statements (CREATE EXTENSION, CREATE TABLE)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # 1. Create the 'vector' extension if it doesn't exist
        logging.info("Checking for 'vector' extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logging.info("✅ 'vector' extension ensured.")

        # 2. Create 'langchain_pg_collection' table if it doesn't exist
        logging.info("Checking for 'langchain_pg_collection' table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS langchain_pg_collection (
                uuid UUID PRIMARY KEY,
                name VARCHAR UNIQUE,
                cmetadata JSONB
            );
        """)
        logging.info("✅ 'langchain_pg_collection' table ensured.")

        # 3. Create 'langchain_pg_embedding' table if it doesn't exist
        logging.info("Checking for 'langchain_pg_embedding' table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                id SERIAL PRIMARY KEY,
                collection_id UUID REFERENCES langchain_pg_collection (uuid) ON DELETE CASCADE,
                embedding VECTOR(384), -- CORRECTED: Added dimension (384)
                document TEXT,
                cmetadata JSONB,
                custom_id TEXT,
                uuid UUID UNIQUE
            );
        """)
        logging.info("✅ 'langchain_pg_embedding' table ensured.")

        logging.info("🎉 Database schema initialized successfully!")

    except psycopg2.Error as e:
        logging.error(f"❌ PostgreSQL database error during initialization: {e}", exc_info=True)
        logging.error("Please ensure your PostgreSQL server is running, your database credentials in .env are correct, and the specified database exists.")
        logging.error("If 'vector' extension creation fails, you might need to install it manually on your PostgreSQL server.")
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred during database initialization: {e}", exc_info=True)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    init_vector_db_tables()
