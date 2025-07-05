# ===============================================================
# File: src/clear_db.py
# Description: Clears the PostgreSQL database tables used by PGVector.
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

def clear_vector_db_tables():
    """
    Clears the PostgreSQL database by dropping the tables required by LangChain's PGVector store.
    """
    conn = None
    cur = None
    try:
        logging.info(f"Attempting to connect to PostgreSQL: {PGVECTOR_CONN_STRING_PSYCOPG2.split('@')[-1]}")
        conn = psycopg2.connect(PGVECTOR_CONN_STRING_PSYCOPG2)
        # Set isolation level to autocommit for DDL statements (DROP TABLE)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Drop 'langchain_pg_embedding' table if it exists
        logging.info("Checking for and dropping 'langchain_pg_embedding' table...")
        cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;")
        logging.info("✅ 'langchain_pg_embedding' table dropped if it existed.")

        # Drop 'langchain_pg_collection' table if it exists
        logging.info("Checking for and dropping 'langchain_pg_collection' table...")
        cur.execute("DROP TABLE IF EXISTS langchain_pg_collection CASCADE;")
        logging.info("✅ 'langchain_pg_collection' table dropped if it existed.")

        logging.info("🎉 Database tables cleared successfully!")

    except psycopg2.Error as e:
        logging.error(f"❌ PostgreSQL database error during clearing: {e}", exc_info=True)
        logging.error("Please ensure your PostgreSQL server is running and your database credentials in .env are correct.")
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred during database clearing: {e}", exc_info=True)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    clear_vector_db_tables()