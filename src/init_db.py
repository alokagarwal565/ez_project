# ===============================================================
# File: src/init_db.py
# Description: Initializes the PostgreSQL database with tables required by PGVector.
# ===============================================================

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2 import OperationalError, errors as psycopg2_errors # Import specific psycopg2 errors
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
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logging.info("✅ 'vector' extension ensured.")
        except psycopg2_errors.InsufficientPrivilege:
            logging.error("❌ Insufficient privileges to create 'vector' extension. Please ensure your PostgreSQL user has CREATE privilege on the database or install the extension manually.")
            raise # Re-raise to be caught by outer except block
        except Exception as e:
            logging.error(f"❌ Error creating 'vector' extension: {e}", exc_info=True)
            raise # Re-raise to be caught by outer except block


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
        # Removed backslashes from the end of lines in the SQL query
        cur.execute("""
            CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                id SERIAL PRIMARY KEY,
                collection_id UUID REFERENCES langchain_pg_collection (uuid) ON DELETE CASCADE,
                embedding VECTOR(384),
                document TEXT,
                cmetadata JSONB,
                custom_id TEXT,
                uuid UUID UNIQUE
            );
        """)
        logging.info("✅ 'langchain_pg_embedding' table ensured.")

        logging.info("🎉 Database schema initialized successfully!")

    except OperationalError as e:
        logging.error(f"❌ PostgreSQL connection error during initialization: {e}", exc_info=True)
        logging.error("Please ensure your PostgreSQL server is running, your database credentials in .env are correct, and the specified database exists.")
        logging.error("Actionable: Check your .env file for PG_USER, PG_PASSWORD, PG_HOST, PG_PORT, PG_DBNAME. Also, verify PostgreSQL server status.")
    except psycopg2_errors.InsufficientPrivilege as e:
        logging.error(f"❌ PostgreSQL permission error during initialization: {e}", exc_info=True)
        logging.error("Actionable: Your PostgreSQL user lacks necessary privileges (e.g., CREATE EXTENSION). Grant appropriate permissions or install 'vector' extension manually.")
    except psycopg2_errors.UndefinedTable as e:
        logging.error(f"❌ PostgreSQL table definition error during initialization: {e}", exc_info=True)
        logging.error("Actionable: A table definition might be incorrect or a dependency is missing. Review table schemas.")
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred during database initialization: {e}", exc_info=True)
        logging.error("Actionable: Review the logs for more details on this unexpected error.")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    init_vector_db_tables()
