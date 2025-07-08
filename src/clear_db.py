# ===============================================================
# File: src/clear_db.py
# Description: Clears the PostgreSQL database tables used by PGVector and chat sessions.
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

def clear_all_vector_db_and_chat_tables():
    """
    Clears the PostgreSQL database by dropping all tables related to LangChain's PGVector store
    and chat sessions. Use with caution!
    """
    conn = None
    cur = None
    try:
        logging.info(f"Attempting to connect to PostgreSQL: {PGVECTOR_CONN_STRING_PSYCOPG2.split('@')[-1]}")
        conn = psycopg2.connect(PGVECTOR_CONN_STRING_PSYCOPG2)
        # Set isolation level to autocommit for DDL statements (DROP TABLE)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Drop tables in reverse order of dependency to avoid foreign key issues
        tables_to_drop = [
            "chat_history",
            "langchain_pg_embedding",
            "langchain_pg_collection",
            "chat_sessions"
        ]

        for table in tables_to_drop:
            logging.info(f"Checking for and dropping '{table}' table...")
            try:
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
                logging.info(f"✅ '{table}' table dropped if it existed.")
            except psycopg2_errors.InsufficientPrivilege:
                logging.error(f"❌ Insufficient privileges to drop '{table}' table. Please ensure your PostgreSQL user has appropriate permissions.")
                raise # Re-raise to be caught by outer except block
            except Exception as e:
                logging.error(f"❌ Error dropping '{table}' table: {e}", exc_info=True)
                raise # Re-raise to be caught by outer except block

        logging.info("🎉 All database tables cleared successfully!")

    except OperationalError as e:
        logging.error(f"❌ PostgreSQL database connection error during clearing: {e}", exc_info=True)
        logging.error("Actionable: Database connection failed. Please ensure your PostgreSQL server is running and your database credentials in .env are correct.")
    except psycopg2_errors.InsufficientPrivilege as e:
        logging.error(f"❌ PostgreSQL permission error during clearing: {e}", exc_info=True)
        logging.error("Actionable: Your PostgreSQL user lacks necessary privileges to drop tables. Grant appropriate permissions.")
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred during database clearing: {e}", exc_info=True)
        logging.error("Actionable: Review the logs for more details on this unexpected error.")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    clear_all_vector_db_and_chat_tables()
