import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Application Paths ---
# Folder to temporarily store uploaded documents
DOC_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
# Folder to store prompt templates
PROMPT_TEMPLATES_FOLDER = os.path.join(os.path.dirname(__file__), "prompt_templates")

# Ensure the document folder exists
os.makedirs(DOC_FOLDER, exist_ok=True)

# --- LLM and Embedding Model Settings ---
# Chat model to be used for Q&A, summary, and challenge mode (e.g., "gemini-1.5-flash-latest")
CHAT_MODEL = "gemini-1.5-flash-latest"
# Embedding model for vectorizing documents (e.g., "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
# Collection name for the PGVector database
COLLECTION_NAME = "research_summarization_collection"

# --- Google API Settings ---
# Google API Key for Gemini. Loaded from .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- PostgreSQL Connection Settings for PGVector ---
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DBNAME = os.getenv("PG_DBNAME")

# PostgreSQL connection string for psycopg2 (used by PGVector for some operations)
PGVECTOR_CONN_STRING_PSYCOPG2 = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}"
# PostgreSQL connection string for SQLAlchemy (used by LangChain's PGVector)
PGVECTOR_CONN_STRING_SQLACHEMY = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}"

# --- Logging Configuration ---
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Validation ---
if not GOOGLE_API_KEY:
    logging.warning("GOOGLE_API_KEY is not set. Gemini API calls will fail.")
if not all([PG_USER, PG_PASSWORD, PG_HOST, PG_PORT, PG_DBNAME]):
    logging.warning("PostgreSQL environment variables are not fully set. Database operations might fail.")
