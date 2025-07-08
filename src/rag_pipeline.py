# ===============================================================
# File: src/rag_pipeline.py
# ===============================================================
import logging
import os
import pandas as pd
import streamlit as st
from typing import List, Any, Tuple
from langchain_community.document_loaders import PyPDFLoader, TextLoader # Removed CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import PGVector
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from helpers import create_hnsw_indexing
from PIL import Image
import numpy as np
import psycopg2
from psycopg2 import OperationalError, errors as psycopg2_errors # Import specific psycopg2 errors
import google.generativeai as genai
import json
import re # Import regex module

from config import (
    DOC_FOLDER,
    CHAT_MODEL,
    EMBEDDING_MODEL_NAME,
    COLLECTION_NAME,
    PGVECTOR_CONN_STRING_SQLACHEMY,
    PGVECTOR_CONN_STRING_PSYCOPG2,
    GOOGLE_API_KEY
)
from prompt_templates import RAG_PROMPT_TEMPLATE, SUMMARY_PROMPT_TEMPLATE, CHUNK_SUMMARY_PROMPT_TEMPLATE, DOCUMENT_COMPARISON_PROMPT

# Define a heuristic for maximum tokens for direct summarization
# This is a rough estimate and might need fine-tuning based on actual LLM token limits and performance
MAX_DIRECT_SUMMARY_TOKENS = 8000 # Roughly equivalent to 4000 words, leaves room for prompt

@st.cache_resource
def get_embedding_model():
    """Loads the SBERT embedding model from HuggingFace."""
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        logging.info("✅ HuggingFace Embedding Model loaded successfully.")
        return embedding_model
    except Exception as e:
        logging.error(f"❌ Error loading HuggingFace Embedding Model: {e}", exc_info=True)
        st.error(f"Failed to load embedding model. Please check your internet connection, model name, or if the model is correctly installed. Error: {e}")
        return None

@st.cache_resource
def get_gemini_rag_llm():
    """Initializes Gemini LLM for RAG."""
    if not GOOGLE_API_KEY:
        st.error("Google API Key is required for Gemini LLM. Please set GOOGLE_API_KEY in your .env file.")
        return None
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        llm = genai.GenerativeModel(CHAT_MODEL) # Use CHAT_MODEL from config
        logging.info(f"✅ Gemini GenerativeModel ({CHAT_MODEL}) initialized successfully for RAG.")
        return llm
    except Exception as e:
        logging.error(f"❌ Error initializing Gemini LLM: {e}", exc_info=True)
        st.error(f"Failed to initialize Gemini LLM. Please check your Google API Key in the .env file and ensure it's valid. Error: {e}")
        return None


@st.cache_resource
def load_existing_vector_db(_embedding_model: Any) -> PGVector | None:
    """Loads the existing vector database."""
    try:
        vector_db = PGVector(
            embedding_function=_embedding_model,
            collection_name=COLLECTION_NAME,
            connection_string=PGVECTOR_CONN_STRING_SQLACHEMY,
        )
        logging.info("✅ Existing vector DB loaded successfully.")
        create_hnsw_indexing() # Ensure HNSW index is created for performance
        return vector_db
    except OperationalError as e:
        logging.error(f"❌ PostgreSQL connection error when loading vector DB: {e}", exc_info=True)
        st.error(f"Database connection failed when loading knowledge base. Please ensure PostgreSQL is running and your database credentials in .env are correct. Error: {e}")
        return None
    except Exception as e:
        logging.error(f"❌ Error loading existing vector DB: {e}", exc_info=True)
        st.error(f"Failed to load vector database. Ensure PostgreSQL is running, PGVector extension is installed, and tables are initialized. Error: {e}")
        return None

def load_documents(file_path: str) -> List[Document]:
    """Load documents from a given file path."""
    loader = None
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    # Removed CSVLoader
    else:
        logging.warning(f"Unsupported file type for {file_path}. Skipping.")
        st.warning(f"Unsupported file type: {os.path.basename(file_path)}. Please upload PDF or TXT.") # Updated message
        return []

    if loader:
        try:
            documents = loader.load()
            logging.info(f"Loaded {len(documents)} documents from {os.path.basename(file_path)}")
            return documents
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}", exc_info=True)
            st.error(f"Error loading document '{os.path.basename(file_path)}'. Please check the file's integrity. Error: {e}")
            return []
    return []

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks.")
    return chunks

def create_vector_db_and_rebuild(embedding_model: Any, documents_to_process: List[str] = None, progress_bar=None, status_text=None) -> int:
    """
    Creates/rebuilds the vector database by processing documents.
    If documents_to_process is None, processes all documents in DOC_FOLDER.
    Otherwise, processes only the specified file paths.
    Returns the number of chunks processed.
    """
    if status_text: status_text.info("🔄 Rebuilding knowledge base. This may take a moment...")
    if progress_bar: progress_bar.progress(0, text="Starting knowledge base rebuild...")

    try:
        all_file_paths = []
        if documents_to_process:
            all_file_paths = documents_to_process
            logging.info(f"Processing specific documents for rebuild: {len(all_file_paths)} files.")
        else:
            logging.info(f"Processing all documents in {DOC_FOLDER} for rebuild.")
            for root, _, files in os.walk(DOC_FOLDER):
                for file in files:
                    file_path = os.path.join(root, file)
                    all_file_paths.append(file_path)

        if not all_file_paths:
            st.warning("No documents found to process for the knowledge base.")
            if progress_bar: progress_bar.progress(100, text="No documents to process.")
            return 0

        documents = []
        for i, file_path in enumerate(all_file_paths):
            if status_text: status_text.info(f"Loading document: {os.path.basename(file_path)}...")
            if progress_bar: progress_bar.progress(int((i / len(all_file_paths)) * 20), text=f"Loading document: {os.path.basename(file_path)}...")
            docs = load_documents(file_path)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for doc in docs:
                # Store relative path in metadata for multi-document Q&A and comparison
                doc.metadata["source"] = os.path.relpath(file_path, project_root)
            documents.extend(docs)

        if not documents:
            st.warning("No valid documents loaded for processing. Knowledge base not rebuilt.")
            if progress_bar: progress_bar.progress(100, text="No valid documents loaded.")
            return 0

        if status_text: status_text.info("Chunking text...")
        if progress_bar: progress_bar.progress(20, text="Chunking text...")
        chunks = split_documents(documents)

        # Clear existing collection before adding new documents to ensure a "rebuild"
        conn = None
        cur = None
        try:
            if status_text: status_text.info(f"Clearing existing data for collection '{COLLECTION_NAME}'...")
            if progress_bar: progress_bar.progress(40, text=f"Clearing existing data for collection '{COLLECTION_NAME}'...")
            conn = psycopg2.connect(PGVECTOR_CONN_STRING_PSYCOPG2)
            conn.autocommit = True
            cur = conn.cursor()
            # Get collection_id for the specific collection name
            cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (COLLECTION_NAME,))
            collection_uuid_result = cur.fetchone()
            if collection_uuid_result:
                collection_uuid = collection_uuid_result[0]
                logging.info(f"Deleting existing documents from collection '{COLLECTION_NAME}' ({collection_uuid})")
                cur.execute("DELETE FROM langchain_pg_embedding WHERE collection_id = %s;", (collection_uuid,))
                conn.commit()
                logging.info("Existing documents cleared from collection.")
            else:
                logging.info(f"Collection '{COLLECTION_NAME}' does not exist, no documents to clear.")
        except OperationalError as e:
            logging.error(f"❌ PostgreSQL connection error during clearing: {e}", exc_info=True)
            st.error(f"Database connection failed during data clearing. Please ensure PostgreSQL is running and your database credentials in .env are correct. Error: {e}")
            return 0
        except psycopg2_errors.UndefinedTable as e:
            logging.warning(f"Collection table not found during clearing (might be first run): {e}")
            st.warning("Database tables not found. Attempting to create them (this is normal on first run).")
        except Exception as e:
            logging.warning(f"Could not clear existing documents from collection '{COLLECTION_NAME}': {e}", exc_info=True)
            st.warning(f"Failed to clear existing documents from knowledge base. Proceeding with adding new documents. Error: {e}")
        finally:
            if cur: cur.close()
            if conn: conn.close()

        logging.info("Creating/updating vector database with new chunks.")
        if status_text: status_text.info("Embedding chunks and adding to database...")
        if progress_bar: progress_bar.progress(60, text="Embedding chunks and adding to database...")

        vector_db = PGVector(
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
            connection_string=PGVECTOR_CONN_STRING_SQLACHEMY,
            # pre_delete_collection=True # This option is for dropping the entire table, not just clearing a collection
        )

        vector_db.add_documents(chunks)

        logging.info(f"✅ Vector database rebuilt with {len(chunks)} chunks.")
        if status_text: st.success(f"✅ Knowledge base rebuilt with {len(chunks)} chunks from {len(all_file_paths)} files.")
        if progress_bar: progress_bar.progress(100, text="Knowledge base rebuilt successfully!")
        return len(chunks) # Return number of chunks
    except OperationalError as e:
        logging.error(f"❌ PostgreSQL connection error during vector DB rebuild: {e}", exc_info=True)
        st.error(f"Database connection failed during knowledge base rebuild. Please ensure PostgreSQL is running and your database credentials in .env are correct. Error: {e}")
        if progress_bar: progress_bar.progress(0, text="Failed to rebuild knowledge base.")
        return 0
    except Exception as e:
        logging.error(f"❌ Error during vector DB rebuild: {e}", exc_info=True)
        st.error(f"Failed to rebuild knowledge base. Please check the document content or database configuration. Error: {e}")
        if progress_bar: progress_bar.progress(0, text="Failed to rebuild knowledge base.")
        return 0

def delete_documents_from_vector_db(file_paths: List[str]):
    """Deletes documents from the vector database based on file paths."""
    if not file_paths:
        logging.info("No file paths provided for deletion from vector DB.")
        return

    st.info(f"🗑️ Deleting {len(file_paths)} document(s) from the knowledge base...")
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(PGVECTOR_CONN_STRING_PSYCOPG2)
        cur = conn.cursor()

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        relative_file_paths = [os.path.relpath(path, project_root) for path in file_paths]

        cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (COLLECTION_NAME,))
        collection_id_result = cur.fetchone()

        if not collection_id_result:
            st.warning(f"Collection '{COLLECTION_NAME}' not found in the vector DB. No documents to delete.")
            logging.info(f"Collection '{COLLECTION_NAME}' not found, skipping deletion.")
            return

        collection_id = collection_id_result[0]

        delete_query = """
            DELETE FROM langchain_pg_embedding
            WHERE collection_id = %s AND cmetadata->>'source' = ANY(%s::text[]);
        """
        cur.execute(delete_query, (collection_id, relative_file_paths))
        conn.commit()

        deleted_rows = cur.rowcount
        logging.info(f"✅ Deleted {deleted_rows} chunks from {len(file_paths)} documents from vector DB.")
        st.success(f"✅ Deleted {deleted_rows} chunks from {len(file_paths)} documents from knowledge base.")

    except OperationalError as e:
        logging.error(f"❌ PostgreSQL connection error during document deletion: {e}", exc_info=True)
        st.error(f"Database connection failed during document deletion. Please ensure PostgreSQL is running and your database credentials in .env are correct. Error: {e}")
    except Exception as e:
        logging.error(f"Error during document deletion from PGVector: {e}", exc_info=True)
        st.error(f"Failed to delete documents from Vector DB. Error: {e}")
    finally:
        if cur: cur.close()
        if conn: conn.close()


def parse_llm_response_with_snippets(response_text: str) -> Tuple[str, List[str]]:
    """
    Parses the LLM's response to separate the main answer from supporting snippets
    identified by <SNIPPET>...</SNIPPET> tags.
    It also removes these tags from the main answer text.
    """
    # Regex to find content within <SNIPPET>...</SNIPPET> tags
    snippet_pattern = re.compile(r'<SNIPPET>(.*?)</SNIPPET>', re.DOTALL)

    found_snippets = snippet_pattern.findall(response_text)

    # Remove the <SNIPPET> tags and their content from the main answer text
    main_answer = re.sub(snippet_pattern, '', response_text).strip()

    # Clean up any extra whitespace or empty lines that might result from tag removal
    main_answer = re.sub(r'\n\s*\n', '\n\n', main_answer).strip()

    return main_answer, found_snippets

def create_rag_chain(_llm: genai.GenerativeModel):
    """
    Create the RAG chain using a direct Google GenerativeModel.
    This chain will manually construct the prompt and call generate_content.
    """
    logging.info("Creating RAG chain with Gemini GenerativeModel.")

    def format_docs(docs):
        # Format documents, including their source for context
        formatted_string = ""
        for i, doc in enumerate(docs):
            # Extract just the filename from the full path in metadata
            source_filename = os.path.basename(doc.metadata.get("source", "Unknown Source"))
            formatted_string += f"--- Document Context (Source: {source_filename}) ---\n"
            formatted_string += doc.page_content
            formatted_string += "\n\n"
        return formatted_string.strip()

    def rag_callable(question_with_context: str, retriever: Any): # Renamed 'question' to 'question_with_context'
        try:
            docs = retriever.invoke(question_with_context) # Use the contextualized question for retrieval
            formatted_context = format_docs(docs)

            final_prompt = RAG_PROMPT_TEMPLATE.format(context=formatted_context, question=question_with_context)

            response = _llm.generate_content(final_prompt)
            # Parse the response to separate answer and snippets
            answer_text, snippets = parse_llm_response_with_snippets(response.text)
            return answer_text, snippets, docs # Return answer, snippets, and original docs
        except Exception as e:
            logging.error(f"Error generating content with Gemini in RAG chain: {e}", exc_info=True)
            # Provide a more user-friendly error message
            st.error(f"Failed to get an answer from the AI. This might be due to an issue with the AI model or your query. Error: {e}")
            return f"Sorry, I couldn't process your request right now. Please try again or rephrase your question. Error: {e}", [], []

    logging.info("RAG callable created successfully.")
    return rag_callable

def classify_query_department(query: str) -> Tuple[str, str] :
    """
    Classifies a user query into a department and identifies relevant keywords.
    Uses a direct Gemini API call.
    """
    if not GOOGLE_API_KEY:
        logging.warning("Google API Key not set. Cannot classify query by department using Gemini.")
        st.warning("AI features requiring Google API Key are not available. Please set GOOGLE_API_KEY in your .env file.")
        return "general", ""

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(CHAT_MODEL) # Use CHAT_MODEL from config

        classification_prompt = f"""
        Given the following user query, classify it into one of these departments: 'HR', 'IT', 'Sales', 'Finance', 'Legal', 'Product', 'Marketing', 'Data Science', 'Operations', 'Research', 'Customer Support', 'General'.
        Also, extract up to 3 most relevant keywords from the query.

        Format your response as a JSON object with 'department' and 'keywords' (a list of strings).
        Example: {{"department": "HR", "keywords": ["leave policy", "benefits"]}}
        Example: {{"department": "Finance", "keywords": ["budget", "expense report"]}}

        Query: "{query}"
        """
        response = model.generate_content(classification_prompt)

        try:
            response_json = json.loads(response.text)
            department = response_json.get("department", "General")
            keywords = response_json.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = [keywords] if isinstance(keywords, str) else []
            logging.info(f"Query classified to department: {department}, Keywords: {', '.join(keywords)}")
            return department, ", ".join(keywords)
        except json.JSONDecodeError:
            logging.warning(f"Gemini response for classification was not valid JSON: {response.text[:100]}...")
            st.warning("AI model returned an unparseable response for query classification. Using general classification.")
            # Fallback to simple keyword-based classification if JSON parsing fails
            lower_query = query.lower()
            if "hr" in lower_query or "human resources" in lower_query or "leave" in lower_query or "salary" in lower_query:
                return "HR", "HR"
            elif "it" in lower_query or "tech" in lower_query or "software" in lower_query:
                return "IT", "IT"
            elif "sales" in lower_query or "customer" in lower_query or "client" in lower_query:
                return "Sales", "Sales"
            elif "finance" in lower_query or "budget" in lower_query or "expense" in lower_query:
                return "Finance", "Finance"
            elif "legal" in lower_query or "contract" in lower_query or "compliance" in lower_query:
                return "Legal", "Legal"
            return "General", ""

    except Exception as e:
        logging.error(f"Error classifying query with Gemini: {e}", exc_info=True)
        st.error(f"Failed to classify query using AI. Error: {e}")
        return "general", ""

def get_document_content_for_summary(file_path: str) -> str:
    """
    Extracts and returns the full text content of a document for summarization.
    Supports PDF and TXT files.
    """
    documents = load_documents(file_path)
    if not documents:
        return ""

    # Concatenate page content from all documents/pages
    full_content = "\n".join([doc.page_content for doc in documents])
    return full_content

def generate_auto_summary(document_content: str, llm: genai.GenerativeModel) -> str:
    """
    Generates a concise summary (approx. 150 words) of the document content using Gemini.
    For very large documents, it implements a map-reduce style summarization.
    """
    if not document_content or not llm:
        return "Could not generate summary: missing document content or LLM."

    logging.info("Generating auto-summary for the document.")

    try:
        # Check if the document content is too large for direct summarization
        if len(document_content) > MAX_DIRECT_SUMMARY_TOKENS: # Using character count as a proxy for tokens
            logging.info("Document content is very large, performing map-reduce summarization.")
            st.info("Document is very large. Summarizing in multiple steps, this may take longer...")

            # Split the document into chunks for individual summarization
            # Use a text splitter similar to how documents are chunked for vector DB, but for raw text
            summary_text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=MAX_DIRECT_SUMMARY_TOKENS // 2, # Make chunks smaller for chunk summaries
                chunk_overlap=100,
                length_function=len,
                is_separator_regex=False,
            )
            content_chunks = summary_text_splitter.split_text(document_content)

            chunk_summaries = []
            for i, chunk in enumerate(content_chunks):
                logging.info(f"Summarizing chunk {i+1}/{len(content_chunks)}")
                chunk_prompt = CHUNK_SUMMARY_PROMPT_TEMPLATE.format(text_chunk=chunk)
                chunk_response = llm.generate_content(chunk_prompt)
                chunk_summaries.append(chunk_response.text)
                st.progress((i + 1) / len(content_chunks), text=f"Summarizing chunk {i+1} of {len(content_chunks)}...")

            # Combine chunk summaries and summarize them again
            combined_chunk_summaries = "\n\n".join(chunk_summaries)
            logging.info(f"Combined {len(chunk_summaries)} chunk summaries. Final summarization step.")

            final_summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(document_content=combined_chunk_summaries)
            response = llm.generate_content(final_summary_prompt)
            return response.text
        else:
            # Direct summarization for smaller documents
            summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(document_content=document_content)
            response = llm.generate_content(summary_prompt)
            return response.text
    except Exception as e:
        logging.error(f"❌ Error generating auto-summary with Gemini: {e}", exc_info=True)
        st.error(f"Failed to generate summary. This might be due to an issue with the AI model or the document content. Error: {e}")
        return f"Failed to generate summary: {e}"

def compare_documents_with_llm(query: str, llm: genai.GenerativeModel, retriever: Any) -> str:
    """
    Compares concepts or findings across multiple documents using the LLM.
    Retrieves relevant chunks and prompts the LLM to perform the comparison.
    """
    if not query or not llm or not retriever:
        return "Cannot perform comparison: missing query, LLM, or retriever."

    logging.info(f"Performing document comparison for query: {query}")

    try:
        # Retrieve relevant documents based on the comparison query
        # The retriever should return documents with 'source' metadata
        docs = retriever.invoke(query)

        if not docs:
            return "No relevant information found in the documents for this comparison query."

        # Group documents by source for clearer presentation to the LLM
        context_by_source = {}
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Get project root for relative path

        for doc in docs:
            # Ensure source is a relative path for display
            source_path = doc.metadata.get("source", "unknown_source.txt")
            # Extract just the filename for display
            source_filename = os.path.basename(source_path)
            if source_filename not in context_by_source:
                context_by_source[source_filename] = []
            context_by_source[source_filename].append(doc.page_content)

        formatted_context = ""
        for source, contents in context_by_source.items():
            formatted_context += f"--- Content from: {source} ---\n"
            formatted_context += "\n".join(contents)
            formatted_context += "\n\n"

        # Use the document comparison prompt
        comparison_prompt = DOCUMENT_COMPARISON_PROMPT.format(
            document_context=formatted_context.strip(),
            comparison_query=query
        )

        response = llm.generate_content(comparison_prompt)
        return response.text

    except Exception as e:
        logging.error(f"❌ Error comparing documents with Gemini: {e}", exc_info=True)
        st.error(f"Failed to compare documents. This might be due to an issue with the AI model or the documents. Error: {e}")
        return f"Failed to compare documents: {e}"
