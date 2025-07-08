# ===============================================================
# File: src/rag_pipeline.py
# ===============================================================
import logging
import os
import pandas as pd
import streamlit as st
from typing import List, Any, Tuple, Dict, Union # Import Union
from langchain_community.document_loaders import PyPDFLoader, TextLoader
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
import google.generativeai as genai # Re-import for Gemini
from langchain_community.chat_models import ChatOllama # Import ChatOllama
import json
import re # Import regex module
import uuid # For generating UUIDs for chat sessions
from datetime import datetime # For timestamps

from config import (
    DOC_FOLDER,
    GEMINI_MODEL_NAME, # New: Gemini model name
    LOCAL_LLM_MODEL_NAME, # Local LLM model name
    OLLAMA_BASE_URL,     # Ollama base URL
    EMBEDDING_MODEL_NAME,
    COLLECTION_NAME, # This will now be a base name, actual collection name will be chat_id
    PGVECTOR_CONN_STRING_SQLACHEMY,
    PGVECTOR_CONN_STRING_PSYCOPG2,
    GOOGLE_API_KEY # Re-import for Gemini
)
from prompt_templates import RAG_PROMPT_TEMPLATE, SUMMARY_PROMPT_TEMPLATE, CHUNK_SUMMARY_PROMPT_TEMPLATE, DOCUMENT_COMPARISON_PROMPT

# Define a heuristic for maximum tokens for direct summarization
# This is a rough estimate and might need fine-tuning based on actual LLM token limits and performance
MAX_DIRECT_SUMMARY_TOKENS = 8000 # Roughly equivalent to 4000 words, leaves room for prompt

# Define a type alias for the LLM, as it can be either Gemini or Ollama
LLM_TYPE = Union[genai.GenerativeModel, ChatOllama]

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

# Removed @st.cache_resource as LLM depends on user selection
def get_llm(llm_choice: str) -> LLM_TYPE | None: # Updated function signature
    """Initializes the chosen LLM (Gemini or Local Ollama)."""
    if llm_choice == "Gemini":
        if not GOOGLE_API_KEY:
            st.error("Google API Key is required for Gemini LLM. Please set GOOGLE_API_KEY in your .env file.")
            return None
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            llm = genai.GenerativeModel(GEMINI_MODEL_NAME) # Use GEMINI_MODEL_NAME from config
            logging.info(f"✅ Gemini GenerativeModel ({GEMINI_MODEL_NAME}) initialized successfully.")
            return llm
        except Exception as e:
            logging.error(f"❌ Error initializing Gemini LLM: {e}", exc_info=True)
            st.error(f"Failed to initialize Gemini LLM. Please check your Google API Key in the .env file and ensure it's valid. Error: {e}")
            return None
    elif llm_choice == "Local LLM (Ollama)":
        if not LOCAL_LLM_MODEL_NAME:
            st.error("Local LLM model name is not set. Please set LOCAL_LLM_MODEL_NAME in your .env or config.py file.")
            return None
        if not OLLAMA_BASE_URL:
            st.error("Ollama base URL is not set. Please set OLLAMA_BASE_URL in your .env or config.py file.")
            return None
        try:
            llm = ChatOllama(model=LOCAL_LLM_MODEL_NAME, base_url=OLLAMA_BASE_URL)
            logging.info(f"✅ Local LLM ({LOCAL_LLM_MODEL_NAME}) initialized successfully via Ollama at {OLLAMA_BASE_URL}.")
            return llm
        except Exception as e:
            logging.error(f"❌ Error initializing Local LLM (Ollama): {e}", exc_info=True)
            st.error(f"Failed to initialize Local LLM. Please ensure Ollama is running and the model '{LOCAL_LLM_MODEL_NAME}' is pulled. Error: {e}")
            return None
    else:
        st.error("Invalid LLM choice.")
        return None


def load_existing_vector_db(_embedding_model: Any, chat_session_id: str) -> PGVector | None:
    """Loads the existing vector database for a specific chat session."""
    # The collection_name for PGVector will be the chat_session_id
    collection_name_for_session = str(chat_session_id)
    try:
        vector_db = PGVector(
            embedding_function=_embedding_model,
            collection_name=collection_name_for_session,
            connection_string=PGVECTOR_CONN_STRING_SQLACHEMY,
        )
        logging.info(f"✅ Existing vector DB loaded successfully for session: {chat_session_id}.")
        create_hnsw_indexing() # Ensure HNSW index is created for performance
        return vector_db
    except OperationalError as e:
        logging.error(f"❌ PostgreSQL connection error when loading vector DB for session {chat_session_id}: {e}", exc_info=True)
        st.error(f"Database connection failed when loading knowledge base for this chat. Please ensure PostgreSQL is running and your database credentials in .env are correct. Error: {e}")
        return None
    except Exception as e:
        logging.error(f"❌ Error loading existing vector DB for session {chat_session_id}: {e}", exc_info=True)
        st.error(f"Failed to load vector database for this chat. Ensure PostgreSQL is running, PGVector extension is installed, and tables are initialized. Error: {e}")
        return None

def load_documents(file_path: str) -> List[Document]:
    """Load documents from a given file path."""
    loader = None
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        logging.warning(f"Unsupported file type for {file_path}. Skipping.")
        st.warning(f"Unsupported file type: {os.path.basename(file_path)}. Please upload PDF or TXT.")
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

def create_vector_db_and_rebuild(embedding_model: Any, documents_to_process_paths: List[str], chat_session_id: str, progress_bar=None, status_text=None) -> int:
    """
    Creates/rebuilds the vector database for a specific chat session by processing documents.
    It clears existing embeddings for the session's collection and then adds new ones from
    ALL provided document paths.
    Returns the number of chunks processed.
    """
    if status_text: status_text.info("🔄 Rebuilding knowledge base. This may take a moment...")
    if progress_bar: progress_bar.progress(0, text="Starting knowledge base rebuild...")

    # The collection_name for PGVector will be the chat_session_id
    collection_name_for_session = str(chat_session_id)

    try:
        if not documents_to_process_paths:
            st.warning("No documents found to process for the knowledge base.")
            if progress_bar: progress_bar.progress(100, text="No documents to process.")
            return 0

        documents = []
        for i, file_path in enumerate(documents_to_process_paths):
            if status_text: status_text.info(f"Loading document: {os.path.basename(file_path)}...")
            if progress_bar: progress_bar.progress(int((i / len(documents_to_process_paths)) * 20), text=f"Loading document: {os.path.basename(file_path)}...")
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

        # Clear existing embeddings for this specific chat session's collection
        # This ensures that when new documents are added, or existing ones are re-processed,
        # the vector store is consistent with the current set of documents for the session.
        conn = None
        cur = None
        try:
            if status_text: status_text.info(f"Clearing existing data for session '{chat_session_id}'...")
            if progress_bar: progress_bar.progress(40, text=f"Clearing existing data for session '{chat_session_id}'...")
            conn = psycopg2.connect(PGVECTOR_CONN_STRING_PSYCOPG2)
            conn.autocommit = True
            cur = conn.cursor()

            # First, ensure the collection exists in langchain_pg_collection and get its UUID
            cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (collection_name_for_session,))
            collection_uuid_result = cur.fetchone()

            if collection_uuid_result:
                collection_uuid = collection_uuid_result[0]
                logging.info(f"Deleting existing embeddings for collection '{collection_name_for_session}' ({collection_uuid})")
                cur.execute("DELETE FROM langchain_pg_embedding WHERE collection_id = %s;", (collection_uuid,))
                conn.commit()
                logging.info("Existing embeddings cleared for session collection.")
            else:
                logging.info(f"Collection '{collection_name_for_session}' does not exist, no embeddings to clear. Will be created by PGVector.")
        except OperationalError as e:
            logging.error(f"❌ PostgreSQL connection error during clearing for session {chat_session_id}: {e}", exc_info=True)
            st.error(f"Database connection failed during data clearing for this chat. Please ensure PostgreSQL is running and your database credentials in .env are correct. Error: {e}")
            return 0
        except psycopg2_errors.UndefinedTable as e:
            logging.warning(f"Collection table not found during clearing (might be first run): {e}")
            st.warning("Database tables not found. Attempting to create them (this is normal on first run).")
        except Exception as e:
            logging.warning(f"Could not clear existing embeddings for session '{chat_session_id}': {e}", exc_info=True)
            st.warning(f"Failed to clear existing embeddings for this chat. Proceeding with adding new documents. Error: {e}")
        finally:
            if cur: cur.close()
            if conn: conn.close()

        logging.info(f"Creating/updating vector database with new chunks for session {chat_session_id}.")
        if status_text: status_text.info("Embedding chunks and adding to database...")
        if progress_bar: progress_bar.progress(60, text="Embedding chunks and adding to database...")

        # Initialize PGVector with the specific collection name (chat_session_id)
        vector_db = PGVector(
            embedding_function=embedding_model,
            collection_name=collection_name_for_session,
            connection_string=PGVECTOR_CONN_STRING_SQLACHEMY,
        )

        vector_db.add_documents(chunks)

        logging.info(f"✅ Vector database rebuilt with {len(chunks)} chunks for session {chat_session_id}.")
        if status_text: st.success(f"✅ Knowledge base rebuilt with {len(chunks)} chunks from {len(documents_to_process_paths)} files for this chat.")
        if progress_bar: progress_bar.progress(100, text="Knowledge base rebuilt successfully!")
        return len(chunks) # Return number of chunks
    except OperationalError as e:
        logging.error(f"❌ PostgreSQL connection error during vector DB rebuild for session {chat_session_id}: {e}", exc_info=True)
        st.error(f"Database connection failed during knowledge base rebuild for this chat. Please ensure PostgreSQL is running and your database credentials in .env are correct. Error: {e}")
        if progress_bar: progress_bar.progress(0, text="Failed to rebuild knowledge base.")
        return 0
    except Exception as e:
        logging.error(f"❌ Error during vector DB rebuild for session {chat_session_id}: {e}", exc_info=True)
        st.error(f"Failed to rebuild knowledge base for this chat. Please check the document content or database configuration. Error: {e}")
        if progress_bar: progress_bar.progress(0, text="Failed to rebuild knowledge base.")
        return 0

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

def create_rag_chain(_llm: LLM_TYPE): # Type hint changed to LLM_TYPE
    """
    Create the RAG chain using the provided LLM instance (Gemini or ChatOllama).
    This chain will manually construct the prompt and call the appropriate invocation method.
    """
    logging.info(f"Creating RAG chain with LLM type: {type(_llm).__name__}.")

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

    def rag_callable(question_with_context: str, retriever: Any):
        try:
            docs = retriever.invoke(question_with_context)
            formatted_context = format_docs(docs)

            final_prompt = RAG_PROMPT_TEMPLATE.format(context=formatted_context, question=question_with_context)

            response_text = ""
            if isinstance(_llm, genai.GenerativeModel):
                response = _llm.generate_content(final_prompt)
                response_text = response.text
            elif isinstance(_llm, ChatOllama):
                response = _llm.invoke(final_prompt)
                response_text = response.content
            else:
                raise ValueError("Unsupported LLM type provided to RAG chain.")

            # Parse the response to separate answer and snippets
            answer_text, snippets = parse_llm_response_with_snippets(response_text)
            return answer_text, snippets, docs # Return answer, snippets, and original docs
        except Exception as e:
            logging.error(f"Error generating content with LLM in RAG chain: {e}", exc_info=True)
            # Provide a more user-friendly error message
            st.error(f"Failed to get an answer from the AI. This might be due to an issue with the AI model or your query. Error: {e}")
            return f"Sorry, I couldn't process your request right now. Please try again or rephrase your question. Error: {e}", [], []

    logging.info("RAG callable created successfully.")
    return rag_callable

def get_all_document_chunks_text_for_session(chat_session_id: str) -> str:
    """
    Retrieves and concatenates all document text chunks stored in the database
    for a given chat session. This replaces reading from local files for summarization,
    challenge generation, and comparison.
    """
    conn = None
    cur = None
    full_content = []
    try:
        conn = psycopg2.connect(PGVECTOR_CONN_STRING_PSYCOPG2)
        cur = conn.cursor()

        # Get the collection_id (UUID) for this chat session
        cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (str(chat_session_id),))
        collection_uuid_result = cur.fetchone()

        if collection_uuid_result:
            collection_uuid = collection_uuid_result[0]
            logging.info(f"Retrieving all chunks for collection_id: {collection_uuid}")
            # Retrieve all document content (chunks) for this collection
            cur.execute(
                "SELECT document FROM langchain_pg_embedding WHERE collection_id = %s ORDER BY id;",
                (collection_uuid,)
            )
            for row in cur.fetchall():
                full_content.append(row[0]) # row[0] is the 'document' TEXT column
            logging.info(f"Retrieved {len(full_content)} chunks from DB for session {chat_session_id}.")
        else:
            logging.warning(f"No PGVector collection found for session {chat_session_id}. Cannot retrieve document content from DB.")
            st.warning("No document content found in the database for this chat session.")
        
        return "\n\n".join(full_content) # Join all chunks into a single string
    except OperationalError as e:
        logging.error(f"❌ PostgreSQL connection error when retrieving document content for session {chat_session_id}: {e}", exc_info=True)
        st.error(f"Database connection failed when retrieving document content. Please ensure PostgreSQL is running and your database credentials in .env are correct. Error: {e}")
        return ""
    except Exception as e:
        logging.error(f"❌ Error retrieving document content from DB for session {chat_session_id}: {e}", exc_info=True)
        st.error(f"Failed to retrieve document content from database. Error: {e}")
        return ""
    finally:
        if cur: cur.close()
        if conn: conn.close()

def get_document_content_for_summary(chat_session_id: str) -> str:
    """
    Extracts and returns the full text content of documents for summarization
    by retrieving it directly from the database.
    """
    return get_all_document_chunks_text_for_session(chat_session_id)

def generate_auto_summary(document_content: str, llm: LLM_TYPE) -> str: # Type hint changed to LLM_TYPE
    """
    Generates a concise summary (approx. 150 words) of the document content using the LLM.
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
                
                chunk_response_text = ""
                if isinstance(llm, genai.GenerativeModel):
                    chunk_response = llm.generate_content(chunk_prompt)
                    chunk_response_text = chunk_response.text
                elif isinstance(llm, ChatOllama):
                    chunk_response = llm.invoke(chunk_prompt)
                    chunk_response_text = chunk_response.content
                else:
                    raise ValueError("Unsupported LLM type for chunk summarization.")
                
                chunk_summaries.append(chunk_response_text)
                st.progress((i + 1) / len(content_chunks), text=f"Summarizing chunk {i+1} of {len(content_chunks)}...")

            # Combine chunk summaries and summarize them again
            combined_chunk_summaries = "\n\n".join(chunk_summaries)
            logging.info(f"Combined {len(chunk_summaries)} chunk summaries. Final summarization step.")

            final_summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(document_content=combined_chunk_summaries)
            
            final_response_text = ""
            if isinstance(llm, genai.GenerativeModel):
                response = llm.generate_content(final_summary_prompt)
                final_response_text = response.text
            elif isinstance(llm, ChatOllama):
                response = llm.invoke(final_summary_prompt)
                final_response_text = response.content
            else:
                raise ValueError("Unsupported LLM type for final summarization.")
            
            return final_response_text
        else:
            # Direct summarization for smaller documents
            summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(document_content=document_content)
            
            response_text = ""
            if isinstance(llm, genai.GenerativeModel):
                response = llm.generate_content(summary_prompt)
                response_text = response.text
            elif isinstance(llm, ChatOllama):
                response = llm.invoke(summary_prompt)
                response_text = response.content
            else:
                raise ValueError("Unsupported LLM type for direct summarization.")
            
            return response_text
    except Exception as e:
        logging.error(f"❌ Error generating auto-summary with LLM: {e}", exc_info=True)
        st.error(f"Failed to generate summary. This might be due to an issue with the AI model or the document content. Error: {e}")
        return f"Failed to generate summary: {e}"

def compare_documents_with_llm(query: str, llm: LLM_TYPE, retriever: Any) -> str: # Type hint changed to LLM_TYPE
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
        # project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # No longer needed for file access

        for doc in docs:
            # The 'source' metadata should already contain the relative path/filename
            source_filename = os.path.basename(doc.metadata.get("source", "Unknown Source"))
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

        response_text = ""
        if isinstance(llm, genai.GenerativeModel):
            response = llm.generate_content(comparison_prompt)
            response_text = response.text
        elif isinstance(llm, ChatOllama):
            response = llm.invoke(comparison_prompt)
            response_text = response.content
        else:
            raise ValueError("Unsupported LLM type for document comparison.")
        
        return response_text

    except Exception as e:
        logging.error(f"❌ Error comparing documents with LLM: {e}", exc_info=True)
        st.error(f"Failed to compare documents. This might be due to an issue with the AI model or the documents. Error: {e}")
        return f"Failed to compare documents: {e}"

# --- New functions for Chat Session Management ---

def create_chat_session_db(chat_name: str, documents_metadata: List[Dict[str, str]] = None) -> str:
    """Inserts a new chat session into the database and returns its UUID."""
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(PGVECTOR_CONN_STRING_PSYCOPG2)
        cur = conn.cursor()
        new_uuid = uuid.uuid4()
        cur.execute(
            """
            INSERT INTO chat_sessions (uuid, name, documents_metadata)
            VALUES (%s, %s, %s) RETURNING uuid;
            """,
            (str(new_uuid), chat_name, json.dumps(documents_metadata) if documents_metadata else json.dumps([])) # Store as JSONB
        )
        session_uuid = cur.fetchone()[0]
        conn.commit()
        logging.info(f"Created new chat session: {chat_name} with ID: {session_uuid}")
        return str(session_uuid)
    except Exception as e:
        logging.error(f"❌ Error creating chat session in DB: {e}", exc_info=True)
        st.error(f"Failed to create new chat session. Error: {e}")
        return None
    finally:
        if cur: cur.close()
        if conn: conn.close()

def load_chat_sessions_db() -> List[Dict[str, Any]]:
    """Retrieves all chat sessions from the database."""
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(PGVECTOR_CONN_STRING_PSYCOPG2)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT uuid, name, documents_metadata, created_at
            FROM chat_sessions
            ORDER BY created_at DESC;
            """
        )
        sessions = []
        for row in cur.fetchall():
            documents_metadata = row[2]
            if isinstance(documents_metadata, str): # If JSONB stored as string, parse it
                try:
                    documents_metadata = json.loads(documents_metadata)
                except json.JSONDecodeError:
                    documents_metadata = [] # Fallback if parsing fails
            elif documents_metadata is None:
                documents_metadata = []

            sessions.append({
                "id": str(row[0]),
                "name": row[1],
                "documents_metadata": documents_metadata, # This is now a list of dicts
                "created_at": row[3].strftime("%Y-%m-%d %H:%M:%S") if row[3] else "N/A"
            })
        logging.info(f"Loaded {len(sessions)} chat sessions from DB.")
        return sessions
    except Exception as e:
        logging.error(f"❌ Error loading chat sessions from DB: {e}", exc_info=True)
        st.error(f"Failed to load past chat sessions. Error: {e}")
        return []
    finally:
        if cur: cur.close()
        if conn: conn.close()

def delete_chat_session_db(chat_session_id: str):
    """Deletes a chat session and all its associated data (history, embeddings) from the database."""
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(PGVECTOR_CONN_STRING_PSYCOPG2)
        conn.autocommit = True # For DDL operations like dropping collection
        cur = conn.cursor()

        # Delete from chat_sessions (this will cascade delete from chat_history due to ON DELETE CASCADE)
        cur.execute("DELETE FROM chat_sessions WHERE uuid = %s;", (chat_session_id,))
        logging.info(f"Deleted chat session {chat_session_id} and its history.")

        # Also delete the associated collection from langchain_pg_collection and its embeddings
        # The collection name is the chat_session_id itself
        cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (chat_session_id,))
        collection_uuid_result = cur.fetchone()

        if collection_uuid_result:
            collection_uuid = collection_uuid_result[0]
            cur.execute("DELETE FROM langchain_pg_embedding WHERE collection_id = %s;", (collection_uuid,))
            cur.execute("DELETE FROM langchain_pg_collection WHERE uuid = %s;", (collection_uuid,))
            logging.info(f"Deleted PGVector collection {collection_uuid} and its embeddings for session {chat_session_id}.")
        else:
            logging.info(f"No PGVector collection found for session {chat_session_id}, skipping embedding deletion.")

        st.success(f"Chat session '{chat_session_id}' and its data deleted successfully.")
    except Exception as e:
        logging.error(f"❌ Error deleting chat session {chat_session_id} from DB: {e}", exc_info=True)
        st.error(f"Failed to delete chat session. Error: {e}")
    finally:
        if cur: cur.close()
        if conn: conn.close()

def save_chat_message_db(chat_session_id: str, role: str, content: str, snippets: List[str] = None):
    """Saves a chat message (user question or AI answer) to the database."""
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(PGVECTOR_CONN_STRING_PSYCOPG2)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO chat_history (chat_session_id, role, content, snippets)
            VALUES (%s, %s, %s, %s);
            """,
            (chat_session_id, role, content, json.dumps(snippets) if snippets else None)
        )
        conn.commit()
        logging.debug(f"Saved message for session {chat_session_id}, role: {role}")
    except Exception as e:
        logging.error(f"❌ Error saving chat message for session {chat_session_id}: {e}", exc_info=True)
        st.error(f"Failed to save chat message. Error: {e}")
    finally:
        if cur: cur.close()
        if conn: conn.close()

def load_chat_history_db(chat_session_id: str) -> List[Dict[str, Any]]:
    """Loads chat history for a given chat session from the database."""
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(PGVECTOR_CONN_STRING_PSYCOPG2)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT role, content, snippets
            FROM chat_history
            WHERE chat_session_id = %s
            ORDER BY timestamp;
            """,
            (chat_session_id,)
        )
        history = []
        for row in cur.fetchall():
            snippets_data = row[2]
            # Ensure snippets_data is a list, even if it was stored as null/None
            if snippets_data is None:
                snippets_list = []
            elif isinstance(snippets_data, str): # If JSONB stored as string, parse it
                try:
                    snippets_list = json.loads(snippets_data)
                except json.JSONDecodeError:
                    snippets_list = [] # Fallback if parsing fails
            else: # Assume it's already a list or other direct JSONB type
                snippets_list = snippets_data

            history.append({
                "role": row[0],
                "content": row[1],
                "snippets": snippets_list
            })
        logging.info(f"Loaded {len(history)} messages for session {chat_session_id}.")
        return history
    except Exception as e:
        logging.error(f"❌ Error loading chat history for session {chat_session_id}: {e}", exc_info=True)
        st.error(f"Failed to load chat history. Error: {e}")
        return []
    finally:
        if cur: cur.close()
        if conn: conn.close()

def rename_chat_session_db(chat_session_id: str, new_name: str):
    """Renames a chat session in the database."""
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(PGVECTOR_CONN_STRING_PSYCOPG2)
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE chat_sessions
            SET name = %s
            WHERE uuid = %s;
            """,
            (new_name, chat_session_id)
        )
        conn.commit()
        logging.info(f"Renamed chat session {chat_session_id} to '{new_name}'.")
        st.success(f"Chat renamed to '{new_name}'!")
    except Exception as e:
        logging.error(f"❌ Error renaming chat session {chat_session_id}: {e}", exc_info=True)
        st.error(f"Failed to rename chat. Error: {e}")
    finally:
        if cur: cur.close()
        if conn: conn.close()

def update_chat_session_documents_metadata_db(chat_session_id: str, documents_metadata: List[Dict[str, str]]):
    """
    Updates the list of document metadata (name and path) for an existing chat session.
    This function expects the complete list of documents for the session.
    """
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(PGVECTOR_CONN_STRING_PSYCOPG2)
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE chat_sessions
            SET documents_metadata = %s
            WHERE uuid = %s;
            """,
            (json.dumps(documents_metadata), chat_session_id)
        )
        conn.commit()
        logging.info(f"Updated documents metadata for session {chat_session_id}.")
    except Exception as e:
        logging.error(f"❌ Error updating documents metadata for session {chat_session_id}: {e}", exc_info=True)
        st.error(f"Failed to update document metadata for this chat. Error: {e}")
    finally:
        if cur: cur.close()
        if conn: conn.close()

def get_num_chunks_for_session(chat_session_id: str) -> int:
    """Retrieves the number of chunks stored for a given chat session."""
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(PGVECTOR_CONN_STRING_PSYCOPG2)
        cur = conn.cursor()
        cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s;", (str(chat_session_id),))
        collection_uuid_result = cur.fetchone()
        if collection_uuid_result:
            collection_uuid = collection_uuid_result[0]
            cur.execute("SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = %s;", (collection_uuid,))
            # Fetchone returns a tuple, get the first element for the count
            count = cur.fetchone()[0] 
            return int(count) # Ensure count is an integer
        return 0
    except Exception as e:
        logging.error(f"❌ Error getting chunk count for session {chat_session_id}: {e}", exc_info=True)
        return 0
    finally:
        if cur: cur.close()
        if conn: conn.close()

