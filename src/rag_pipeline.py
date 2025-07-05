# ===============================================================
# File: src/rag_pipeline.py
# ===============================================================
import logging
import os
import pandas as pd
import streamlit as st
from typing import List, Any, Tuple
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
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
from prompt_templates import RAG_PROMPT_TEMPLATE

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
        st.error(f"Failed to load embedding model: {e}. Please check your internet connection or model name.")
        return None

@st.cache_resource
def get_gemini_rag_llm():
    """Initializes Gemini LLM for RAG."""
    if not GOOGLE_API_KEY:
        st.error("Google API Key is required for Gemini LLM in RAG. Please set GOOGLE_API_KEY in your .env file.")
        return None
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        llm = genai.GenerativeModel("gemini-1.5-flash-latest")
        logging.info("✅ Gemini GenerativeModel initialized successfully for RAG.")
        return llm
    except Exception as e:
        logging.error(f"❌ Error initializing Gemini LLM: {e}", exc_info=True)
        st.error(f"Failed to initialize Gemini LLM: {e}. Please check your Google API Key.")
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
    except Exception as e:
        logging.error(f"❌ Error loading existing vector DB: {e}", exc_info=True)
        st.error(f"Failed to load vector database: {e}. Please ensure PostgreSQL is running and accessible.")
        return None

def load_documents(file_path: str) -> List[Document]:
    """Load documents from a given file path."""
    loader = None
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".csv"): # CSVLoader is from langchain_community.document_loaders
        loader = CSVLoader(file_path)
    else:
        logging.warning(f"Unsupported file type for {file_path}. Skipping.")
        return []

    if loader:
        try:
            documents = loader.load()
            logging.info(f"Loaded {len(documents)} documents from {os.path.basename(file_path)}")
            return documents
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}", exc_info=True)
            st.error(f"Error loading {os.path.basename(file_path)}: {e}")
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

def create_vector_db_and_rebuild(embedding_model: Any, documents_to_process: List[str] = None):
    """
    Creates/rebuilds the vector database by processing documents.
    If documents_to_process is None, processes all documents in DOC_FOLDER.
    Otherwise, processes only the specified file paths.
    """
    st.info("🔄 Rebuilding knowledge base. This may take a moment...")
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
            return

        documents = []
        for file_path in all_file_paths:
            docs = load_documents(file_path)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for doc in docs:
                relative_path = os.path.relpath(file_path, project_root)
                doc.metadata["source"] = relative_path
            documents.extend(docs)

        if not documents:
            st.warning("No valid documents loaded for processing. Knowledge base not rebuilt.")
            return

        chunks = split_documents(documents)

        # Clear existing collection before adding new documents to ensure a "rebuild"
        conn = None
        cur = None
        try:
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
        except Exception as e:
            logging.warning(f"Could not clear existing documents from collection '{COLLECTION_NAME}': {e}")
        finally:
            if cur: cur.close()
            if conn: conn.close()

        logging.info("Creating/updating vector database with new chunks.")
        
        vector_db = PGVector(
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
            connection_string=PGVECTOR_CONN_STRING_SQLACHEMY,
            # pre_delete_collection=True # This option is for dropping the entire table, not just clearing a collection
        )
        
        vector_db.add_documents(chunks)
        
        logging.info(f"✅ Vector database rebuilt with {len(chunks)} chunks.")
        st.success(f"✅ Knowledge base rebuilt with {len(chunks)} chunks from {len(all_file_paths)} files.")
    except Exception as e:
        logging.error(f"❌ Error during vector DB rebuild: {e}", exc_info=True)
        st.error(f"Failed to rebuild knowledge base: {e}")

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

    except Exception as e:
        logging.error(f"Error during document deletion from PGVector: {e}", exc_info=True)
        st.error(f"Failed to delete documents from Vector DB: {e}")
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
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_callable(question_with_context: str, retriever: Any): # Renamed 'question' to 'question_with_context'
        docs = retriever.invoke(question_with_context) # Use the contextualized question for retrieval
        formatted_context = format_docs(docs)
        
        final_prompt = RAG_PROMPT_TEMPLATE.format(context=formatted_context, question=question_with_context)
        
        try:
            response = _llm.generate_content(final_prompt)
            # Parse the response to separate answer and snippets
            answer_text, snippets = parse_llm_response_with_snippets(response.text)
            return answer_text, snippets, docs # Return answer, snippets, and original docs
        except Exception as e:
            logging.error(f"Error generating content with Gemini: {e}", exc_info=True)
            return f"Error: Could not generate response. {e}", [], []

    logging.info("RAG callable created successfully.")
    return rag_callable

def classify_query_department(query: str) -> Tuple[str, str] :
    """
    Classifies a user query into a department and identifies relevant keywords.
    Uses a direct Gemini API call.
    """
    if not GOOGLE_API_KEY:
        logging.warning("Google API Key not set. Cannot classify query by department using Gemini.")
        return "general", ""

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

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
        return "general", ""

def get_document_content_for_summary(file_path: str) -> str:
    """
    Extracts and returns the full text content of a document for summarization.
    Supports PDF, TXT, and CSV files.
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
    """
    if not document_content or not llm:
        return "Could not generate summary: missing document content or LLM."

    logging.info("Generating auto-summary for the document.")
    summary_prompt = f"""
    Please summarize the following document content in approximately 150 words.
    Focus on the main topics, key arguments, and conclusions.

    Document Content:
    ---
    {document_content}
    ---

    Concise Summary:
    """
    try:
        response = llm.generate_content(summary_prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error generating auto-summary with Gemini: {e}", exc_info=True)
        return f"Failed to generate summary: {e}"

# The following two functions are duplicated from challenge_mode.py.
# While they are present here, the main.py now imports them directly from challenge_mode.py
# for better modularity. They are kept here for completeness of the original file.
# In a real refactor, these would be removed from rag_pipeline.py.
def generate_challenge_questions(document_content: str, llm: genai.GenerativeModel, num_questions: int = 3) -> List[str]:
    """
    Generates logic-based challenge questions from the document content using Gemini.
    """
    if not document_content or not llm:
        return []

    logging.info(f"Generating {num_questions} challenge questions.")
    question_prompt = f"""
    Based on the following document content, generate {num_questions} challenging, logic-based questions.
    These questions should require understanding and inference, not just direct recall.
    Ensure the questions are answerable from the provided text.
    Format each question with a number, e.g., "1. Question one?".

    Document Content:
    ---
    {document_content}
    ---

    Challenge Questions:
    """
    try:
        response = llm.generate_content(question_prompt)
        # Parse the response into a list of questions
        questions_raw = response.text.strip().split('\n')
        questions = [q.strip() for q in questions_raw if q.strip() and q[0].isdigit()]
        return questions[:num_questions] # Return only the requested number of questions
    except Exception as e:
        logging.error(f"Error generating challenge questions with Gemini: {e}", exc_info=True)
        return [f"Failed to generate questions: {e}"]

def evaluate_user_answer(question: str, user_answer: str, document_content: str, llm: genai.GenerativeModel) -> str:
    """
    Evaluates a user's answer against the document content using Gemini.
    Provides detailed feedback.
    """
    if not question or not user_answer or not document_content or not llm:
        return "Cannot evaluate: missing question, answer, document content, or LLM."

    logging.info(f"Evaluating user answer for question: {question[:50]}...")
    evaluation_prompt = f"""
    You are an evaluator. Your task is to compare a user's answer to a question based on a given document.
    Provide constructive feedback.

    Question: "{question}"
    User's Answer: "{user_answer}"

    Document Content (for reference):
    ---
    {document_content}
    ---

    Please provide your evaluation in the following format:
    1. **Accuracy**: Is the user's answer factually correct based on the document? (Yes/No/Partially)
    2. **Completeness**: Does the user's answer address all aspects of the question that can be derived from the document? (Yes/No/Partially)
    3. **Clarity**: Is the user's answer clear and easy to understand? (Yes/No)
    4. **Justification/Feedback**: Explain why the answer is accurate/inaccurate or complete/incomplete, referencing the document where appropriate. Suggest improvements if necessary.
    5. **Score**: Assign a score from 0 to 10 for the user's answer, where 10 is excellent and 0 is completely incorrect.
    """
    try:
        response = llm.generate_content(evaluation_prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error evaluating user answer with Gemini: {e}", exc_info=True)
        return f"Failed to evaluate answer: {e}"
