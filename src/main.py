# ===============================================================
# File: src/main.py
# Description: Main Streamlit application for the Smart Assistant.
# ===============================================================

import streamlit as st
import os
import logging
from typing import List, Tuple, Dict
import shutil # For clearing the data folder
import uuid # For generating UUIDs for chat sessions
from datetime import datetime # For timestamps
import google.generativeai as genai # Import for type checking
from langchain_community.chat_models import ChatOllama # Import for type checking

from config import DOC_FOLDER, GOOGLE_API_KEY 
from rag_pipeline import (
    get_embedding_model,
    get_llm, 
    load_existing_vector_db,
    create_vector_db_and_rebuild,
    create_rag_chain,
    generate_auto_summary,
    get_document_content_for_summary, 
    compare_documents_with_llm,
    get_all_document_chunks_text_for_session,
    create_chat_session_db,
    load_chat_sessions_db,
    delete_chat_session_db,
    save_chat_message_db,
    load_chat_history_db,
    rename_chat_session_db,
    update_chat_session_documents_metadata_db,
    get_num_chunks_for_session, 
    LLM_TYPE 
)
from challenge_mode import generate_challenge_questions, evaluate_user_answer
from helpers import create_base64_download_button

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="QueryDoc AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
# Initialize embedding model first, it's cached and needed globally
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = get_embedding_model()

# LLM choice and initialization
if "llm_choice" not in st.session_state:
    st.session_state.llm_choice = "Gemini" # Default choice

if "llm" not in st.session_state:
    st.session_state.llm = get_llm(st.session_state.llm_choice)


# Chat Session Management State
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None # UUID of the currently active chat session
if "chat_sessions_list" not in st.session_state:
    st.session_state.chat_sessions_list = [] # List of dicts: {'id', 'name', 'documents_metadata', 'created_at'}
if "active_chat_name" not in st.session_state:
    st.session_state.active_chat_name = "New Chat" # Display name for the current chat
if "document_uploaded_for_current_session" not in st.session_state:
    st.session_state.document_uploaded_for_current_session = False # Tracks if a document is uploaded for the *current* active session
# Changed to store a list of document metadata (name and path)
if "current_documents_metadata" not in st.session_state:
    st.session_state.current_documents_metadata = [] # List of {'name': str, 'path': str}
# New: Store full document text content retrieved from DB for summarization/challenge
if "current_session_full_document_text" not in st.session_state:
    st.session_state.current_session_full_document_text = ""


# Other session states (reset per session, or managed per session)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # Chat history for the *current* active session
if "auto_summary" not in st.session_state:
    st.session_state.auto_summary = None
if "summary_generated" not in st.session_state:
    st.session_state.summary_generated = False
if "interaction_mode" not in st.session_state:
    st.session_state.interaction_mode = "Ask Anything" # Default mode
if "challenge_questions" not in st.session_state:
    st.session_state.challenge_questions = []
if "user_answers" not in st.session_state:
    st.session_state.user_answers = ["", "", ""] # For 3 questions
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = [{}, {}, {}]
if "current_document_chunks" not in st.session_state:
    st.session_state.current_document_chunks = 0
if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None

# RAG chain and vectorstore are initialized conditionally later based on active session
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# --- Helper Functions ---
def clear_doc_folder_only_temp_files():
    """
    Clears only temporary files from the DOC_FOLDER.
    This function is now primarily for cleanup of newly uploaded files after processing,
    or for ensuring a clean slate for a brand new chat.
    """
    if os.path.exists(DOC_FOLDER):
        for filename in os.listdir(DOC_FOLDER):
            file_path = os.path.join(DOC_FOLDER, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                logging.info(f"Cleaned up: {file_path}")
            except Exception as e:
                logging.error(f"Failed to delete {file_path}. Reason: {e}")
    os.makedirs(DOC_FOLDER, exist_ok=True) # Recreate the folder if it was deleted

def reset_current_chat_session():
    """Resets the state for a new chat session."""
    # Create the new chat session entry in DB immediately
    new_chat_name = f"New Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    # Pass empty list for documents_metadata for a new chat
    st.session_state.current_chat_id = create_chat_session_db(new_chat_name, documents_metadata=[])
    if not st.session_state.current_chat_id: # If DB creation failed
        st.error("Failed to create a new chat session in the database. Please check your database connection.")
        return # Prevent further execution if session creation failed

    st.session_state.active_chat_name = new_chat_name
    st.session_state.document_uploaded_for_current_session = False
    st.session_state.current_documents_metadata = [] # Reset to empty list
    st.session_state.current_session_full_document_text = "" # Reset full document text
    st.session_state.chat_history = []
    st.session_state.auto_summary = None
    st.session_state.summary_generated = False
    st.session_state.interaction_mode = "Ask Anything"
    st.session_state.challenge_questions = []
    st.session_state.user_answers = ["", "", ""]
    st.session_state.evaluation_results = [{}, {}, {}]
    st.session_state.current_document_chunks = 0
    st.session_state.comparison_result = None
    clear_doc_folder_only_temp_files() # Clear local data folder for a truly new session
    st.session_state.rag_chain = None # Reset RAG chain
    st.session_state.vectorstore = None # Reset vectorstore

    st.session_state.chat_sessions_list = load_chat_sessions_db() # Reload sessions list
    st.rerun() # Rerun to update UI with new chat

def load_chat_session(chat_id: str):
    """Loads an existing chat session from the database."""
    session_data = next((s for s in st.session_state.chat_sessions_list if s['id'] == chat_id), None)
    if not session_data:
        st.error("Selected chat session not found.")
        return

    # No need to clear DOC_FOLDER or copy files back here, as all content will come from DB

    st.session_state.current_chat_id = chat_id
    st.session_state.active_chat_name = session_data['name']
    # Load the list of document metadata (for display/download links)
    st.session_state.current_documents_metadata = session_data.get('documents_metadata', [])

    # Load chat history from DB
    st.session_state.chat_history = load_chat_history_db(chat_id)

    # Reset RAG chain and vectorstore initially
    st.session_state.rag_chain = None
    st.session_state.vectorstore = None
    st.session_state.document_uploaded_for_current_session = False # Assume false until embeddings are loaded

    # --- Crucial: Load vectorstore and rebuild RAG chain for the selected chat_id ---
    if st.session_state.embedding_model and st.session_state.llm:
        with st.spinner(f"Loading knowledge base for '{st.session_state.active_chat_name}'..."):
            # Attempt to load the existing vector DB for this chat session
            vectorstore_loaded = load_existing_vector_db(st.session_state.embedding_model, st.session_state.current_chat_id)
            
            if vectorstore_loaded:
                st.session_state.vectorstore = vectorstore_loaded
                st.session_state.rag_chain = create_rag_chain(st.session_state.llm)
                st.session_state.document_uploaded_for_current_session = True # RAG system is ready
                st.success(f"Loaded chat '{st.session_state.active_chat_name}' with its document embeddings and history.")
                # Update chunk count by querying the DB
                st.session_state.current_document_chunks = get_num_chunks_for_session(st.session_state.current_chat_id)
                # Also load the full document text from DB for summarization/challenge
                st.session_state.current_session_full_document_text = get_all_document_chunks_text_for_session(st.session_state.current_chat_id)
            else:
                st.error("Failed to load knowledge base (embeddings) for this chat session. Q&A will not work. Please check DB connection and if embeddings exist.")
                st.session_state.document_uploaded_for_current_session = False # RAG not ready
                st.session_state.current_document_chunks = 0
                st.session_state.current_session_full_document_text = "" # Ensure it's empty if DB load fails
    else:
        st.error("Embedding model or LLM not initialized. Cannot load knowledge base.")
        st.session_state.document_uploaded_for_current_session = False
        st.session_state.current_document_chunks = 0
        st.session_state.current_session_full_document_text = ""


    # Reset other session-specific states
    st.session_state.auto_summary = None
    st.session_state.summary_generated = False
    st.session_state.interaction_mode = "Ask Anything"
    st.session_state.challenge_questions = []
    st.session_state.user_answers = ["", "", ""]
    st.session_state.evaluation_results = [{}, {}, {}]
    st.session_state.comparison_result = None

    st.rerun() # Rerun to update UI

def delete_selected_chat_session(chat_id: str):
    """Deletes a chat session and its data."""
    delete_chat_session_db(chat_id)
    st.session_state.chat_sessions_list = load_chat_sessions_db() # Reload sessions list
    if st.session_state.current_chat_id == chat_id:
        reset_current_chat_session() # If current chat is deleted, start a new one
    st.rerun() # Rerun to update UI

def process_uploaded_documents_for_session(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]):
    """Handles saving the uploaded files, processing them, and rebuilding the DB for the current session."""
    if not st.session_state.current_chat_id:
        st.error("No active chat session. Please start a 'New Chat' first.")
        return

    current_session_documents_metadata = st.session_state.current_documents_metadata # Get current list
    files_to_process_for_embedding = [] # Paths of files that will be passed to create_vector_db_and_rebuild

    for uploaded_file in uploaded_files:
        file_path = os.path.join(DOC_FOLDER, uploaded_file.name)
        
        # Save the uploaded file to DOC_FOLDER (temporary staging)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        doc_info = {'name': uploaded_file.name, 'path': file_path}

        # Update metadata for the current session
        existing_doc_index = -1
        for i, doc_meta in enumerate(current_session_documents_metadata):
            if doc_meta.get('name') == uploaded_file.name:
                existing_doc_index = i
                break

        if existing_doc_index != -1:
            current_session_documents_metadata[existing_doc_index] = doc_info
            logging.info(f"Updated existing document '{uploaded_file.name}' path in metadata.")
        else:
            current_session_documents_metadata.append(doc_info)
            logging.info(f"Added new document '{uploaded_file.name}' to metadata.")
        
        files_to_process_for_embedding.append(file_path) # Add to list for embedding

    # Update session state with the modified list of documents
    st.session_state.current_documents_metadata = current_session_documents_metadata

    # Update chat session metadata in DB with the COMPLETE list of documents
    update_chat_session_documents_metadata_db(
        st.session_state.current_chat_id,
        st.session_state.current_documents_metadata # Pass the full list
    )
    
    # Reset chat history and challenge mode when new documents are uploaded
    st.session_state.chat_history = []
    reset_challenge_mode()
    st.session_state.auto_summary = None # Clear previous summary
    st.session_state.summary_generated = False # Reset summary generation flag
    st.session_state.comparison_result = None # Clear previous comparison result

    # UI for progress
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    with status_placeholder:
        status_placeholder.info("Initializing models...")

    embedding_model = st.session_state.embedding_model
    llm = st.session_state.llm # Use the already initialized LLM

    if embedding_model and llm:
        # Pass only the newly uploaded files for processing, as create_vector_db_and_rebuild
        # will clear and rebuild the collection for the session.
        # This function now ensures that the original files are read and embedded.
        num_chunks = create_vector_db_and_rebuild(
            embedding_model,
            documents_to_process_paths=files_to_process_for_embedding, # Only process newly uploaded files
            chat_session_id=st.session_state.current_chat_id,
            progress_bar=progress_bar,
            status_text=status_placeholder
        )
        st.session_state.current_document_chunks = num_chunks # Store number of chunks

        # After rebuilding, always attempt to load the vectorstore for the current session
        st.session_state.vectorstore = load_existing_vector_db(embedding_model, st.session_state.current_chat_id)

        if st.session_state.vectorstore:
            st.session_state.rag_chain = create_rag_chain(llm) # Pass LLM, not vectorstore here
            st.session_state.document_uploaded_for_current_session = True # Set to True here
            status_placeholder.success("Document(s) processed and knowledge base ready for this chat!")
            # Load full document text from DB immediately after successful embedding
            st.session_state.current_session_full_document_text = get_all_document_chunks_text_for_session(st.session_state.current_chat_id)
        else:
            status_placeholder.error("Failed to create vector database. Please check logs.")
            st.session_state.document_uploaded_for_current_session = False # Set to False on failure
            st.session_state.current_session_full_document_text = "" # Clear if DB fails
        progress_bar.progress(100) # Ensure progress bar is full
    else:
        status_placeholder.error("LLM or Embedding model could not be initialized. Check configuration.")
        st.session_state.document_uploaded_for_current_session = False # Set to False on model init failure
        st.session_state.current_session_full_document_text = ""
    
    # Clear the temporary DOC_FOLDER files after they have been processed and embedded
    clear_doc_folder_only_temp_files()

    st.session_state.chat_sessions_list = load_chat_sessions_db() # Reload sessions list to reflect document update
    st.rerun() # Rerun to update UI with new document

def reset_challenge_mode():
    """Resets challenge mode state."""
    st.session_state.challenge_questions = []
    st.session_state.user_answers = ["", "", ""]
    st.session_state.evaluation_results = [{}, {}, {}]

def get_chat_history_as_markdown() -> str:
    """Formats the chat history into a Markdown string for export."""
    markdown_output = f"# Chat History for: {st.session_state.active_chat_name}\n\n"
    for message in st.session_state.chat_history:
        role = message["role"].capitalize()
        content = message["content"]
        markdown_output += f"**{role}:** {content}\n\n"
        if role == "Assistant" and "snippets" in message and message["snippets"]:
            markdown_output += "### Supporting Snippets:\n"
            for i, snippet in enumerate(message["snippets"]):
                markdown_output += f"> **Snippet {i+1}:** {snippet}\n\n"
        markdown_output += "---\n\n" # Separator
    return markdown_output

def get_evaluation_results_as_markdown() -> str:
    """Formats the challenge mode evaluation results into a Markdown string for export."""
    markdown_output = f"# Challenge Mode Results for: {st.session_state.active_chat_name}\n\n"
    if not st.session_state.challenge_questions:
        markdown_output += "No challenge questions were generated or evaluated.\n"
        return markdown_output

    for i, q in enumerate(st.session_state.challenge_questions):
        markdown_output += f"## Question {i+1}:\n"
        markdown_output += f"**Question:** {q}\n\n"
        markdown_output += f"**Your Answer:** {st.session_state.user_answers[i]}\n\n"

        result = st.session_state.evaluation_results[i]
        if result:
            markdown_output += "### Evaluation:\n"
            markdown_output += f"- **Accuracy:** {result.get('accuracy_score', 'N/A')}/5\n"
            markdown_output += f"- **Completeness:** {result.get('completeness_score', 'N/A')}/5\n"
            markdown_output += f"- **Clarity:** {result.get('clarity_score', 'N/A')}/5\n"
            markdown_output += f"**Feedback:** {result.get('feedback', 'No detailed feedback provided.')}\n\n"
        else:
            markdown_output += "No evaluation available for this question.\n\n"
        markdown_output += "---\n\n" # Separator
    return markdown_output

# --- Initial Load / Session Setup ---
# Initialize models at the very top of the script's execution
# This ensures they are always ready for any session loading or new session creation
if st.session_state.embedding_model is None:
    st.session_state.embedding_model = get_embedding_model()

# Handle LLM choice and re-initialization if choice changes
current_llm_choice = st.session_state.llm_choice
st.session_state.llm = get_llm(current_llm_choice) # Re-initialize LLM based on current choice

# If models failed to load, display an error and stop
if st.session_state.embedding_model is None or st.session_state.llm is None:
    st.error("Critical: Embedding model or LLM could not be initialized. Please check your .env file, Ollama server status, or Google API Key.")
    st.stop() # Stop the app if core models aren't available

if st.session_state.current_chat_id is None:
    # On first load, or if no chat is active, create a new one
    reset_current_chat_session()
    # The rerun from reset_current_chat_session will handle re-rendering

# Load chat sessions list on every run to keep it updated
st.session_state.chat_sessions_list = load_chat_sessions_db()

# --- Sidebar ---
with st.sidebar:
    st.title("📚 QueryDoc AI")
    st.markdown("---")

    # LLM Selection
    # Streamlit's st.radio now supports the 'help' parameter directly for tooltips.
    # The label for st.radio can be the subheader itself.
    llm_options = ["Gemini", "Local LLM (Ollama)"]
    selected_llm = st.radio(
        "Choose LLM", # This will be the main label for the radio group
        llm_options,
        key="llm_selector",
        index=llm_options.index(st.session_state.llm_choice),
        help="Gemini: Faster, cloud-based (data processed by Google). Local LLM (Ollama): Slower, runs locally (better privacy)."
    )

    if selected_llm != st.session_state.llm_choice:
        st.session_state.llm_choice = selected_llm
        st.session_state.llm = get_llm(st.session_state.llm_choice) # Re-initialize LLM
        st.rerun() # Rerun to apply new LLM choice

    st.markdown("---")

    # New Chat Button
    if st.button("➕ New Chat", key="new_chat_button"):
        reset_current_chat_session()

    st.markdown("---")
    st.subheader("Past Chats")

    if st.session_state.chat_sessions_list:
        for session in st.session_state.chat_sessions_list:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                display_name = session['name']
                # Display document names if available
                if session['documents_metadata']:
                    doc_names = ", ".join([d['name'] for d in session['documents_metadata']])
                    display_name += f" ({doc_names})"
                if session['id'] == st.session_state.current_chat_id:
                    st.markdown(f"**➡️ {display_name}** <small>({session['created_at']})</small>", unsafe_allow_html=True)
                else:
                    if st.button(display_name, key=f"load_chat_{session['id']}"):
                        load_chat_session(session['id'])
                        st.stop() # Stop execution and rerun to load new session
                    st.markdown(f"<small>({session['created_at']})</small>", unsafe_allow_html=True)

            with col2:
                if st.button("🗑️", key=f"delete_chat_{session['id']}"):
                    delete_selected_chat_session(session['id'])
                    st.stop() # Stop execution and rerun to update UI
    else:
        st.info("No past chats. Click '➕ New Chat' to start one!")

    st.markdown("---")
    st.subheader("Current Chat Document(s)") 
    if st.session_state.current_chat_id:
        st.markdown(f"**Active Chat:** `{st.session_state.active_chat_name}`")
        
        if st.session_state.current_documents_metadata:
            st.markdown(f"**Document(s):**")
            for doc_meta in st.session_state.current_documents_metadata:
                doc_name = doc_meta.get('name')
                doc_path = doc_meta.get('path') # This is the original path where it was uploaded
                
                if doc_name and doc_path:
                    # Provide download button for the original file path
                    # This will attempt to download from the original path, not necessarily DOC_FOLDER
                    if os.path.exists(doc_path):
                        create_base64_download_button(doc_path, label=f"📄 Download {doc_name}")
                    else:
                        st.info(f"Original file '{doc_name}' not found at its stored path. Document content for Q&A, Summarization, Challenge, and Comparison is retrieved from the database.")
                else:
                    st.warning("Invalid document metadata found for this chat.")
        else:
            st.info("No document(s) uploaded for this chat.")

        # Allow renaming current chat
        new_chat_name = st.text_input("Rename Chat:", value=st.session_state.active_chat_name, key="rename_chat_input")
        if new_chat_name != st.session_state.active_chat_name:
            rename_chat_session_db(st.session_state.current_chat_id, new_chat_name)
            st.session_state.active_chat_name = new_chat_name
            st.session_state.chat_sessions_list = load_chat_sessions_db() # Reload to update sidebar name
            st.rerun()

        # Changed to accept_multiple_files=True
        uploaded_files = st.file_uploader(
            "Upload document(s) for this chat",
            type=["pdf", "txt"],
            accept_multiple_files=True, # Allow multiple documents
            key="document_uploader_main"
        )

        # Process uploaded files if any new ones are detected
        if uploaded_files:
            # Check if any of the uploaded files are genuinely new or updated
            new_files_detected = False
            for uploaded_file in uploaded_files:
                existing_doc_names = {d['name'] for d in st.session_state.current_documents_metadata}
                if uploaded_file.name not in existing_doc_names:
                    new_files_detected = True
                    break
            
            # If new files are detected or it's the first upload for this session
            if new_files_detected or (not st.session_state.current_documents_metadata and uploaded_files):
                process_uploaded_documents_for_session(uploaded_files) 
                st.rerun() # Rerun to update UI with new document(s)

    st.markdown("---")
    st.subheader("About")
    st.info("This assistant helps you interact with your research papers, legal files, or technical manuals. It uses advanced AI models for understanding, summarization, and interactive questioning.")
    st.markdown("Developed by Alok Agarwal for EZ.")

# --- Main Content Area ---
# Inject global CSS
st.markdown(
    """
    <style>
    /* Ensure no transform property interferes with fixed positioning */
    /* Target common Streamlit root containers and HTML/body */
    html, body {
        height: 100%;
        margin: 0;
        padding: 0;
        overflow: hidden; /* Prevent body scroll if Streamlit manages its own scroll */
    }

    .stApp {
        height: 100vh; /* Ensure Streamlit app takes full height */
        overflow: hidden; /* Prevent main app scroll, let inner containers handle it */
        transform: none !important; /* Crucial for fixed positioning */
    }

    /* Specific Streamlit containers that might interfere */
    .st-emotion-cache-z5fcl4, /* Main content block */
    .st-emotion-cache-1iy32u7 { /* Tab content area */
        transform: none !important; /* Ensure these don't create new stacking contexts */
    }

    /* Make the entire main content area a flex container */
    .st-emotion-cache-z5fcl4 { /* This is the main block container for the app content */
        display: flex;
        flex-direction: column;
        height: 100vh; /* Full viewport height */
        padding-top: 1rem; /* Keep existing padding */
        /* Adjusted padding-bottom to account for the fixed input's height */
        padding-bottom: 100px; /* Increased for more clearance for the fixed input */
    }

    /* Target the div that contains the tabs and the content below them */
    .st-emotion-cache-1iy32u7 { /* Common class for the tab content area */
        flex-grow: 1; /* Allow it to take available space */
        display: flex;
        flex-direction: column;
        min-height: 0; /* Important for flex items with overflow */
    }

    /* Styles for the chat history container within tab1 */
    .chat-history-scroll-area {
        flex-grow: 1; /* Take all available space within its flex parent */
        overflow-y: auto; /* Enable vertical scrolling */
        min-height: 0; /* Essential for flex items with overflow */
        padding-right: 15px; /* Add some padding for the scrollbar */
    }

    /* Styles for the fixed input container */
    .fixed-input-container {
        position: fixed;
        bottom: 0;
        /* Calculate left dynamically based on sidebar width and main content padding */
        /* Assuming sidebar is ~300px and main content has ~1rem (16px) left padding */
        left: calc(300px + 1rem); /* Adjust this if sidebar or padding changes */
        right: 0;
        background-color: var(--background-color); /* Use Streamlit's background color */
        padding: 10px 20px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: center;
        /* Responsive adjustments */
        width: calc(100% - (300px + 2rem)); /* Adjust width for sidebar and padding */
        max-width: 800px; /* Limit width for better readability on large screens */
        margin: 0 auto; /* Center the input if max-width is applied */
    }

    /* Adjust for smaller screens where sidebar might collapse or be narrower */
    @media (max-width: 768px) {
        .fixed-input-container {
            left: 1rem; /* Less padding on mobile */
            right: 1rem;
            width: calc(100% - 2rem); /* Full width minus padding */
        }
        .st-emotion-cache-z5fcl4 {
            padding-bottom: 120px; /* More padding for smaller screens */
        }
    }

    /* Style for the chat input within the fixed container */
    .fixed-input-container .stTextInput {
        width: 100%;
    }

    /* Hide the default chat input that might appear elsewhere, if any */
    /* This targets the original st.chat_input element's container */
    .st-emotion-cache-1kyxreq { /* Common class for st.chat_input's outer div */
        visibility: hidden;
        height: 0px;
        margin: 0px;
        padding: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.header(f"Chat Session: {st.session_state.active_chat_name}")

if st.session_state.current_documents_metadata:
    st.markdown(f"### Current Document(s):")
    for doc_meta in st.session_state.current_documents_metadata:
        doc_name = doc_meta.get('name')
        doc_path = doc_meta.get('path') # This is the original path where it was uploaded
        
        if doc_name and doc_path:
            # Provide download button for the original file path
            # This will attempt to download from the original path, not necessarily DOC_FOLDER
            if os.path.exists(doc_path):
                create_base64_download_button(doc_path, label=f"📄 Download {doc_name}")
            else:
                st.info(f"Original file '{doc_name}' not found locally at '{doc_path}'. Document content for Q&A, Summarization, Challenge, and Comparison is retrieved from the database.")
        else:
            st.warning("Invalid document metadata found for this chat.")
    st.markdown(f"**Total Chunks:** {st.session_state.current_document_chunks}")
else:
    st.info("No document(s) uploaded for this chat session. Please upload one or more using the uploader in the sidebar to enable RAG features.")

st.markdown("---")

# Use tabs for interaction modes
tab1, tab2, tab3, tab4 = st.tabs(["❓ Ask Anything (Intelligent Q&A)", "📝 Summarize Document", "🧠 Challenge Me!", "📊 Document Comparison"])

with tab1: # Ask Anything Mode
    st.markdown("### ❓ Ask Anything about the Document(s)")
    st.info("Ask questions and get answers with supporting snippets from your document(s). You can ask follow-up questions!")

    # Chat history container
    chat_history_container = st.container(border=False)

    with chat_history_container:
        st.markdown('<div class="chat-history-scroll-area">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and "snippets" in message and message["snippets"]:
                    with st.expander("🔍 Supporting Snippets"):
                        for i, snippet in enumerate(message["snippets"]):
                            st.markdown(f"**Snippet {i+1}:**")
                            st.markdown(f"> *{snippet}*") 
        st.markdown('</div>', unsafe_allow_html=True)

    # Export Chat History button
    if st.session_state.chat_history:
        st.download_button(
            label="⬇️ Download Chat History",
            data=get_chat_history_as_markdown(),
            file_name=f"chat_history_{st.session_state.current_chat_id}.md",
            mime="text/markdown",
            key="download_chat_history"
        )


with tab2: # Summarize Document Mode
    st.markdown("### 📝 Document Summary")
    st.info("Get a concise summary of your uploaded document(s).")

    # Summarization now relies on content retrieved from the database
    if st.session_state.current_document_chunks > 0: # Check if there are chunks in DB
        if not st.session_state.summary_generated:
            with st.spinner("Generating document summary..."):
                # Get content directly from DB using the current chat ID
                all_document_content = get_document_content_for_summary(st.session_state.current_chat_id)

                if all_document_content.strip() and st.session_state.llm:
                    st.session_state.auto_summary = generate_auto_summary(all_document_content, st.session_state.llm)
                    st.session_state.summary_generated = True # Set flag to true
                    st.rerun() # Rerun to display summary
                else:
                    st.warning("Cannot generate summary. Document content not available from database or LLM not initialized.")

        if st.session_state.auto_summary:
            st.markdown("#### Concise Summary (≤ 150 words)")
            st.info(st.session_state.auto_summary)
        else:
            st.warning("Summary not available. Document content might be empty in DB or AI model failed to summarize.")
    else:
        st.info("No document content found in the database for summarization. Please upload document(s) to enable this feature.")

with tab3: # Challenge Me Mode
    st.markdown("### 🧠 Challenge Me!")
    st.info("The assistant will generate logic-based questions from the document(s). Try to answer them!")

    # Challenge mode now relies on content retrieved from the database
    if st.session_state.current_document_chunks > 0: # Check if there are chunks in DB
        if st.button("Generate New Questions", key="generate_questions_button"):
            reset_challenge_mode() # Clear previous questions and answers
            
            # Get content directly from DB using the current chat ID
            all_document_content_for_challenge = get_all_document_chunks_text_for_session(st.session_state.current_chat_id)

            if all_document_content_for_challenge.strip() and st.session_state.llm:
                with st.spinner("Generating challenge questions..."):
                    questions = generate_challenge_questions(all_document_content_for_challenge, st.session_state.llm)
                    st.session_state.challenge_questions = questions
                    st.session_state.user_answers = [""] * len(questions) # Initialize answers for new questions
                    st.session_state.evaluation_results = [{}] * len(questions)
                    st.rerun() # Rerun to display new questions
            else:
                st.warning("Cannot generate questions. Document content not available from database or LLM not initialized.")


        if st.session_state.challenge_questions:
            st.markdown("#### Your Challenge Questions:")
            for i, q in enumerate(st.session_state.challenge_questions):
                st.markdown(f"**Question {i+1}:** {q}")
                st.session_state.user_answers[i] = st.text_area(f"Your answer for Q{i+1}:", value=st.session_state.user_answers[i], key=f"user_answer_{i}")

            if st.button("Evaluate My Answers", key="evaluate_answers_button"):
                # Get content directly from DB using the current chat ID
                all_document_content_for_challenge = get_all_document_chunks_text_for_session(st.session_state.current_chat_id)

                if all_document_content_for_challenge.strip() and st.session_state.llm:
                    with st.spinner("Evaluating your answers..."):
                        for i, q in enumerate(st.session_state.challenge_questions):
                            user_ans = st.session_state.user_answers[i]
                            if user_ans.strip():
                                evaluation = evaluate_user_answer(q, user_ans, all_document_content_for_challenge, st.session_state.llm)
                                st.session_state.evaluation_results[i] = evaluation
                            else:
                                st.session_state.evaluation_results[i] = {
                                    "accuracy_score": 0,
                                    "completeness_score": 0,
                                    "clarity_score": 0,
                                    "feedback": "Please provide an answer to evaluate."
                                }
                    st.rerun() # Rerun to display evaluations
                else:
                    st.warning("Cannot evaluate answers. Document content not available from database or LLM not initialized.")

            st.markdown("#### Evaluation Results:")
            for i, result in enumerate(st.session_state.evaluation_results):
                if result: # Check if the result dictionary is not empty
                    st.markdown(f"**Feedback for Q{i+1}:**")
                    st.markdown(f"""
                        - **Accuracy:** {result.get('accuracy_score', 'N/A')}/5
                        - **Completeness:** {result.get('completeness_score', 'N/A')}/5
                        - **Clarity:** {result.get('clarity_score', 'N/A')}/5
                    """)
                    st.info(result.get('feedback', 'No detailed feedback provided.'))

            # Export Challenge Results button
            if any(st.session_state.evaluation_results): # Only show if at least one evaluation is present
                st.download_button(
                    label="⬇️ Download Challenge Results",
                    data=get_evaluation_results_as_markdown(),
                    file_name=f"challenge_results_{st.session_state.current_chat_id}.md",
                    mime="text/markdown",
                    key="download_challenge_results"
                )
    else:
        st.info("No document content found in the database to generate challenge questions. Please upload document(s) to enable this feature.")


with tab4: # Document Comparison Mode
    st.markdown("### 📊 Compare Documents")
    st.info("Enter a concept or topic to compare how it's discussed across your uploaded documents.")

    # Document comparison now relies on embeddings and retrieved chunks from the database
    if st.session_state.current_document_chunks > 0: # Check if there are chunks in DB
        comparison_query = st.text_input("What concept or topic would you like to compare?", key="comparison_query")
        if st.button("Compare Documents", key="compare_button", disabled=not comparison_query.strip()):
            if st.session_state.llm and st.session_state.vectorstore:
                with st.spinner("Comparing documents... This might take a moment."):
                    # For comparison, we need to retrieve documents from the current chat's vectorstore
                    st.session_state.comparison_result = compare_documents_with_llm(
                        comparison_query,
                        st.session_state.llm,
                        st.session_state.vectorstore.as_retriever(search_kwargs={"k": 10}) # Retrieve more docs for comparison
                    )
                    st.rerun() # Rerun to display comparison result
            else:
                st.warning("LLM or Vectorstore not initialized. Please ensure document(s) are uploaded and processed.")

        if st.session_state.comparison_result:
            st.markdown("#### Comparison Analysis:")
            st.markdown(st.session_state.comparison_result)
        else:
            st.info("Enter a query and click 'Compare Documents' to see the analysis.")
    else:
        st.info("No document content found in the database for comparison. Please upload document(s) to enable this feature.")


# --- Fixed Input at the Bottom (Always Visible) ---
st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)
# The chat input is enabled if document_uploaded_for_current_session is True,
# which now correctly reflects if the RAG chain and vectorstore are ready.
if st.session_state.document_uploaded_for_current_session:
    user_question = st.chat_input("Ask a question about the document(s)...", key="chat_input_bottom_fixed")
else:
    # Show a disabled input or a message if no document is uploaded or embeddings failed to load
    st.text_input("Please upload document(s) to ask questions.", disabled=True, key="chat_input_disabled")
    user_question = None # Ensure user_question is None if input is disabled
st.markdown('</div>', unsafe_allow_html=True)

# Logic for processing user_question, now triggered if user_question is not None
if user_question:
    # Append user question to history immediately
    user_message_data = {"role": "user", "content": user_question}
    st.session_state.chat_history.append(user_message_data)
    save_chat_message_db(st.session_state.current_chat_id, "user", user_question)

    # Process and append assistant answer
    with st.spinner("Finding answer..."):
        # Check if RAG chain and vectorstore are ready (document_uploaded_for_current_session flag)
        if st.session_state.rag_chain and st.session_state.vectorstore and st.session_state.document_uploaded_for_current_session:
            try:
                # Prepare contextual query
                contextual_query = user_question
                # Add last few turns for context (e.g., last 2 user/assistant pairs)
                # Ensure we don't go out of bounds if chat_history is small
                history_for_context = st.session_state.chat_history[-4:]
                if history_for_context:
                    context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history_for_context])
                    contextual_query = f"Conversation history:\n{context_str}\n\nNew question: {user_question}\n" # Added newline for clarity
                    logging.info(f"Contextual query: {contextual_query}")

                answer_text, snippets, retrieved_docs = st.session_state.rag_chain(
                    contextual_query, st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                )

                assistant_message_data = {
                    "role": "assistant",
                    "content": answer_text,
                    "snippets": snippets
                }
                st.session_state.chat_history.append(assistant_message_data)
                save_chat_message_db(st.session_state.current_chat_id, "assistant", answer_text, snippets)

                st.rerun() # Rerun to update chat history display
            except Exception as e:
                st.error(f"Error getting answer: {e}")
                logging.error(f"Error during RAG chain invocation: {e}", exc_info=True)
                error_message_data = {"role": "assistant", "content": f"Sorry, I encountered an error: {e}"}
                st.session_state.chat_history.append(error_message_data)
                save_chat_message_db(st.session_state.current_chat_id, "assistant", error_message_data["content"])
        else:
            warning_message = "Please upload document(s) and ensure the knowledge base is built for this chat session to ask questions."
            st.warning(warning_message)
            warning_message_data = {"role": "assistant", "content": warning_message}
            st.session_state.chat_history.append(warning_message_data)
            save_chat_message_db(st.session_state.current_chat_id, "assistant", warning_message_data["content"])
