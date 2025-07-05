# ===============================================================
# File: src/main.py
# Description: Main Streamlit application for the Smart Assistant.
# ===============================================================

import streamlit as st
import os
import logging
from typing import List
import shutil # For clearing the data folder

from config import DOC_FOLDER, GOOGLE_API_KEY
from rag_pipeline import (
    get_embedding_model,
    get_gemini_rag_llm,
    load_existing_vector_db,
    create_vector_db_and_rebuild,
    create_rag_chain,
    generate_auto_summary,
    get_document_content_for_summary
)
from challenge_mode import generate_challenge_questions, evaluate_user_answer
from helpers import create_base64_download_button

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Smart Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
if "current_document_path" not in st.session_state:
    st.session_state.current_document_path = None
if "auto_summary" not in st.session_state:
    st.session_state.auto_summary = None
if "interaction_mode" not in st.session_state:
    st.session_state.interaction_mode = "Ask Anything" # Default mode
if "challenge_questions" not in st.session_state:
    st.session_state.challenge_questions = []
if "user_answers" not in st.session_state:
    st.session_state.user_answers = ["", "", ""] # For 3 questions
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = ["", "", ""]
if "last_uploaded_file_name" not in st.session_state: # To track if the file changed
    st.session_state.last_uploaded_file_name = None
if "chat_history" not in st.session_state: # New: For ChatGPT-style interface
    st.session_state.chat_history = []
if "summary_generated" not in st.session_state: # To track if summary has been generated
    st.session_state.summary_generated = False


# --- Helper Functions ---
def clear_data_folder():
    """Clears all files from the DOC_FOLDER."""
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

def process_uploaded_document(uploaded_file):
    """Handles saving the uploaded file, processing it, and rebuilding the DB."""
    clear_data_folder() # Clear previous documents
    
    file_path = os.path.join(DOC_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.session_state.current_document_path = file_path
    st.session_state.document_uploaded = True
    st.session_state.last_uploaded_file_name = uploaded_file.name # Update tracker
    
    # Reset chat history and challenge mode when a new document is uploaded
    st.session_state.chat_history = []
    reset_challenge_mode()
    st.session_state.auto_summary = None # Clear previous summary
    st.session_state.summary_generated = False # Reset summary generation flag

    with st.spinner("Processing document and building knowledge base..."):
        embedding_model = st.session_state.embedding_model
        llm = st.session_state.llm

        if not embedding_model:
            embedding_model = get_embedding_model()
            st.session_state.embedding_model = embedding_model
        
        if not llm:
            llm = get_gemini_rag_llm() # Call the correct Gemini LLM getter
            st.session_state.llm = llm

        if embedding_model and llm:
            # Pass the specific file path to rebuild
            create_vector_db_and_rebuild(embedding_model, documents_to_process=[file_path])
            st.session_state.vectorstore = load_existing_vector_db(embedding_model) # Reload vectorstore after rebuild
            
            if st.session_state.vectorstore:
                st.session_state.rag_chain = create_rag_chain(llm) # Pass LLM, not vectorstore here
                st.success("Document processed and knowledge base ready!")
            else:
                st.error("Failed to create vector database. Please check logs.")
        else:
            st.error("LLM or Embedding model could not be initialized. Check API keys and configuration.")

def reset_challenge_mode():
    """Resets challenge mode state."""
    st.session_state.challenge_questions = []
    st.session_state.user_answers = ["", "", ""]
    st.session_state.evaluation_results = ["", "", ""]

# --- Sidebar ---
with st.sidebar:
    st.title("📚 Smart Research Assistant")
    st.markdown("Upload a document (PDF/TXT/CSV) to get started. The assistant can answer questions and challenge your comprehension.")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "csv"], # Added CSV
        accept_multiple_files=False,
        key="document_uploader"
    )

    # Check if a new file was uploaded or if the app was rerun with the same file
    if uploaded_file and uploaded_file.name != st.session_state.last_uploaded_file_name:
        process_uploaded_document(uploaded_file)
        st.session_state.interaction_mode = "Ask Anything" # Default to Ask Anything after upload
        st.rerun() # Rerun to update UI with new document

    st.markdown("---")
    st.subheader("About")
    st.info("This assistant helps you interact with your research documents. It uses advanced AI models for understanding, summarization, and interactive questioning.")
    st.markdown("Developed by EZ.")

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
    }

    /* Style for the chat input within the fixed container */
    .fixed-input-container .stTextInput {
        width: 100%;
        max-width: 800px; /* Limit width for better readability */
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


if not st.session_state.document_uploaded:
    st.info("Please upload a document in the sidebar to begin.")
else:
    st.markdown(f"### Current Document: {os.path.basename(st.session_state.current_document_path)}")
    create_base64_download_button(st.session_state.current_document_path)

    st.markdown("---")

    # Use tabs for interaction modes
    tab1, tab2, tab3 = st.tabs(["❓ Ask Anything (Intelligent Q&A)", "📝 Summarize Document", "🧠 Challenge Me!"])

    with tab1: # Ask Anything Mode
        st.markdown("### ❓ Ask Anything about the Document")
        st.info("Ask questions and get answers with supporting snippets from your document. You can ask follow-up questions!")

        # Chat history container
        # This container will now manage its own scroll, and the input will be fixed below it.
        # Removed fixed height to allow flex-grow to manage height
        chat_history_container = st.container(border=False) 

        with chat_history_container:
            # Apply custom class to the inner div of this container for scrolling
            st.markdown('<div class="chat-history-scroll-area">', unsafe_allow_html=True)
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if message["role"] == "assistant" and "snippets" in message and message["snippets"]:
                        with st.expander("🔍 Supporting Snippets"):
                            for i, snippet in enumerate(message["snippets"]):
                                st.markdown(f"**Snippet {i+1}:** {snippet}")
            st.markdown('</div>', unsafe_allow_html=True)

        # The chat input is now moved outside the tab structure to be always visible
        # The logic for processing the user_question will still be inside the tab
        # but the input field itself is rendered globally.

    with tab2: # Summarize Document Mode
        st.markdown("### 📝 Document Summary")
        st.info("Get a concise summary of your uploaded document.")

        if st.session_state.document_uploaded:
            if not st.session_state.summary_generated:
                with st.spinner("Generating document summary..."):
                    document_content_for_summary = get_document_content_for_summary(st.session_state.current_document_path)
                    if document_content_for_summary and st.session_state.llm:
                        st.session_state.auto_summary = generate_auto_summary(document_content_for_summary, st.session_state.llm)
                        st.session_state.summary_generated = True # Set flag to true
                        st.rerun() # Rerun to display summary
                    else:
                        st.warning("Cannot generate summary. Document content not available or LLM not initialized.")
            
            if st.session_state.auto_summary:
                st.markdown("#### Concise Summary (≤ 150 words)")
                st.info(st.session_state.auto_summary)
            else:
                st.warning("Summary not available. Document might be too large or LLM failed to summarize.")
        else:
            st.info("Please upload a document to generate a summary.")

    with tab3: # Challenge Me Mode
        st.markdown("### 🧠 Challenge Me!")
        st.info("The assistant will generate logic-based questions from the document. Try to answer them!")

        if st.button("Generate New Questions", key="generate_questions_button", disabled=not st.session_state.document_uploaded):
            reset_challenge_mode() # Clear previous questions and answers
            document_content_for_challenge = get_document_content_for_summary(st.session_state.current_document_path)
            if document_content_for_challenge and st.session_state.llm:
                with st.spinner("Generating challenge questions..."):
                    questions = generate_challenge_questions(document_content_for_challenge, st.session_state.llm)
                    st.session_state.challenge_questions = questions
                    st.session_state.user_answers = [""] * len(questions) # Initialize answers for new questions
                    st.session_state.evaluation_results = [""] * len(questions) # Clear previous evaluations
                    st.rerun() # Rerun to display new questions
            else:
                st.warning("Cannot generate questions. Document content not available or LLM not initialized.")

        if st.session_state.challenge_questions:
            st.markdown("#### Your Challenge Questions:")
            for i, q in enumerate(st.session_state.challenge_questions):
                st.markdown(f"**Question {i+1}:** {q}")
                st.session_state.user_answers[i] = st.text_area(f"Your answer for Q{i+1}:", value=st.session_state.user_answers[i], key=f"user_answer_{i}")
            
            if st.button("Evaluate My Answers", key="evaluate_answers_button"):
                document_content_for_challenge = get_document_content_for_summary(st.session_state.current_document_path)
                if document_content_for_challenge and st.session_state.llm:
                    with st.spinner("Evaluating your answers..."):
                        for i, q in enumerate(st.session_state.challenge_questions):
                            user_ans = st.session_state.user_answers[i]
                            if user_ans.strip():
                                evaluation = evaluate_user_answer(q, user_ans, document_content_for_challenge, st.session_state.llm)
                                st.session_state.evaluation_results[i] = evaluation
                            else:
                                st.session_state.evaluation_results[i] = "Please provide an answer to evaluate."
                    st.rerun() # Rerun to display evaluations
                else:
                    st.warning("Cannot evaluate answers. Document content not available or LLM not initialized.")

            st.markdown("#### Evaluation Results:")
            for i, result in enumerate(st.session_state.evaluation_results):
                if result:
                    st.markdown(f"**Feedback for Q{i+1}:**")
                    st.info(result)

# --- Fixed Input at the Bottom (Always Visible) ---
# This container will be targeted by the fixed-input-container CSS
# It is now placed outside the conditional blocks for document_uploaded and tabs
st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)
if st.session_state.document_uploaded:
    user_question = st.chat_input("Ask a question about the document...", key="chat_input_bottom_fixed")
else:
    # Show a disabled input or a message if no document is uploaded
    st.text_input("Please upload a document to ask questions.", disabled=True, key="chat_input_disabled")
    user_question = None # Ensure user_question is None if input is disabled
st.markdown('</div>', unsafe_allow_html=True)

# Logic for processing user_question, now triggered if user_question is not None
if user_question:
    # Append user question to history immediately
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    
    # Process and append assistant answer
    with st.spinner("Finding answer..."):
        if st.session_state.rag_chain and st.session_state.vectorstore:
            try:
                # Prepare contextual query
                contextual_query = user_question
                # Add last few turns for context (e.g., last 2 user/assistant pairs)
                # Ensure we don't go out of bounds if chat_history is small
                history_for_context = st.session_state.chat_history[-4:] 
                if history_for_context:
                    context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history_for_context])
                    contextual_query = f"Conversation history:\n{context_str}\n\nNew question: {user_question}"
                    logging.info(f"Contextual query: {contextual_query}")

                answer_text, snippets, retrieved_docs = st.session_state.rag_chain(
                    contextual_query, st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                )
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer_text,
                    "snippets": snippets
                })
                st.rerun() # Rerun to update chat history display
            except Exception as e:
                st.error(f"Error getting answer: {e}")
                logging.error(f"Error during RAG chain invocation: {e}", exc_info=True)
                st.session_state.chat_history.append({"role": "assistant", "content": f"Sorry, I encountered an error: {e}"})
        else:
            st.warning("Please upload a document and ensure the knowledge base is built.")
            st.session_state.chat_history.append({"role": "assistant", "content": "Please upload a document and ensure the knowledge base is built."})
