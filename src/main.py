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
    load_existing_vector_db, # <--- ADDED THIS IMPORT
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
                
                # Generate auto summary
                document_content_for_summary = get_document_content_for_summary(file_path)
                if document_content_for_summary:
                    st.session_state.auto_summary = generate_auto_summary(document_content_for_summary, llm)
                    st.success("Document processed and knowledge base ready!")
                else:
                    st.warning("Could not extract content for summarization.")
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
    st.markdown("Upload a document (PDF/TXT) to get started. The assistant can answer questions and challenge your comprehension.")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt"],
        accept_multiple_files=False,
        key="document_uploader"
    )

    # Check if a new file was uploaded or if the app was rerun with the same file
    if uploaded_file and uploaded_file.name != st.session_state.last_uploaded_file_name:
        process_uploaded_document(uploaded_file)
        reset_challenge_mode() # Reset challenge mode if a new document is uploaded
        st.session_state.interaction_mode = "Ask Anything" # Default to Ask Anything after upload
        st.rerun() # Rerun to update UI with new document

    st.markdown("---")
    st.subheader("Interaction Mode")
    if st.session_state.document_uploaded:
        st.session_state.interaction_mode = st.radio(
            "Choose a mode:",
            ("Ask Anything", "Challenge Me"),
            index=0 if st.session_state.interaction_mode == "Ask Anything" else 1
        )
    else:
        st.info("Upload a document to enable interaction modes.")

    st.markdown("---")
    st.subheader("About")
    st.info("This assistant helps you interact with your research documents. It uses advanced AI models for understanding, summarization, and interactive questioning.")
    st.markdown("Developed by EZ.")

# --- Main Content Area ---
st.header("Document Interaction")

if not st.session_state.document_uploaded:
    st.info("Please upload a document in the sidebar to begin.")
else:
    st.markdown(f"### Current Document: {os.path.basename(st.session_state.current_document_path)}")
    create_base64_download_button(st.session_state.current_document_path)

    if st.session_state.auto_summary:
        st.markdown("#### Document Summary (≤ 150 words)")
        st.info(st.session_state.auto_summary)
    else:
        st.warning("Summary not available. Document might be too large or LLM failed to summarize.")

    st.markdown("---")

    # --- Ask Anything Mode ---
    if st.session_state.interaction_mode == "Ask Anything":
        st.markdown("### ❓ Ask Anything about the Document")
        user_question = st.text_area("Enter your question here:", key="ask_anything_question")

        if st.button("Get Answer", key="get_answer_button", disabled=not user_question.strip()):
            if st.session_state.rag_chain and st.session_state.vectorstore: # Ensure vectorstore is also available
                with st.spinner("Finding answer..."):
                    try:
                        # The rag_chain now returns (response_text, snippets, docs)
                        answer_text, snippets, retrieved_docs = st.session_state.rag_chain(
                            user_question, st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                        )
                        st.markdown("#### ✨ Answer:")
                        st.write(answer_text)
                        
                        if snippets:
                            st.markdown("---")
                            st.markdown("#### 🔍 Supporting Snippets from Document:")
                            for i, snippet in enumerate(snippets):
                                with st.expander(f"Snippet {i+1}"):
                                    st.write(snippet) # Display each snippet inside an expander
                        else:
                            st.info("No specific supporting snippets identified for this answer.")
                            
                    except Exception as e:
                        st.error(f"Error getting answer: {e}")
                        logging.error(f"Error during RAG chain invocation: {e}", exc_info=True)
            else:
                st.warning("Please upload a document and ensure the knowledge base is built.")

    # --- Challenge Me Mode ---
    elif st.session_state.interaction_mode == "Challenge Me":
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