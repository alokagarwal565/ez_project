# ===============================================================
# File: src/main.py
# Description: Main Streamlit application for the Smart Assistant.
# ===============================================================

import streamlit as st
import os
import logging
from typing import List, Tuple
import shutil # For clearing the data folder

from config import DOC_FOLDER, GOOGLE_API_KEY
from rag_pipeline import (
    get_embedding_model,
    get_gemini_rag_llm,
    load_existing_vector_db,
    create_vector_db_and_rebuild,
    create_rag_chain,
    generate_auto_summary,
    get_document_content_for_summary,
    compare_documents_with_llm # Import the new comparison function
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
if "current_document_paths" not in st.session_state: # Changed to list for multi-doc
    st.session_state.current_document_paths = []
if "auto_summary" not in st.session_state:
    st.session_state.auto_summary = None
if "interaction_mode" not in st.session_state:
    st.session_state.interaction_mode = "Ask Anything" # Default mode
if "challenge_questions" not in st.session_state:
    st.session_state.challenge_questions = []
if "user_answers" not in st.session_state:
    st.session_state.user_answers = ["", "", ""] # For 3 questions
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = [{}, {}, {}]
if "last_uploaded_file_names" not in st.session_state: # Changed to list for multi-doc
    st.session_state.last_uploaded_file_names = []
if "chat_history" not in st.session_state: # For ChatGPT-style interface
    st.session_state.chat_history = []
if "summary_generated" not in st.session_state: # To track if summary has been generated
    st.session_state.summary_generated = False
if "current_document_chunks" not in st.session_state: # To store number of chunks
    st.session_state.current_document_chunks = 0
if "comparison_result" not in st.session_state: # For document comparison
    st.session_state.comparison_result = None


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

def process_uploaded_document(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]):
    """Handles saving the uploaded files, processing them, and rebuilding the DB."""
    clear_data_folder() # Clear previous documents

    saved_file_paths = []
    uploaded_file_names = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(DOC_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_file_paths.append(file_path)
        uploaded_file_names.append(uploaded_file.name)

    st.session_state.current_document_paths = saved_file_paths
    st.session_state.document_uploaded = True
    st.session_state.last_uploaded_file_names = uploaded_file_names # Update tracker

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
    llm = st.session_state.llm

    if not embedding_model:
        embedding_model = get_embedding_model()
        st.session_state.embedding_model = embedding_model

    if not llm:
        llm = get_gemini_rag_llm() # Call the correct Gemini LLM getter
        st.session_state.llm = llm

    if embedding_model and llm:
        # Pass the specific file paths to rebuild, along with progress objects
        num_chunks = create_vector_db_and_rebuild(
            embedding_model,
            documents_to_process=saved_file_paths,
            progress_bar=progress_bar,
            status_text=status_placeholder
        )
        st.session_state.current_document_chunks = num_chunks # Store number of chunks

        st.session_state.vectorstore = load_existing_vector_db(embedding_model) # Reload vectorstore after rebuild

        if st.session_state.vectorstore:
            st.session_state.rag_chain = create_rag_chain(llm) # Pass LLM, not vectorstore here
            status_placeholder.success("Document(s) processed and knowledge base ready!")
        else:
            status_placeholder.error("Failed to create vector database. Please check logs.")
        progress_bar.progress(100) # Ensure progress bar is full
    else:
        status_placeholder.error("LLM or Embedding model could not be initialized. Check API keys and configuration.")

def reset_challenge_mode():
    """Resets challenge mode state."""
    st.session_state.challenge_questions = []
    st.session_state.user_answers = ["", "", ""]
    st.session_state.evaluation_results = [{}, {}, {}]

def get_chat_history_as_markdown() -> str:
    """Formats the chat history into a Markdown string for export."""
    markdown_output = "# Chat History\n\n"
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
    markdown_output = "# Challenge Mode Results\n\n"
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


# --- Sidebar ---
with st.sidebar:
    st.title("📚 Smart Research Assistant")
    st.markdown("Upload documents (PDF/TXT) to get started. The assistant can answer questions, summarize, and challenge your comprehension.")

    uploaded_files = st.file_uploader(
        "Upload document(s)",
        type=["pdf", "txt"], # Removed "csv"
        accept_multiple_files=True, # Allow multiple files
        key="document_uploader"
    )

    # Check if new files were uploaded or if the app was rerun with different files
    current_uploaded_file_names = [f.name for f in uploaded_files] if uploaded_files else []
    if uploaded_files and (current_uploaded_file_names != st.session_state.last_uploaded_file_names or not st.session_state.document_uploaded):
        process_uploaded_document(uploaded_files)
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


if not st.session_state.document_uploaded:
    st.info("Please upload document(s) in the sidebar to begin.")
else:
    st.markdown(f"### Current Document(s):")
    for i, doc_path in enumerate(st.session_state.current_document_paths):
        file_name = os.path.basename(doc_path)
        file_size_bytes = os.path.getsize(doc_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        st.markdown(f"- **{file_name}** ({file_size_mb:.2f} MB)")
        create_base64_download_button(doc_path, label=f"📄 Download {file_name}")

    st.markdown(f"**Total Chunks:** {st.session_state.current_document_chunks}")

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
                                st.markdown(f"> *{snippet}*") # Use blockquote for snippets for better visual distinction
            st.markdown('</div>', unsafe_allow_html=True)

        # Export Chat History button
        if st.session_state.chat_history:
            st.download_button(
                label="⬇️ Download Chat History",
                data=get_chat_history_as_markdown(),
                file_name="chat_history.md",
                mime="text/markdown",
                key="download_chat_history"
            )


    with tab2: # Summarize Document Mode
        st.markdown("### 📝 Document Summary")
        st.info("Get a concise summary of your uploaded document(s). For multiple documents, a combined summary will be generated.")

        if st.session_state.document_uploaded:
            if not st.session_state.summary_generated:
                with st.spinner("Generating document summary..."):
                    # For summary, concatenate content from all documents
                    all_document_content = ""
                    for doc_path in st.session_state.current_document_paths:
                        all_document_content += get_document_content_for_summary(doc_path) + "\n\n"

                    if all_document_content.strip() and st.session_state.llm:
                        st.session_state.auto_summary = generate_auto_summary(all_document_content, st.session_state.llm)
                        st.session_state.summary_generated = True # Set flag to true
                        st.rerun() # Rerun to display summary
                    else:
                        st.warning("Cannot generate summary. Document content not available or LLM not initialized.")

            if st.session_state.auto_summary:
                st.markdown("#### Concise Summary (≤ 150 words)")
                st.info(st.session_state.auto_summary)
            else:
                st.warning("Summary not available. Document(s) might be too large or AI model failed to summarize.")
        else:
            st.info("Please upload document(s) to generate a summary.")

    with tab3: # Challenge Me Mode
        st.markdown("### 🧠 Challenge Me!")
        st.info("The assistant will generate logic-based questions from the document(s). Try to answer them!")

        if st.button("Generate New Questions", key="generate_questions_button", disabled=not st.session_state.document_uploaded):
            reset_challenge_mode() # Clear previous questions and answers
            # For challenge mode, combine content from all documents for question generation
            all_document_content_for_challenge = ""
            for doc_path in st.session_state.current_document_paths:
                all_document_content_for_challenge += get_document_content_for_summary(doc_path) + "\n\n"

            if all_document_content_for_challenge.strip() and st.session_state.llm:
                with st.spinner("Generating challenge questions..."):
                    questions = generate_challenge_questions(all_document_content_for_challenge, st.session_state.llm)
                    st.session_state.challenge_questions = questions
                    st.session_state.user_answers = [""] * len(questions) # Initialize answers for new questions
                    st.session_state.evaluation_results = [{}] * len(questions)
                    st.rerun() # Rerun to display new questions
            else:
                st.warning("Cannot generate questions. Document content not available or LLM not initialized.")

        if st.session_state.challenge_questions:
            st.markdown("#### Your Challenge Questions:")
            for i, q in enumerate(st.session_state.challenge_questions):
                st.markdown(f"**Question {i+1}:** {q}")
                st.session_state.user_answers[i] = st.text_area(f"Your answer for Q{i+1}:", value=st.session_state.user_answers[i], key=f"user_answer_{i}")

            if st.button("Evaluate My Answers", key="evaluate_answers_button"):
                all_document_content_for_challenge = ""
                for doc_path in st.session_state.current_document_paths:
                    all_document_content_for_challenge += get_document_content_for_summary(doc_path) + "\n\n"

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
                    st.warning("Cannot evaluate answers. Document content not available or LLM not initialized.")

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
                    file_name="challenge_results.md",
                    mime="text/markdown",
                    key="download_challenge_results"
                )

    with tab4: # Document Comparison Mode
        st.markdown("### 📊 Compare Documents")
        st.info("Enter a concept or topic to compare how it's discussed across your uploaded documents.")

        if not st.session_state.document_uploaded or len(st.session_state.current_document_paths) < 2:
            st.warning("Please upload at least two documents to enable document comparison.")
        else:
            comparison_query = st.text_input("What concept or topic would you like to compare?", key="comparison_query")
            if st.button("Compare Documents", key="compare_button", disabled=not comparison_query.strip()):
                if st.session_state.llm and st.session_state.vectorstore:
                    with st.spinner("Comparing documents... This might take a moment."):
                        st.session_state.comparison_result = compare_documents_with_llm(
                            comparison_query,
                            st.session_state.llm,
                            st.session_state.vectorstore.as_retriever(search_kwargs={"k": 10}) # Retrieve more docs for comparison
                        )
                        st.rerun() # Rerun to display comparison result
                else:
                    st.warning("LLM or Vectorstore not initialized. Please ensure documents are uploaded and processed.")

            if st.session_state.comparison_result:
                st.markdown("#### Comparison Analysis:")
                st.markdown(st.session_state.comparison_result)
            elif st.session_state.document_uploaded and len(st.session_state.current_document_paths) >= 2:
                st.info("Enter a query and click 'Compare Documents' to see the analysis.")


# --- Fixed Input at the Bottom (Always Visible) ---
st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)
if st.session_state.document_uploaded:
    user_question = st.chat_input("Ask a question about the document(s)...", key="chat_input_bottom_fixed")
else:
    # Show a disabled input or a message if no document is uploaded
    st.text_input("Please upload document(s) to ask questions.", disabled=True, key="chat_input_disabled")
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
                    contextual_query = f"Conversation history:\n{context_str}\n\nNew question: {user_question}\n" # Added newline for clarity
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
            st.warning("Please upload document(s) and ensure the knowledge base is built.")
            st.session_state.chat_history.append({"role": "assistant", "content": "Please upload document(s) and ensure the knowledge base is built."})
