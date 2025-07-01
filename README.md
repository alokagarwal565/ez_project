Smart Research Assistant
This project implements an AI-powered assistant designed to help users understand and reason through large documents like research papers, legal files, or technical manuals. It goes beyond basic summarization and keyword search by providing contextual understanding, logic-based question generation, and answer justification.

Features
Document Upload: Supports PDF and TXT file formats.

Auto Summary: Generates a concise summary (≤ 150 words) immediately after document upload.

Two Interaction Modes:

Ask Anything: Users can ask free-form questions based on the document, and the assistant provides answers grounded in the document's content with clear justifications.

Challenge Me: The system generates three logic-based or comprehension-focused questions from the document. Users can attempt to answer these, and the assistant evaluates their responses, providing feedback and justification based on the document.

Contextual Understanding & Justification: All answers and evaluations are directly supported by the uploaded document, preventing hallucination and providing clear references.

Clean Web Interface: Built with Streamlit for an intuitive and responsive user experience.

Application Architecture / Reasoning Flow
The application follows a Retrieval-Augmented Generation (RAG) architecture, leveraging a Large Language Model (LLM) and a vector database for efficient information retrieval and generation.

Frontend (Streamlit):

Handles user interaction: document upload, mode selection ("Ask Anything" / "Challenge Me"), question input, and display of answers/feedback.

Manages session state to maintain context across interactions.

Backend (Python - RAG Pipeline):

Document Loading & Splitting (rag_pipeline.py):

When a user uploads a PDF or TXT file, it's saved locally.

PyPDFLoader or TextLoader loads the document.

RecursiveCharacterTextSplitter breaks the document into smaller, overlapping "chunks" suitable for embedding.

Embedding (rag_pipeline.py):

HuggingFaceEmbeddings (or GoogleGenerativeAIEmbeddings if GOOGLE_API_KEY is provided) converts each text chunk into a numerical vector (embedding).

Vector Database (PGVector):

PGVector (PostgreSQL with pgvector extension) stores these embeddings along with their original text chunks.

When a new document is uploaded, the existing collection for the assistant is cleared, and the new document's chunks are added, effectively "rebuilding" the knowledge base for the current session.

LLM Integration (rag_pipeline.py, challenge_mode.py):

ChatGoogleGenerativeAI is used as the primary LLM (e.g., gemini-1.5-flash-latest).

Ask Anything Mode:

User's question is embedded.

The vector database retrieves the most semantically similar document chunks (context).

The retrieved context and the user's question are fed into the LLM via a ChatPromptTemplate (RAG_PROMPT_TEMPLATE).

The LLM generates an answer based only on the provided context and includes a justification by quoting/paraphrasing from the source.

Challenge Me Mode:

Question Generation: The full document content is sent to the LLM with a specific prompt (CHALLENGE_QUESTION_GENERATION_PROMPT) to generate three logic-based questions.

Answer Evaluation: For each user answer, the original question, the user's answer, and the full document content are sent to the LLM with an evaluation prompt (ANSWER_EVALUATION_PROMPT). The LLM provides feedback and justification.

Auto Summary (rag_pipeline.py):

The full document content is passed to the LLM with a SUMMARY_PROMPT_TEMPLATE to generate a concise summary.

Setup Instructions
Follow these steps to get the Smart Research Assistant up and running on your local machine.

Prerequisites
Python 3.9+

PostgreSQL with pgvector extension enabled:

Install PostgreSQL (e.g., via Homebrew on macOS, apt on Ubuntu, or official installer on Windows).

Enable the pgvector extension. You might need to install it first (e.g., CREATE EXTENSION vector; in your PostgreSQL client).

Ensure your PostgreSQL server is running.

1. Clone the Repository
(Assuming this project is provided as a repository)

git clone <repository_url>
cd smart_assistant_research_summarization

2. Set Up Virtual Environment (Recommended)
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Configure Environment Variables
Create a .env file in the root directory of the project (smart_assistant_research_summarization/) and add your configurations.

# .env
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"

# PostgreSQL Database Configuration
PG_USER="your_pg_user"
PG_PASSWORD="your_pg_password"
PG_HOST="localhost"
PG_PORT="5432"
PG_DBNAME="your_pg_database_name" # e.g., postgres

GOOGLE_API_KEY: Obtain this from Google AI Studio. This is crucial for the LLM operations.

PG_USER, PG_PASSWORD, PG_HOST, PG_PORT, PG_DBNAME: Your PostgreSQL database connection details. Ensure the PG_DBNAME database exists and your user has permissions.

5. Run the Application
Navigate to the src directory and run the Streamlit application:

cd src
streamlit run main.py

This will open the application in your web browser, typically at http://localhost:8501.

Usage
Upload Document: Use the "Upload a document" file uploader in the sidebar to select a PDF or TXT file.

Processing: The application will process the document, build the knowledge base, and display a concise summary.

Choose Mode: Select "Ask Anything" or "Challenge Me" from the sidebar.

Ask Anything: Type your question in the text area and click "Get Answer."

Challenge Me: Click "Generate New Questions" to get three questions. Type your answers and click "Evaluate My Answers."

Review Results: The assistant will provide answers/feedback with justifications based on the document content.

Project Structure
smart_assistant_research_summarization/

.env: Environment variables (ignored by Git).

requirements.txt: Python dependencies.

README.md: Project description, setup, and architecture.

data/: Directory for temporarily storing uploaded documents.

src/: Contains the core Python source code.

main.py: The main Streamlit application script.

config.py: Configuration settings for paths, models, and database.

helpers.py: Utility functions (e.g., download button, DB extension check).

rag_pipeline.py: Handles document loading, splitting, embedding, vector database interaction, RAG chain creation, and auto-summarization.

prompt_templates.py: Stores all LLM prompt templates.

challenge_mode.py: Contains logic for generating challenge questions and evaluating user answers.

Evaluation Criteria Focus
This project is built with the following evaluation criteria in mind:

Response Quality (Accuracy + Justification): Answers are strictly grounded in the document, and justifications are provided by quoting or referencing the source.

Reasoning Mode Functionality: The "Challenge Me" mode is implemented to generate logic-based questions and provide detailed, justified evaluations.

UI/UX and Smooth Flow: Streamlit provides a clean, interactive interface.

Code Structure & Documentation: Modularized code, clear comments, and this comprehensive README.md.

Minimal Hallucination & Good Context Use: Achieved through robust RAG practices and prompt engineering that emphasizes using only provided context.