# 📚 QueryDoc AI

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)](https://www.postgresql.org/)

## 🚀 Overview

The **QueryDoc AI** is an intelligent Streamlit-based application designed to help users interact with their documents through various AI-powered functionalities. It leverages **Retrieval Augmented Generation (RAG)** to answer questions based on uploaded documents, summarizes content, generates and evaluates challenge questions, and compares information across multiple documents.

The application supports both **Google's Gemini LLM** and **local Ollama models**, providing flexibility in AI model selection. It uses **PostgreSQL with PGVector** for efficient document indexing and retrieval, enabling powerful semantic search and AI-driven insights directly from your research materials.

## ✨ Features

### 📄 Document Management
- **Document Upload**: Upload PDF and TXT documents with automatic processing and indexing
- **Persistent Storage**: Documents are stored locally and indexed for efficient querying
- **Multi-Document Support**: Handle multiple documents within sessions for comprehensive research

### 💬 Chat System
- **Persistent Chat Sessions**: Create, load, rename, and delete chat sessions
- **Session History**: All chat history and associated document metadata saved in PostgreSQL database
- **Context Continuity**: Maintain conversation context across sessions

### 🧠 AI-Powered Interactions
- **Retrieval Augmented Generation (RAG)**: Ask questions about uploaded documents and get accurate, context-aware answers with source citations
- **Automated Summarization**: Generate concise summaries of uploaded documents with a single click
- **Flexible LLM Support**: Seamlessly switch between Google's Gemini models and self-hosted local LLMs via Ollama

### 💡 Challenge Mode
- **Question Generation**: Automatically generate logic-based challenge questions from document content
- **Answer Evaluation**: Evaluate user answers based on accuracy, completeness, and clarity with structured feedback
- **Comprehension Testing**: Test your understanding with AI-generated questions tailored to your documents

### 📊 Advanced Features
- **Document Comparison**: Compare concepts or findings across multiple uploaded documents
- **Vector Database Integration**: Utilizes PGVector for storing and querying document embeddings
- **HNSW Indexing**: Optimized search performance with Hierarchical Navigable Small World indexing
- **Source Citations**: Get detailed references to original document sections for all AI responses

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Backend Logic** | Python |
| **Large Language Models** | Google Gemini (`gemini-1.5-flash-latest`) & Local LLMs (Ollama) |
| **Embedding Model** | HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Vector Database** | PostgreSQL with PGVector extension |
| **Document Processing** | LangChain, PyPDFLoader, TextLoader |
| **Text Splitting** | RecursiveCharacterTextSplitter |
| **Additional Libraries** | psycopg2, python-dotenv, PIL, numpy |

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.9+**
- **PostgreSQL** (version 13 or higher recommended)
- **Google API Key** for Gemini (optional, if using Gemini LLM)
- **Ollama** (optional, if using local LLMs)

## 🔧 Setup and Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/alokagarwal565/ez_project.git
cd smart-research-assistant
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the root directory of the project and populate it with your credentials:

```env
# Google API Key (for Gemini LLM)
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"

# PostgreSQL Database Credentials
PG_USER="your_pg_user"
PG_PASSWORD="your_pg_password"
PG_HOST="localhost"
PG_PORT="5432"
PG_DBNAME="your_database_name"
```

> **⚠️ Important**: Replace the placeholder values with your actual credentials and ensure the `PG_DBNAME` database exists in your PostgreSQL server.

### Step 5: Set Up PostgreSQL with PGVector

#### Install PostgreSQL and PGVector Extension

1. **Install PostgreSQL** (version 13+)
2. **Enable PGVector extension**:
   ```sql
   CREATE EXTENSION vector;
   ```

#### Initialize Database Schema

```bash
python src/init_db.py
```

This script will create the necessary tables for PGVector and chat session management.

### Step 6: Set Up Ollama (Optional)

If you plan to use local LLMs:

1. **Download and install Ollama** from [ollama.ai](https://ollama.ai)
2. **Pull your desired model**:
   ```bash
   ollama pull llama3.2:1b
   ```

### Step 7: Run the Application

```bash
streamlit run src/main.py
```

Your web browser should automatically open the Streamlit application at `http://localhost:8501`.

## 🎯 Usage

### 1. Start a New Chat Session
- Click "New Chat" in the sidebar to begin a new session
- All sessions are automatically saved and can be accessed later

### 2. Upload Documents
- Use the "Upload Document" section to upload PDF or TXT files
- Documents are automatically processed, chunked, and embedded into the vector database
- Multiple documents can be uploaded to the same session

### 3. Select Your AI Model
- Choose between "Gemini" or "Local LLM (Ollama)" from the sidebar
- Switch models anytime during your session

### 4. Interaction Modes

#### Ask Anything (RAG Q&A)
- Type your questions in the chat input
- Get accurate, context-aware answers with source citations
- AI responses include references to original document sections

#### Auto Summary
- Click "Generate Auto-Summary" to get a concise summary of uploaded documents
- Summaries are automatically generated and can be regenerated as needed

#### Challenge Mode
- Generate logic-based challenge questions from your document content
- Answer the questions and receive detailed evaluations
- Get feedback on accuracy, completeness, and clarity with structured scoring

#### Document Comparison
- Compare insights and findings across multiple uploaded documents
- Identify relationships and contrasts between different sources

### 5. Manage Chat Sessions
- View all your chat sessions in the sidebar
- Click on any session to load its history and associated documents
- Rename or delete sessions as needed
- All chat history is preserved across sessions

## 📁 Project Structure

```
.
├── .env                      # Environment variables (create this file)
├── requirements.txt          # Python dependencies
├── src/
│   ├── main.py              # Main Streamlit application entry point
│   ├── config.py            # Configuration settings (paths, LLM names, DB connections)
│   ├── rag_pipeline.py      # RAG pipeline logic, document processing, LLM handling, DB operations
│   ├── challenge_mode.py    # Logic for generating and evaluating challenge questions
│   ├── init_db.py           # Script to initialize PostgreSQL database schema
│   ├── clear_db.py          # Script to clear PostgreSQL database tables
│   ├── helpers.py           # Utility functions (cosine similarity, download button, HNSW indexing)
│   └── prompt_templates.py  # Stores various prompt templates for LLM interactions
└── data/                     # Folder for temporarily storing uploaded documents (created automatically)
```

## 🔧 Database Management

### Initialize Database
```bash
python src/init_db.py
```

### Clear Database (if needed)
```bash
python src/clear_db.py
```

## 🚀 Advanced Features

### Vector Search Optimization
- **HNSW Indexing**: Hierarchical Navigable Small World indexing for faster vector searches
- **Cosine Similarity**: Efficient similarity calculations for document retrieval
- **Chunking Strategy**: Intelligent text splitting for optimal embedding performance

### Multi-Model Support
- **Gemini Integration**: Google's latest Gemini models via `google-generativeai` library
- **Local LLM Support**: Self-hosted models through Ollama integration
- **Seamless Switching**: Change between models without losing session context

### Session Persistence
- **Database Storage**: All chat sessions and metadata stored in PostgreSQL
- **Document Association**: Each session maintains links to its uploaded documents
- **History Preservation**: Complete conversation history across all sessions

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

Please ensure your code follows the existing style and includes appropriate tests.

## 🆘 Support

If you encounter any problems or have questions:

1. Check the [Issues](https://github.com/alokagarwal565/ez_project/issues) page
2. Create a new issue with detailed information about your setup and the error
3. Include relevant logs and environment details

## 🙏 Acknowledgments

- [Google Gemini](https://deepmind.google/technologies/gemini/) for the powerful language model
- [Ollama](https://ollama.ai/) for local LLM support
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [PGVector](https://github.com/pgvector/pgvector) for efficient vector storage
- [LangChain](https://python.langchain.com/) for the RAG pipeline components
- [HuggingFace](https://huggingface.co/) for the embedding models

---

**Made with ❤️ by [Alok Agarwal]**