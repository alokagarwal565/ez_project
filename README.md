# 📚 QueryDoc AI: Smart Document Assistant

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)](https://www.postgresql.org/)

## 🚀 Overview

**QueryDoc AI** is a Streamlit-based web application designed to help users interact with their documents using advanced Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) techniques. It allows users to upload PDF or TXT documents, ask questions, get summaries, engage in challenge mode, and compare information across multiple documents. The application leverages PostgreSQL with PGVector for efficient document storage and retrieval.

![image](https://github.com/user-attachments/assets/82e0b643-078d-4586-a190-9a3c8f1c17cb)

## 📊 Project Flow Diagram

The diagram below captures the rightward data and interaction flow within **QueryDoc AI**, from user inputs to LLM-based output generation. It outlines key components like Streamlit UI, document processing, vector storage, RAG interaction, and result rendering.

![diagram-export-7-9-2025-12_04_13-AM](https://github.com/user-attachments/assets/1ce34592-7959-4a2f-9bae-02900ae5c06a)

> **Main Flow Includes**:
> - **User Actions**: Upload documents, select LLMs, manage chat, and enter prompts  
> - **Document Processing**: Chunking, embedding, and storing documents in PGVector  
> - **LLM Initialization**: Choose Gemini (cloud) or Ollama (local) models  
> - **RAG Modes**: Ask Anything, Summarize, Challenge, and Compare Documents  
> - **Data Access**: Pulls relevant embeddings from the vector DB for LLM queries  
> - **Results Rendering**: Processed answers and summaries shown via Streamlit  

## 🎥 Demo & Documentation

### 🎥 Video Demo  
[![Watch the Demo](https://img.shields.io/badge/Watch%20Demo-Loom-FF6B6B?style=for-the-badge&logo=loom)](https://www.loom.com/share/3eedd9ec6ec24900a80bd3816840b25d?sid=524c0bc7-e69c-4415-9727-92913c0eeb3e)

### 📄 Project Report  
[![View Report](https://img.shields.io/badge/View%20Report-Google%20Docs-4285F4?style=for-the-badge&logo=googledocs&logoColor=white)](https://docs.google.com/document/d/1mBcF8HrJaba2aH0Db8vorfT5cFaCkrl0QnLRIWUB62c/edit?usp=sharing)

## ✨ Features

### 📄 Document Upload & Processing
- **Easy Upload**: Upload PDF or TXT documents through the intuitive interface
- **Intelligent Processing**: Documents are automatically split into chunks and embedded into a vector database
- **Multi-Document Support**: Handle multiple documents for comprehensive research

### 💬 Persistent Chat Sessions
- **Session Management**: Create new chat sessions or load existing ones
- **Complete History**: All interactions are saved with chat history and associated documents
- **Session Operations**: Rename, delete, and organize your chat sessions

### 🧠 Flexible LLM Integration
- **Gemini Models**: Integrate with Google's Gemini models (e.g., `gemini-1.5-flash-latest`) via API key
- **Local LLM Support**: Connect to local LLMs served via Ollama (e.g., `llama3.2:1b`) for enhanced privacy and control
- **Easy Switching**: Seamlessly switch between different LLM providers

### 🎯 Interactive Modes

#### Ask Anything (RAG)
- Ask questions about your uploaded documents
- Get accurate, context-aware answers with direct source citations
- Leverages retrieval-augmented generation for precise responses

![image](https://github.com/user-attachments/assets/7fb18304-868a-4246-8101-19f271c57bd2)

#### Auto Summary
- Generate concise summaries of your entire document
- Summarize individual chunks for focused insights
- Quick overview of key document content

![image](https://github.com/user-attachments/assets/5d93bd75-d0e9-48e4-bf15-ffbbb9d65985)

#### Challenge Mode
- Test your understanding with AI-generated logic-based questions
- Questions are based on your document content
- Get evaluated on accuracy, completeness, and clarity with detailed feedback

![image](https://github.com/user-attachments/assets/cb6561f7-b7de-4f86-b474-327bc80e98b4)

#### Document Comparison
- Compare concepts or findings across multiple uploaded documents
- Identify commonalities, differences, and contradictions
- Cross-reference information from different sources

![image](https://github.com/user-attachments/assets/38ecb670-a1fa-4148-b505-ade32bbd0bea)

### 🔧 Technical Features
- **PostgreSQL with PGVector**: Efficient vector database storage and similarity search
- **Configurable**: Easy switching between LLMs, embedding models, and database connections
- **Comprehensive Logging**: Detailed insights into application operations for debugging
- **Environment Variables**: Secure configuration management

## 🛠️ Technologies Used

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Backend** | Python 3.9+ |
| **RAG Pipeline** | LangChain |
| **Embeddings** | HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Vector Database** | PostgreSQL with PGVector |
| **Database Adapter** | psycopg2 |
| **Environment Management** | python-dotenv |
| **Cloud LLM** | Google Generative AI (Gemini) |
| **Local LLM** | ChatOllama (LangChain Community) |
| **Document Processing** | PyPDF for PDF handling |

## 📋 Prerequisites

Before setting up the application, ensure you have:

- **Python 3.9+**
- **PostgreSQL** (with vector extension support)
- **Google API Key** (optional, for Gemini integration)
- **Ollama** (optional, for local LLM support)

## 🔧 Setup and Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/alokagarwal565/ez_project.git
cd ez_project
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows
.\venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: The `requirements.txt` typically contains: `streamlit`, `langchain`, `langchain-community`, `langchain-huggingface`, `psycopg2-binary`, `python-dotenv`, `google-generativeai`, `pypdf`

### Step 4: Configure Environment Variables

Create a `.env` file in the project's root directory (next to the `src` folder):

```env
# Google API Key (for Gemini) - Optional if only using local LLM
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"

# PostgreSQL Connection Details
PG_USER="your_pg_user"
PG_PASSWORD="your_pg_password"
PG_HOST="localhost"
PG_PORT="5432"
PG_DBNAME="your_database_name"

# Ollama Settings (for Local LLM) - Optional if only using Gemini
OLLAMA_BASE_URL="http://localhost:11434"  # Or your Ollama server address
LOCAL_LLM_MODEL_NAME="llama3.2:1b"       # Or your preferred local Ollama model
```

> **Important**: 
> - Replace placeholders with your actual credentials
> - Ensure your PostgreSQL database specified in `PG_DBNAME` exists
> - If using Ollama, ensure it's running and the specified model is pulled

### Step 5: Initialize the Database

Run the initialization script to create necessary tables and the vector extension:

```bash
python src/init_db.py
```

This script will create:
- `langchain_pg_collection` and `langchain_pg_embedding` tables
- Chat session related tables
- Vector extension in PostgreSQL

### Step 6: Set Up Ollama (Optional)

If you plan to use local LLM via Ollama:

1. **Download and Install Ollama**: Follow instructions on the [Ollama website](https://ollama.ai)

2. **Pull a Model**: 
   ```bash
   ollama pull llama3.2:1b
   ```

3. **Update Environment**: Set `LOCAL_LLM_MODEL_NAME="llama3.2:1b"` in your `.env` file

## 🚀 Usage

### Start the Application

```bash
streamlit run src/main.py
```

This will open the application in your web browser.

### Using the Application

#### 1. Choose Your LLM
- Select "Gemini" or "Local LLM (Ollama)" from the sidebar
- Switch between models as needed

#### 2. Upload Documents
- Use the "Upload Document" section to add PDF or TXT files
- Click "Build Knowledge Base" to process documents and store them in the vector database

#### 3. Chat Sessions
- **New Chat**: Click "New Chat" button to create a fresh session
- **Load Chat**: Use the dropdown to resume previous conversations with associated documents
- **Manage Sessions**: Rename or delete chat sessions as needed

#### 4. Select Interaction Mode
Choose from the available modes:
- **Ask Anything**: Query your documents with natural language
- **Summarize Document**: Get concise document summaries
- **Challenge Mode**: Test your understanding with AI-generated questions
- **Compare Documents**: Analyze multiple documents for insights

## 🗂️ Project Structure

```
.
├── .env                      # Environment variables (create this file)
├── requirements.txt          # Python dependencies
├── src/
│   ├── main.py              # Main Streamlit application
│   ├── config.py            # Configuration settings
│   ├── rag_pipeline.py      # RAG pipeline logic and document processing
│   ├── challenge_mode.py    # Challenge mode functionality
│   ├── init_db.py           # Database initialization script
│   ├── clear_db.py          # Database clearing utility (use with caution)
│   ├── helpers.py           # Utility functions
│   └── prompt_templates.py  # LLM prompt templates
└── data/                     # Document storage (created automatically)
```

## ⚠️ Important Notes

### Database Management
- **Clear Database**: Use `python src/clear_db.py` with extreme caution as it will delete all stored data
- **Backup**: Regular backups of your PostgreSQL database are recommended

### Development Status
This application is under active development. Features may be added or modified. Refer to the specific code files for the most up-to-date implementation details.

## 🔧 Advanced Configuration

### PostgreSQL Setup
Ensure the vector extension is enabled in your PostgreSQL instance:

```sql
CREATE EXTENSION vector;
```

If you don't have privileges to create extensions, contact your database administrator.

### Logging
The application includes comprehensive logging for debugging and monitoring application operations.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 🆘 Support

If you encounter issues:
1. Check the logs for detailed error information
2. Ensure all prerequisites are properly installed
3. Verify your environment variables are correctly configured
4. Consult the demo video and project report for guidance

## 🙏 Acknowledgments

- [Google Gemini](https://deepmind.google/technologies/gemini/) for powerful language models
- [Ollama](https://ollama.ai/) for local LLM support
- [Streamlit](https://streamlit.io/) for the web application framework
- [LangChain](https://python.langchain.com/) for RAG pipeline components
- [PGVector](https://github.com/pgvector/pgvector) for vector database capabilities
- [HuggingFace](https://huggingface.co/) for embedding models

---

**QueryDoc AI: Making Document Intelligence Accessible**

---

**Made with ❤️ by [Alok Agarwal](https://github.com/alokagarwal565)**
