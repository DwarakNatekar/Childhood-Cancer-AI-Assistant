<img width="1892" height="916" alt="vlcsnap-2026-07-05-21h16m36s253" src="https://github.com/user-attachments/assets/89e7c612-60ad-4fcd-84b6-843f57df24eb" />


https://github.com/user-attachments/assets/06192c03-ddfb-40e1-8b44-c3339cbbd308


Perfect. Here's a professional README that suits your project.

---

# 🩺 Childhood Cancer AI Assistant

## Project Overview

The Childhood Cancer AI Assistant is a Retrieval-Augmented Generation (RAG) application designed to answer questions using official WHO Childhood Cancer publications. Instead of relying solely on the Large Language Model's pre-trained knowledge, the application retrieves relevant information from trusted WHO documents stored in a vector database and generates grounded responses with document citations and page references.

The goal of the project is to demonstrate an end-to-end GenAI pipeline involving document ingestion, semantic search, vector databases, LLM integration, and public deployment.

---

# Features

* AI-powered question answering using Retrieval-Augmented Generation (RAG)
* Semantic search across WHO Childhood Cancer publications
* Grounded responses using retrieved document context
* Source citation with document name and page number
* Streaming responses for improved user experience
* Interactive Streamlit web interface
* Public cloud deployment
* Local vector database using ChromaDB
* Open-source LLM integration through Groq

---

# Tech Stack

| Category             | Technology                        |
| -------------------- | --------------------------------- |
| Programming Language | Python                            |
| LLM                  | Groq (Llama 3.3 70B Versatile)    |
| Framework            | LangChain                         |
| Vector Database      | ChromaDB                          |
| Embeddings           | HuggingFace Sentence Transformers |
| UI                   | Streamlit                         |
| Document Processing  | Unstructured                      |
| OCR                  | Tesseract                         |
| PDF Processing       | Poppler                           |
| NLP                  | NLTK                              |
| Version Control      | Git & GitHub                      |

---

# Architecture

```text
WHO PDF Documents
        │
        ▼
Document Loader (Unstructured)
        │
        ▼
Text Chunking
(CharacterTextSplitter)
        │
        ▼
HuggingFace Embeddings
        │
        ▼
Chroma Vector Database
        │
        ▼
Retriever
        │
        ▼
LangChain RetrievalQA
        │
        ▼
Groq Llama 3.3 70B
        │
        ▼
Streamlit User Interface
        │
        ▼
Grounded Response + Source Citation
```

---

# Installation

### Clone Repository

```bash
git clone <repository-url>
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```text
GROQ_API_KEY=your_api_key
```

### Generate Vector Database

```bash
python langchain_rag_ingestion.py
```

### Run Application

```bash
streamlit run chat_bot.py
```

---

# Demo

The application allows users to ask natural language questions related to childhood cancer.

Example Questions

* What are the survival rates for childhood cancer?
* What is the Global Initiative for Childhood Cancer?
* What are the common warning signs?
* How can countries improve childhood cancer outcomes?

Each response includes:

* AI-generated answer
* Source document
* Page number


# Future Improvements

### Multi-document Upload
Allow users to upload their own PDF documents directly through the interface. The application would automatically process, embed, and index them without rebuilding the entire vector database.

### Hybrid Search
Combine semantic vector search with traditional keyword search (BM25) to improve retrieval accuracy for technical terms, abbreviations, and exact matches.

### Reranking
Use a Cross Encoder reranker after retrieval to reorder candidate documents, ensuring the most relevant context is sent to the LLM.

### Conversation Memory
Maintain previous conversation history so users can ask follow-up questions without repeating earlier context.

### PDF Highlighting
Open the cited PDF directly at the referenced page and highlight the exact paragraph used to generate the answer.

### Citation Hyperlinks
Convert citations into clickable links that navigate users directly to the relevant page within the source document.

### Agentic Workflows
Extend the application using AI agents capable of planning, validating retrieved information, searching multiple document collections, and performing multi-step reasoning before generating a response.

### DSPy Optimization
Optimize prompts and retrieval strategies automatically using DSPy instead of manually engineering prompts, improving both response quality and token efficiency.

### Evaluation Framework
Measure response quality using automated metrics such as faithfulness, answer relevance, context precision, and hallucination detection with frameworks like Ragas.

### Enterprise Deployment
Support authentication, user management, REST APIs, Docker containers, monitoring, logging, and cloud-native deployment for production environments.




