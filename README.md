#  Agentic RAG

A production-ready Agentic RAG (Retrieval-Augmented Generation) system built with LangChain, LangGraph, ChromaDB, and Gemini. The agent intelligently selects the right tool based on your question.

---

## 🏗️ Architecture
```
document.pdf
     ↓ Unstructured
  text extraction
     ↓ RecursiveCharacterTextSplitter
    chunks
     ↓ Gemini Embeddings
   vectors
     ↓ ChromaDB
   vector store
     ↓ LangGraph Agent
   smart routing
     ↓ Gemini LLM
   final answer 
```

## 🤖 Agent Flow
```
Question
   ↓
retrieve (ChromaDB)
   ↓
grade_documents
   ↓ relevant          ↓ not relevant
generate           rewrite_query
   ↓                   ↓
answer           retrieve again
                   ↓ still not relevant
                web_search (DuckDuckGo)
                   ↓
                generate
                   ↓
                answer
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Orchestration | LangChain |
| Agent / Graph | LangGraph |
| Vector Store | ChromaDB |
| LLM | Gemini (Google AI) |
| Embeddings | Gemini Embedding |
| Document Parsing | Unstructured |
| Web Search | DuckDuckGo (ddgs) |
| UI | Streamlit |

---

## 📁 Project Structure
```
rag_project/
│
├── data/                  # PDF documents
│   └── document.pdf
│
├── vectorstore/           # ChromaDB persistent storage
│
├── agentic/               # Agentic RAG core
│   ├── __init__.py
│   ├── state.py           # LangGraph state
│   ├── nodes.py           # Agent nodes
│   ├── graph.py           # LangGraph graph
│   ├── tools.py           # Multi-tool definitions
│   ├── multi_agent.py     # Multi-tool agent
│   ├── run.py             # Terminal interface
│   └── run_multi.py       # Multi-tool terminal interface
│
├── app.py                 # Streamlit web interface
├── main.py                # Simple RAG pipeline
├── evaluate.py            # RAG evaluation (LLM-as-judge)
├── .env                   # API keys (not committed)
├── requirements.txt       # Dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/gharbyryhab/rag-project
cd rag_project
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API key

Create a `.env` file at the root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your free API key at [aistudio.google.com](https://aistudio.google.com)

### 5. Add your document

Place your PDF in the `data/` folder:
```
data/document.pdf
```

### 6. Ingest the document
```bash
python main.py
```

---

## 💻 Usage

### Web Interface (Streamlit)
```bash
streamlit run app.py
```

### Terminal — Simple RAG
```bash
python -m agentic.run
```

### Terminal — Multi-Tool Agent
```bash
python -m agentic.run_multi
```

### Evaluate RAG quality
```bash
python evaluate.py
```

---

## 🧠 Features

### Simple RAG
- PDF extraction with Unstructured
- Semantic chunking with overlap
- Vector similarity search
- Source + page number citations

### Agentic RAG (LangGraph)
- Automatic document relevance grading
- Query rewriting when needed
- Web search fallback (DuckDuckGo)
- Conversational memory

### Multi-Tool Agent
| Tool | Trigger |
|---|---|
| 📄 `search_document` | Questions about the document |
| 🌐 `search_web` | General knowledge questions |
| 🧮 `calculate` | Math and performance comparisons |
| 📝 `summarize` | Summarization requests |

### RAG Evaluation
- LLM-as-judge scoring
- 3 metrics: context relevance, faithfulness, completeness
- Scored 4.9/5 on test suite

---

## 📊 Evaluation Results

| Metric | Score |
|---|---|
| Context Relevance | 5.0 / 5 |
| Faithfulness | 5.0 / 5 |
| Completeness | 4.8 / 5 |
| **Global Score** | **4.9 / 5** |

---

## ⚠️ API Limits (Free Tier)

| Model | Limit |
|---|---|
| Gemini 2.5 Flash | 20 req/day |
| Gemini Embedding | 100 req/min · 1000 req/day |

Each question consumes 2–4 API calls depending on the routing path.

---

## 🗺️ Roadmap

- [x] Simple RAG pipeline
- [x] Agentic RAG with LangGraph
- [x] Web search fallback
- [x] Conversational memory
- [x] Multi-tool agent
- [x] Streamlit UI
- [ ] Multi-document support
- [ ] Authentication
- [ ] Docker deployment
