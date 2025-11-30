# 🎓 Virtual TA: RAG-Powered Knowledge Retrieval System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![SQLite](https://img.shields.io/badge/SQLite-07405e?style=for-the-badge&logo=sqlite)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai)

**An intelligent Retrieval-Augmented Generation (RAG) system that transforms course materials and forum discussions into an AI-powered Virtual Teaching Assistant**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [API Documentation](#-api-documentation)

</div>

---

##  Overview

This project implements a production-ready **Retrieval-Augmented Generation (RAG) system** designed to answer student queries by intelligently retrieving relevant context from course materials and Discourse forum discussions. The system combines modern web scraping, semantic search, and large language models to provide accurate, contextual answers with proper source attribution.

###  Key Capabilities

- **Intelligent Data Ingestion**: Automated scraping of course documentation and forum discussions
- **Semantic Search**: Vector-based similarity search using OpenAI embeddings
- **Context-Aware Responses**: LLM-powered answer generation with source citations
- **Multimodal Support**: Handle both text and image-based queries
- **Production-Ready API**: FastAPI-based REST API with proper error handling and logging
- **Scalable Architecture**: Modular design supporting large knowledge bases

---

##  Features

###  Data Collection & Processing

- **Course Material Scraping**
  - Automated crawling of Docsify-based course documentation
  - HTML-to-Markdown conversion for clean text extraction
  - Metadata preservation (URLs, timestamps, document hierarchy)

- **Discourse Forum Scraping**
  - Authenticated session management using Playwright
  - Date-range filtered topic extraction
  - Automatic HTML cleaning and text normalization

- **Intelligent Text Chunking**
  - Context-aware chunking with configurable sizes (default: 1000 chars)
  - Overlap preservation (default: 200 chars) for semantic continuity
  - Sentence and paragraph boundary detection

###  Retrieval Engine

- **Semantic Search**
  - OpenAI `text-embedding-3-small` for high-quality embeddings
  - Cosine similarity-based relevance scoring
  - Configurable similarity thresholds (default: 0.50)

- **Context Enrichment**
  - Adjacent chunk retrieval for fuller context
  - Multi-source aggregation (course docs + forum posts)
  - Relevance-based ranking and deduplication

- **Multimodal Query Processing**
  - GPT-4o Vision integration for image understanding
  - Combined text-image context generation

###  API & Infrastructure

- **FastAPI REST API**
  - `/query` endpoint for question answering
  - `/health` endpoint with database statistics
  - CORS-enabled for frontend integration

- **Robust Error Handling**
  - Exponential backoff for rate limit handling
  - Graceful degradation on API failures
  - Comprehensive logging throughout

- **Performance Optimizations**
  - SQLite database for efficient storage and retrieval
  - Batch processing for embeddings generation
  - Connection pooling and query optimization

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                    │
├──────────────────────┬──────────────────────────────────────┤
│  scrape_course.py    │   scrape_discourse.py                │
│  (Playwright)        │   (Playwright + Auth)                │
│  ├─ Docsify Crawler  │   ├─ Forum Scraper                   │
│  ├─ HTML→Markdown    │   ├─ JSON Export                     │
│  └─ Metadata Extract │   └─ HTML Cleaning                   │
└──────────────────────┴──────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Preprocessing Pipeline                    │
│                      (preprocess.py)                        │
├─────────────────────────────────────────────────────────────┤
│  1. Text Chunking (overlapping, context-aware)              │
│  2. Embedding Generation (OpenAI text-embedding-3-small)    │
│  3. SQLite Storage (discourse_chunks + markdown_chunks)     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Retrieval & Query API                    │
│                         (app.py)                            │
├─────────────────────────────────────────────────────────────┤
│  Query Processing                                           │
│    ├─ Text/Image Embedding Generation                       │
│    ├─ Cosine Similarity Search                              │
│    ├─ Context Enrichment (adjacent chunks)                  │
│    └─ Result Ranking & Deduplication                        │
│                                                             │
│  Answer Generation                                          │
│    ├─ GPT-4o-mini for Response Generation                   │
│    ├─ Source Attribution & Citation                         │
│    └─ Structured JSON Response                              │
└─────────────────────────────────────────────────────────────┘
```

###  Database Schema

**discourse_chunks**
```sql
- id (PRIMARY KEY)
- post_id, topic_id, topic_title
- post_number, author, created_at, likes
- chunk_index, content, url
- embedding (BLOB - JSON serialized vector)
```

**markdown_chunks**
```sql
- id (PRIMARY KEY)
- doc_title, original_url, downloaded_at
- chunk_index, content
- embedding (BLOB - JSON serialized vector)
```

---

##  Installation

### Prerequisites

- **Python 3.11+**
- **pip** (Python package manager)
- **Git**

### Step 1: Clone the Repository

```bash
git clone https://github.com/amnesiac-log/Virtual-TA.git
cd virtual-ta-rag-system
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

### Step 4: Install Playwright Browsers

```bash
playwright install chromium
```

### Step 5: Environment Configuration

Create a `.env` file in the project root:

```env
API_KEY=your_aipipe_api_key_here
```

> **Note**: You can obtain an API key from [aipipe.org](https://aipipe.org) which provides access to OpenAI models.

---

##  Usage

### 1️⃣ Scrape Course Materials

```bash
python scrape_course.py
```

**Output**: Markdown files in `markdown_files/` directory

**Configuration** (in script):
- `BASE_URL`: Course documentation URL
- `OUTPUT_DIR`: Destination for scraped content

### 2️⃣ Scrape Discourse Forum

```bash
python scrape_discourse.py
```

**Process**:
1. First run opens browser for manual Google login
2. Session is saved in `auth.json`
3. Subsequent runs use saved authentication
4. Downloaded threads saved to `downloaded_threads/`

**Configuration** (in script):
- `CATEGORY_ID`: Discourse category to scrape
- `DATE_FROM`, `DATE_TO`: Date range filter

### 3️⃣ Preprocess and Generate Embeddings

```bash
python preprocess.py
```

**Options**:
```bash
python preprocess.py --chunk-size 1000 --chunk-overlap 200 --api-key YOUR_API_KEY
```

**Process**:
- Reads markdown and discourse files
- Chunks content intelligently
- Generates embeddings via OpenAI API
- Stores in `knowledge_base.db`

### 4️⃣ Start the API Server

```bash
python app.py
```

Or with uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

**Server will be available at**: `http://localhost:5000`

---

##  API Documentation

### **POST /query**

Submit a question and receive an AI-generated answer with sources.

**Request Body**:
```json
{
  "question": "What is the difference between supervised and unsupervised learning?",
  "image": null  // Optional: Base64-encoded image
}
```

**Response**:
```json
{
  "answer": "Supervised learning uses labeled training data where the correct output is known, while unsupervised learning works with unlabeled data to find patterns and structures...",
  "links": [
    {
      "url": "https://docs.onlinedegree.iitm.ac.in/machine-learning",
      "text": "Machine learning overview and types"
    },
    {
      "url": "https://discourse.onlinedegree.iitm.ac.in/t/ml-basics/12345/1",
      "text": "Discussion on supervised vs unsupervised learning"
    }
  ]
}
```

**With Image Query**:
```python
import base64
import requests

# Read and encode image
with open("diagram.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:5000/query", json={
    "question": "Explain this diagram",
    "image": image_base64
})
```

### **GET /health**

Check system health and database statistics.

**Response**:
```json
{
  "status": "healthy",
  "database": "connected",
  "api_key_set": true,
  "discourse_chunks": 1523,
  "markdown_chunks": 847,
  "discourse_embeddings": 1523,
  "markdown_embeddings": 847
}
```

---

##  Configuration

### Retrieval Parameters (app.py)

```python
SIMILARITY_THRESHOLD = 0.50    # Minimum cosine similarity for relevance
MAX_RESULTS = 10               # Maximum number of chunks to retrieve
MAX_CONTEXT_CHUNKS = 4         # Chunks per source document
```

### Chunking Parameters (preprocess.py)

```python
CHUNK_SIZE = 1000              # Characters per chunk
CHUNK_OVERLAP = 200            # Overlap between chunks
```

### LLM Configuration (app.py)

```python
# Embedding Model
model: "text-embedding-3-small"

# Answer Generation Model
model: "gpt-4o-mini"
temperature: 0.3               # Lower = more deterministic
```

---

##  Example Queries

### Text-Only Query

```python
import requests

response = requests.post("http://localhost:5000/query", json={
    "question": "How do I handle missing data in pandas?"
})

print(response.json()["answer"])
```

### Multimodal Query

```python
import base64
import requests

with open("code_screenshot.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:5000/query", json={
    "question": "What's wrong with this code?",
    "image": img_b64
})

result = response.json()
print(f"Answer: {result['answer']}\n")
print("Sources:")
for link in result['links']:
    print(f"  - {link['text']}: {link['url']}")
```

---

##  Performance Considerations

### Embedding Generation
- **Time**: ~1-2 seconds per chunk (API dependent)
- **Cost**: Text-embedding-3-small is cost-effective
- **Optimization**: Batch processing with exponential backoff

### Query Performance
- **Latency**: 2-4 seconds end-to-end
  - Embedding: ~500ms
  - Database search: ~100-200ms
  - LLM generation: 1-3s
- **Concurrent Requests**: FastAPI handles async operations efficiently

### Storage
- **Embeddings**: ~6KB per chunk (1536 dimensions × 4 bytes)
- **Database Size**: ~50MB per 1000 chunks (including text)

---

##  Error Handling

The system implements robust error handling:

- **Rate Limiting**: Exponential backoff with configurable retries
- **API Failures**: Graceful degradation with informative error messages
- **Database Errors**: Connection pooling and automatic retry
- **Malformed Input**: Pydantic validation with clear error responses

### Logging

All components use Python's logging module:

```python
# View logs
tail -f app.log

# Adjust log level (in code)
logging.basicConfig(level=logging.INFO)
```

---

##  Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | FastAPI | REST API server |
| **Web Scraping** | Playwright | Dynamic content scraping |
| **HTML Processing** | BeautifulSoup4 | HTML parsing and cleaning |
| **Text Conversion** | html2text, markdownify | Format conversion |
| **Database** | SQLite | Chunk and embedding storage |
| **Embeddings** | OpenAI text-embedding-3-small | Semantic vector generation |
| **LLM** | GPT-4o-mini | Answer generation |
| **Vision** | GPT-4o Vision | Image understanding |
| **HTTP Client** | aiohttp | Async API calls |
| **Data Validation** | Pydantic | Request/response schemas |
| **Numerical Computing** | NumPy | Vector operations |
| **Environment** | python-dotenv | Configuration management |

---

##  Project Structure

```
virtual-ta-rag-system/
├── app.py                      # FastAPI application
├── scrape_course.py            # Course material scraper
├── scrape_discourse.py         # Discourse forum scraper
├── preprocess.py               # Preprocessing pipeline
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
├── knowledge_base.db           # SQLite database (generated)
├── auth.json                   # Discourse auth state (generated)
├── markdown_files/             # Scraped course content (generated)
├── downloaded_threads/         # Scraped forum posts (generated)
└── README.md                   # This file
```

---

##  Use Cases

- **Virtual Teaching Assistant**: Automated Q&A for online courses
- **Documentation Search**: Semantic search over technical documentation
- **Forum Knowledge Base**: Extracting insights from community discussions
- **Student Support**: 24/7 instant answers to common questions
- **Course Analytics**: Understanding common pain points from forum data

---

##  Future Enhancements

- [ ] **Vector Database**: Migrate to Pinecone/Weaviate for scale
- [ ] **Hybrid Search**: Combine semantic + keyword search
- [ ] **Query Expansion**: Automatic query reformulation
- [ ] **Caching Layer**: Redis for frequently asked questions
- [ ] **User Feedback Loop**: Rating system for answer quality
- [ ] **Advanced Chunking**: Implement semantic chunking with LangChain
- [ ] **Multi-language Support**: Handle non-English content
- [ ] **Real-time Updates**: Incremental scraping and indexing

---


</div>
