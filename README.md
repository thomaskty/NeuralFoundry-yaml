# NeuralFoundry

## Overview
### 1. Multi-user Support
- Each user has a unique ID and profile.
- Users can create, view, and manage multiple chat sessions.

### 2. Chat Sessions
- Each user can create multiple chat sessions.
- Each chat session stores metadata (`title`, `created_at`) and is linked to its creator.

### 3. Message Memory & Context
- Chat messages (user + assistant) are stored in the database.
- Each message is converted into a **vector embedding** using OpenAI embeddings (`text-embedding-3-small`, 1536 dims).

### 4. Contextual Response Generation
- When a user sends a query, the system:
  1. Retrieves **top similar past messages** from that chat session via vector similarity search.
  2. Passes retrieved conversation snippets as context to the LLM.
  3. Generates an assistant response based on the context, maintaining chat continuity.

### 5. Routers & API
- **User Router:** Create users, fetch user info, list user chats.
- **Chat Router:** Create chats (user-linked or generic), send messages, fetch messages, delete chats, list all chats.
- Clear RESTful hierarchy:
  - `/users/{user_id}/chats` → user-specific chats
  - `/chats/{chat_id}` → chat-specific operations

### 6. Database & Storage
- PostgreSQL used as the main DB.
- Vector embeddings stored using `pgvector`.
- `ChatSession` ↔ `User` relationship implemented.
- `ChatMessage` ↔ `ChatSession` relationship implemented.

---

## Why This Project Matters (For Junior AI Engineers)
NeuralFoundry is a hands-on, modular RAG playground that shows how real AI systems are built. The code is intentionally organized so you can study or swap components without rewriting the whole stack.

### Techniques Used
- **Chat Memory**  
  We store each message (user + assistant) as text + vector embeddings. Retrieval pulls recent messages and semantically similar older messages to keep context.

- **Knowledge Bases (KBs)**  
  Each KB has documents stored with metadata. We embed chunks and retrieve the most relevant ones at query time.

- **Chat Attachments**  
  You can attach files directly to a chat. Those files are processed into chunks and stored as embeddings, then used as extra context just for that chat.

- **Chunking Strategy**  
  - PDFs, docx, images, HTML, etc. are processed using **Docling** for structure-aware chunking.  
  - `.txt` and `.md` are split directly with overlap (simple, fast, reliable).

- **Retrieval & Similarity Thresholds**  
  Vector similarity search uses `pgvector`.  
  Similarity thresholds (e.g., KB chunk threshold) are **configurable in settings**, so you can tune relevance vs. recall without changing code.

### Architecture Patterns Worth Learning
- **Modular pipelines** (chat pipeline, KB ingestion, attachment ingestion)  
  Easy to plug in other retrieval strategies: BM25, hybrid search, reranking, etc.
- **Config‑driven behavior**  
  Model selection, embedding dimensions, chunk sizes, and thresholds can be adjusted centrally.
- **Metadata‑rich storage**  
  We store metadata for chats, KBs, and attachments, which makes analytics and dashboards possible later.

### Tooling Stack
- **FastAPI** for backend APIs  
- **OpenAI Python SDK** for chat + embeddings  
- **PostgreSQL + pgvector** for vector search  
- **Python CLI** for YAML-based runs  
- **Docker Compose (optional)** for Postgres + pgAdmin  

---

## Ideas You Can Build Next
- **More Retrieval Methods**  
  Add BM25, hybrid search, reranking, or query expansion.
- **Multi‑model Responses**  
  Generate Response A / Response B from different models (or different prompts) and compare.
- **User Feedback Loop**  
  Collect thumbs‑up/down and feed it into evaluation or reranking.
- **Analytics & Dashboards**  
  Use the stored metadata to show most used KBs, attachment usage, query patterns, etc.
- **Agents / MCPs**  
  Add tools, structured workflows, or multi‑step reasoning with agent frameworks.

---

## Python Run (YAML)
Run the system headless using a YAML config file.

### 1. Set environment variables
Create or edit `/Users/thomaskuttyreji/Documents/NeuralFoundry/.env`:
```env
POSTGRES_USER=neuralfoundry
POSTGRES_PASSWORD=neuralfoundry_pw
POSTGRES_DB=neuralfoundry
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

Set your OpenAI key in the macOS environment (not in `.env`):
```bash
export OPENAI_API_KEY="your_key_here"
```

### 2. Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run from YAML
```bash
python3 run.py run.example.yaml
```

### 4. YAML schema (example)
See `run.example.yaml` for a complete sample.

---

## Optional Docker (Postgres + pgAdmin)
Use Docker only for the database + pgAdmin. The app itself runs with Python.

### 1. Set environment variables
Create or edit `/Users/thomaskuttyreji/Documents/NeuralFoundry/.env`:
```env
POSTGRES_USER=neuralfoundry
POSTGRES_PASSWORD=neuralfoundry_pw
POSTGRES_DB=neuralfoundry
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

### 2. Start Postgres + pgAdmin
```bash
docker compose up -d
```

### 3. pgAdmin connection
- Hostname: `host.docker.internal`
- Port: `5432`
- Username: `neuralfoundry`
- Password: `neuralfoundry_pw`
- Database: `neuralfoundry`

---

## What You’ll See In Logs (Sample)
The backend prints a compact retrieval summary so you can quickly understand what context the model is using:
```
============================================================
🐛 RETRIEVAL RESULTS:
   - Recent messages: 5
   - Older messages: 0
   - KB chunks: 0
   - Attachment chunks: 0
   ⚠️  NO KB RESULTS FOUND!
   ⚠️  No KBs are attached to this chat!
   ℹ️  No attachments found in this chat
============================================================
```

---

## Data Model (Quick Snapshot)
Here’s a minimal view of how data is connected:
```
User
  └── ChatSession
        ├── ChatMessage (vector embedding)
        ├── ChatAttachment
        │     └── ChatAttachmentChunk (vector embedding)
        └── ChatSessionKB (links chat ↔ KB)

KnowledgeBase
  └── KBDocument
        └── KBChunk (vector embedding)
```

## Local Dev (API)
If you still want the REST API:
```bash
cd /Users/thomaskuttyreji/Documents/NeuralFoundry
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```
