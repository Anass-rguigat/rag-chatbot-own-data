# RAG Chatbot with Memory

A retrieval-augmented generation (RAG) chatbot that answers only from your own documents. You provide a single knowledge-base folder (PDF, TXT, MD, DOCX); the app indexes it and answers questions with source citations and conversation memory. Two modes: **FAST** (for weak laptops) and **COMPLETE** (for paid API or strong servers).

---

## Architecture

High-level flow:

```
[Documents: PDF, TXT, MD, DOCX]
         |
         v
[Load & chunk] --> RecursiveCharacterTextSplitter (size/overlap depend on FAST vs COMPLETE)
         |
         v
[Embed] --> HuggingFaceEmbeddings (sentence-transformers)
         |
         v
[Index] --> FAISS vector store (cached on disk per mode: .vectorstore / .vectorstore_fast)
         |
         v
[User question] --> [Retriever: similarity (FAST) or MMR (COMPLETE)] --> top-k chunks
         |
         v
[Prompt: context + question] --> LLM (Groq API or Ollama local)
         |
         v
[Stream or invoke] --> Answer + source documents
         |
         v
[ConversationBufferMemory] --> next turn can use chat history
```

- **FAST mode:** Smaller chunks (800/120), fewer docs retrieved (k=5), similarity search only, shorter LLM output (e.g. 512 tokens Groq, 256 Ollama), 60s timeout. Optimized for weak CPUs.
- **COMPLETE mode:** Larger chunks (2000/450), more docs (k=15, fetch_k=40), MMR retrieval, longer output (8192 / 2048), 90s timeout. Optimized for quality.

Groq 429 (rate limit) is handled by retrying with exponential backoff (2s, 5s, 10s); if it still fails, the app can fall back to Ollama for that request.

---

## Code Structure

| Component | Location | Role |
|-----------|----------|------|
| Config & constants | Top of `app.py` | Env vars, FAST/COMPLETE params, prompt templates |
| Cache helpers | `_data_folder_signature`, `_vectorstore_manifest_path`, `_try_load_cached_vectorstore`, `_save_vectorstore_cache` | Decide when to load/save FAISS from disk |
| Document loading | `_load_one_document`, `load_and_process_documents` | Load PDF/TXT/MD/DOCX and split into chunks |
| Vector store | `get_embeddings`, `get_vectorstore(fast_mode)` | Embeddings model and FAISS index (cached per mode) |
| LLM & chain | `get_llm(...)`, `get_chain(use_ollama, fast_mode)` | Groq or Ollama LLM; retriever + prompt + memory = ConversationalRetrievalChain |
| Invoke & retry | `invoke_chain_with_timeout_and_retry`, `GroqRateLimitError` | Run chain with timeout; Groq 429 retry then optional Ollama fallback |
| Streaming | `_run_chain_streaming` | Retrieve, build prompt, stream LLM tokens into UI, update memory |
| UI | Bottom of `app.py` | Streamlit: sidebar (Mode, Model), chat loop, source expanders |

---

## Tools and Their Roles

| Tool / Library | Role |
|----------------|------|
| **Streamlit** | Web UI: chat interface, sidebar (Mode: FAST/COMPLETE, Model: Ollama/Groq), source expanders, streaming placeholder. |
| **LangChain** | Orchestration: `ConversationalRetrievalChain`, retriever, prompts, memory (`ConversationBufferMemory`, `ChatMessageHistory`). |
| **LangChain-Groq** | `ChatGroq`: calls Groq API for the LLM. Used in COMPLETE (e.g. 70b) or FAST (8b). Requires `GROQ_API_KEY`. |
| **LangChain + Ollama** | `ChatOllama`: runs LLM locally (e.g. tinyllama, phi3:mini). No API key; used when Groq hits 429 or user selects Local. |
| **Hugging Face (sentence-transformers)** | `HuggingFaceEmbeddings`: turns text into vectors. Used to embed chunks and queries; model `all-MiniLM-L6-v2` by default. |
| **FAISS** | Vector store for chunk embeddings. Similarity or MMR search returns top-k chunks for the prompt. Cached under `.vectorstore` (COMPLETE) and `.vectorstore_fast` (FAST). |
| **PyPDF / PyPDFLoader** | Load and extract text from PDFs; fallback for missing pages via `PdfReader`. |
| **python-docx** | Load DOCX files (paragraphs and tables) into text for the knowledge base. |
| **RecursiveCharacterTextSplitter** | Splits documents into chunks (size/overlap depend on mode) with separators like `\n\n`, `. ` to avoid cutting sentences. |
| **python-dotenv** | Loads `.env` (API keys, paths, model names) so secrets are not hardcoded. |

---

## How to Run from Scratch

### 1. Prerequisites

- Python 3.10+
- (Optional) [Ollama](https://ollama.com) and a model (e.g. `ollama pull phi3:mini` for FAST, `ollama pull tinyllama` for COMPLETE) if you want the local LLM.

### 2. Clone or download the project

```bash
cd path/to/RAG-Chatbot-with-Memory-
```

### 3. Virtual environment (recommended)

```bash
python -m venv venv
```

Activate:

- Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
- Windows (CMD): `venv\Scripts\activate.bat`
- macOS/Linux: `source venv/bin/activate`

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

The first run may download the embedding model (~90MB).

### 5. Environment configuration

Copy the example env file and set at least the Groq API key:

```bash
copy .env.example .env
```

Edit `.env`:

- **`GROQ_API_KEY`** (required for Groq): get a key at [Groq Console](https://console.groq.com/).
- Optional: `DATA_FOLDER_PATH` or `PDF_FOLDER_PATH` (default: `./fin_ed_docs`), `GROQ_MODEL`, `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, etc. See `.env.example` for all options.

### 6. Add documents

Put at least one supported file (PDF, TXT, MD, DOCX) in the knowledge-base folder (default: `fin_ed_docs/`).

### 7. Run the application

```bash
python -m streamlit run app.py
```

Use `python -m streamlit` if the `streamlit` command is not on your PATH. The app opens at `http://localhost:8501`.

### 8. Use the app

- In the sidebar: choose **Mode** (FAST for laptop, COMPLETE for full quality) and **Model** (Local Ollama or API Groq).
- Type a question; the answer is generated from your documents only, with sources under each reply.
- Follow-up questions use conversation memory.

---

## Configuration Summary

Every setting is read from `.env`; the app uses the defaults below only when a variable is missing. No hardcoded overrides: `.env` is the single source of truth.

| Variable | Mode | Purpose |
|----------|------|---------|
| `GROQ_API_KEY` | both | Groq API key (required when using Groq). |
| `DATA_FOLDER_PATH`, `PDF_FOLDER_PATH` | both | Knowledge-base folder. |
| `TEMPERATURE`, `EMBEDDING_MODEL`, `EMBEDDING_DEVICE` | both | LLM and embedding options. |
| `OLLAMA_BASE_URL`, `USE_OLLAMA` | both | Ollama server and default sidebar choice. |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVER_K`, `RETRIEVER_FETCH_K` | COMPLETE | RAG parameters when Mode = COMPLETE. |
| `GROQ_MODEL`, `GROQ_MAX_TOKENS`, `OLLAMA_MODEL`, `OLLAMA_NUM_PREDICT`, `LLM_TIMEOUT_SEC` | COMPLETE | LLM and timeout when Mode = COMPLETE. |
| `VECTORSTORE_CACHE_PATH` | COMPLETE | Cache directory for COMPLETE index. |
| `CHUNK_SIZE_FAST`, `CHUNK_OVERLAP_FAST`, `RETRIEVER_K_FAST`, `RETRIEVER_FETCH_K_FAST` | FAST | RAG parameters when Mode = FAST. |
| `GROQ_MODEL_FAST`, `GROQ_MAX_TOKENS_FAST`, `OLLAMA_MODEL_FAST`, `OLLAMA_NUM_PREDICT_FAST`, `LLM_TIMEOUT_FAST` | FAST | LLM and timeout when Mode = FAST. |
| `VECTORSTORE_CACHE_PATH_FAST` | FAST | Cache directory for FAST index. |

See `.env.example` for all variables and default values.

---

## Troubleshooting

- **"streamlit" is not recognized** – Run `python -m streamlit run app.py` and ensure the venv is activated.
- **No supported documents found** – Add at least one PDF, TXT, MD, or DOCX in the knowledge-base folder.
- **Slow startup** – First run builds the vector store; later runs load from `.vectorstore` or `.vectorstore_fast`. Change docs or chunk settings to force rebuild.
- **Slow or stuck answers** – Use **FAST** mode and/or **Local (Ollama)** with a small model (e.g. `phi3:mini`). For Ollama, run `ollama pull phi3:mini` (FAST) or `ollama pull tinyllama` (COMPLETE).
- **Rate limit (429) on Groq** – The app retries with backoff and can fall back to Ollama. You can also switch to **Local (Ollama)** in the sidebar or use FAST mode (smaller model).
- **Ollama not responding** – Install Ollama, start it, run `ollama pull <model>`, and set `OLLAMA_BASE_URL` in `.env` if needed.

---

## Project Layout

| Path | Purpose |
|------|---------|
| `app.py` | Main application: config, RAG pipeline, Streamlit UI. |
| `requirements.txt` | Python dependencies. |
| `.env.example` | Example environment variables; copy to `.env`. |
| `.streamlit/config.toml` | Streamlit client config (e.g. toolbar). |
| `fin_ed_docs/` | Default knowledge-base folder. |
| `.vectorstore/` | Cached FAISS index for COMPLETE mode. |
| `.vectorstore_fast/` | Cached FAISS index for FAST mode. |

---

## Verification (completeness and quality)

This application is implemented to the standards expected of a production-ready, enterprise-style "chatbot on your own data":

| Criterion | Status |
|-----------|--------|
| **Own-data only** | Answers are grounded solely in the configured knowledge-base folder (PDF, TXT, MD, DOCX). No external knowledge; explicit "not in the excerpts" when absent. |
| **Configuration** | Single source of truth: all settings from `.env`. No hardcoded overrides; COMPLETE and FAST each have dedicated env vars. |
| **Modes** | FAST (laptop) and COMPLETE (server/API) with distinct RAG and LLM settings, retriever type (similarity vs MMR), and prompts. |
| **Resilience** | Groq 429 retry with exponential backoff; automatic fallback to Ollama on repeated 429. Timeouts and clear error messages in the UI. |
| **UX** | Streaming when supported; source citations with expandable excerpts; conversation memory for follow-ups. Config warning if Groq is selected but API key is missing. |
| **Code quality** | Module and function docstrings; type hints on public helpers; clear sectioning (config, cache, loaders, chain, UI); no emojis or informal comments in code. |
| **Documentation** | README covers architecture, code structure, tools and roles, run-from-scratch, configuration, and troubleshooting. `.env.example` documents every variable. |

---

## License

Use and modify as needed for your project.
