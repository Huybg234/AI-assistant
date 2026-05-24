# Copilot Instructions — AI Document Assistant

## Running the Project

**Start the backend API:**
```bash
# From repo root
uvicorn src.main:app --reload --port 8000
```

**Start the frontend UI:**
```bash
cd ui
python -m http.server 3000
# Open http://localhost:3000 (requires backend running at port 8000)
```

**Change API base URL for the UI:**
Edit the first line of `ui/app.js`:
```js
const API_BASE = "http://localhost:8000";
```

**Required environment variable:**
```
GOOGLE_API_KEY=<your Gemini API key>   # in .env at repo root
```

**LLM parameters** (optional env overrides):
```
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=2048
LLM_TOP_P=0.95
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## Architecture

The system is a RAG + LLM pipeline for PDF document question-answering, summarization, and information extraction.

### Pipeline Stages (per request)

All three AI endpoints (`/api/qa`, `/api/summarize`, `/api/extract`) run through the same 4-stage pipeline orchestrated by `InContextLearningModule.generate_response_from_conversation()`:

1. **Preprocess** (`src/preprocessor.py`) — Strip control chars, normalize whitespace, assign `request_id` (UUID) and timestamp.
2. **Enrich** (`src/prompt_enricher.py`) — Two-tier RAG + few-shot ICL + CoT injection:
   - **FAISS retrieval** (`src/vector_db_ingestor.py:retrieve_similar_text`) → top-15 candidate chunks
   - **CrossEncoder rerank** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) → top-5 final chunks
   - **CoT instructions** from `src/chain_service.py` injected per task type
   - **24 few-shot examples** (10 QA + 6 Summary + 8 Extraction) via LangChain `FewShotPromptTemplate`
3. **LLM call** — Gemini 2.5 Flash via `langchain_google_genai.ChatGoogleGenerativeAI`
4. **Postprocess** (`src/postprocessor.py`) — Safety check (regex + blocked phrases), truncate at 10,000 chars, return standard envelope `{answer, char_count, truncated}`.

### Document Storage

Each ingested PDF lives in `src/documents/{doc_id}/` (UUID):
```
src/documents/{doc_id}/
├── faiss_index.idx   # FAISS IndexFlatL2 (384-dim, all-MiniLM-L6-v2)
├── mapping.npy       # int → chunk text dict
├── chunks.json       # list of chunk strings
├── pages.json        # list of per-page text strings
├── metadata.json     # doc_id, filename, uploaded_at, num_pages, num_chunks, last_edited_at
└── original.pdf      # original uploaded file
```

`src/documents/` is gitignored — it is created at runtime.

### Supported Conversation Types

| `conversation_type` | Endpoint | RAG used |
|---|---|---|
| `documentQA` | `POST /api/qa` | Yes (FAISS + CrossEncoder) |
| `documentSummary` | `POST /api/summarize` | No (full text passed directly) |
| `documentExtraction` | `POST /api/extract` | No (full text passed directly) |

### All API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Service status + ingested document count |
| `GET` | `/api/documents` | List all documents (metadata only) |
| `GET` | `/api/documents/{doc_id}` | Full document: metadata + pages + chunks |
| `GET` | `/api/documents/{doc_id}/pdf` | Serve original PDF inline |
| `PUT` | `/api/documents/{doc_id}/chunks` | Replace chunks and rebuild FAISS index |
| `DELETE` | `/api/documents/{doc_id}` | Permanently remove document |
| `POST` | `/api/upload_and_ingest` | Upload PDF → chunk → embed → store |
| `POST` | `/api/qa` | JSON body `{conversation, conversation_type, doc_ids?}` |
| `POST` | `/api/summarize` | Form: `doc_id` (ingested) **or** `file` (PDF upload) |
| `POST` | `/api/extract` | Form: `request` (JSON string with `query`), `doc_id` **or** `file` |

---

## Key Conventions

### Singleton models
Both the embedding model (`all-MiniLM-L6-v2` in `vector_db_ingestor.py`) and the CrossEncoder reranker (`prompt_enricher.py`) use module-level singletons loaded on first use. Never instantiate them directly — use `_get_embed_model()` and `_get_reranker()`.

### All imports use `src.` prefix
Modules import each other as `from src.X import Y`. The app is launched from the repo root, so `src` is a package. Do not use relative imports.

### LLM responses are always `AIMessage` objects
`self.llm.invoke(prompt)` returns a LangChain `AIMessage`. Text is extracted via `postprocessor.extract_content()`, which handles both `.content: str` and `.content: list[dict]` response shapes.

### Three task types are hard-coded throughout
`"documentQA"`, `"documentSummary"`, `"documentExtraction"` — used as keys in `_COT_STEPS`, `_ROLE_DESCRIPTIONS`, `_FEW_SHOT_EXAMPLES`, and validated in endpoint handlers. Adding a new task type requires updating all four places.

### `doc_ids` vs `doc_id` parameter
`generate_response_from_conversation()` accepts both `doc_ids: List[str]` (preferred) and `doc_id: str` (deprecated, single-doc shorthand). `doc_id` is internally converted to `doc_ids = [doc_id]`. New callers should use `doc_ids`.

### Chunking parameters
`chunk_text()` uses `chunk_size=600` chars and `overlap=150` chars with sentence-boundary alignment (`_SENTENCE_END` regex). `retrieve_similar_text()` fetches `top_k * 3` candidates (capped at 15) before CrossEncoder reranking to top-5.

### Response envelope
All AI endpoints return `{answer/summary/extracted_data, metadata}` where metadata includes `request_id`, `timestamp`, `processing_time_ms`, `truncated`, `char_count`. The `ResponseMetadata` Pydantic model is the single source of truth for this shape.

### LLM responds in the user's input language
The system prompts in `src/chain_service.py` (`_ROLE_DESCRIPTIONS`) instruct Gemini to detect the language of the user's question/request and respond in that same language. If the user writes in Vietnamese, the response is Vietnamese; in English, the response is English; etc.

### Safety layer
`Postprocessor.safety_check()` raises `SafetyViolationError` (a `ValueError` subclass) for empty responses, Gemini refusal phrases, and a small regex blocklist. All endpoint handlers catch this and return HTTP 422.

### Logging goes to `app.log`, not stdout
`InContextLearningModule` configures `logging.basicConfig` to write to `app.log` in the repo root (`filemode="a"`). `print()` calls in `main.py` and `vector_db_ingestor.py` do go to stdout, but structured request/pipeline logs are only in `app.log`.

### `/api/summarize` and `/api/extract` accept `doc_id` OR `file`
Both endpoints are Form-based (not JSON) and accept either an already-ingested `doc_id` or a raw PDF `file` upload. The `/api/extract` `request` field must be a JSON string with a `"query"` key, e.g. `{"query": "Extract all dates"}`.

### Extraction output is parsed from newlines
`/api/extract` splits the LLM's raw text output on `\n` to produce `extracted_data: List[str]`. The LLM is instructed (via CoT) to emit one item per line.
