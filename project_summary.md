# Tóm Tắt Luồng Hoạt Động — AI Document Assistant

## Tổng quan

AI Document Assistant là hệ thống hỏi đáp tài liệu thông minh, kết hợp kỹ thuật **Retrieval-Augmented Generation (RAG)** với **Mô hình Ngôn ngữ Lớn (LLM)** để trả lời câu hỏi chính xác dựa trên nội dung tài liệu PDF thực tế.

---

## Luồng hoạt động

### Giai đoạn 1 — Ingest tài liệu
> 📄 **Code:** `src/vector_db_ingestor.py` — hàm `run_ingestion()`
> 🌐 **Trigger:** `POST /api/upload_and_ingest` trong `src/main.py`

Người dùng upload file PDF → hàm `run_ingestion()` dùng **PyPDF2** trích xuất text từng trang → tách thành các **chunk nhỏ** (~600 ký tự, overlap 150 ký tự) theo ranh giới câu (regex `_SENTENCE_END`) → mỗi chunk được embed thành **vector 384 chiều** bằng `SentenceTransformer("all-MiniLM-L6-v2")` (Singleton `_get_embed_model()`) → lưu vào **FAISS `IndexFlatL2`** cùng `mapping.npy`, `chunks.json`, `pages.json`, `metadata.json`. Mỗi tài liệu được gán một `doc_id` (UUID) và lưu độc lập tại `src/documents/{doc_id}/`.

---

### Giai đoạn 2 — Tiền xử lý câu hỏi
> 📄 **Code:** `src/preprocessor.py` — hàm `preprocess_conversation()`
> 🔗 **Gọi bởi:** `src/in_context_learning_module.py` (Stage 1 trong pipeline)

Câu hỏi của người dùng được làm sạch text (`clean_text()`), validate JSON nếu cần (`validate_json()`), và gắn metadata (request_id UUID, timestamp, message_count). Đây là bước đảm bảo đầu vào hợp lệ trước khi đưa vào pipeline chính.

---

### Giai đoạn 3 — RAG 2 tầng kết hợp LLM *(trọng tâm)*
> 📄 **Code:** `src/prompt_enricher.py` — class `PromptEnricher`, hàm `enrich()`
> 📄 **RAG retrieval:** `src/vector_db_ingestor.py` — hàm `retrieve_similar_text()`
> 📄 **CoT instructions:** `src/chain_service.py` — dict `_COT_STEPS`
> 🔗 **Gọi bởi:** `src/in_context_learning_module.py` (Stage 2 trong pipeline)

Đây là bước cốt lõi — hệ thống thực hiện **RAG 2 giai đoạn** trước khi gọi LLM:

**Bước 3a — Retrieval thô (FAISS)**
> 📄 `src/vector_db_ingestor.py` — hàm `retrieve_similar_text(query, top_k=15, doc_id=None)`

Câu hỏi được embed → FAISS tìm **top-15 chunk** gần nhất trong không gian vector. Nếu có `doc_id` → chỉ tìm trong index của tài liệu đó; nếu không → tìm khắp tất cả tài liệu.

**Bước 3b — Reranking (CrossEncoder)**
> 📄 `src/prompt_enricher.py` — hàm `_retrieve_rag_context()`, Singleton `_get_reranker()`
> Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

Mô hình CrossEncoder nhận từng cặp *(câu hỏi, chunk)* và tính điểm relevance → sắp xếp lại → chọn ra **top-5 chunk** thực sự liên quan nhất.

**Bước 3c — Làm giàu Prompt**
> 📄 `src/prompt_enricher.py` — hàm `enrich()` dùng `FewShotPromptTemplate` (LangChain)
> 📄 Few-shot examples: `src/prompt_enricher.py` — dict `_FEW_SHOT_EXAMPLES` (24 examples)
> 📄 CoT steps: `src/chain_service.py` — hàm `ChainService.get_cot_instructions()`

Top-5 chunk được nhúng vào prompt cùng với:
- **Chain-of-Thought (CoT):** 4 bước suy luận có cấu trúc tùy theo loại tác vụ (`documentQA` / `documentSummary` / `documentExtraction`).
- **In-Context Learning (ICL):** few-shot examples mẫu (10 QA + 6 Summary + 8 Extraction) định hướng LLM trả lời đúng format.

> **Tại sao RAG + LLM?** — LLM thuần túy có thể "bịa" thông tin (hallucination). RAG neo câu trả lời vào các đoạn văn bản gốc cụ thể, buộc LLM chỉ tổng hợp từ ngữ cảnh thực tế thay vì dùng kiến thức nội bộ — tăng độ chính xác và tính truy nguồn.

---

### Giai đoạn 4 — Gọi LLM
> 📄 **Code:** `src/in_context_learning_module.py` — `self.llm.invoke(enriched_prompt)` (Stage 3)
> 🤖 **Model:** `gemini-2.5-flash` qua `langchain_google_genai.ChatGoogleGenerativeAI`
> ⚙️ **Cấu hình:** `temperature=0.2`, `max_output_tokens=2048`, `top_p=0.95` (env vars)

Prompt hoàn chỉnh (CoT + ICL + top-5 RAG chunks + câu hỏi) được gửi đến **Gemini 2.5 Flash**. LLM sinh ra câu trả lời mạch lạc, có cấu trúc, dựa trên đúng ngữ cảnh đã được truy xuất.

---

### Giai đoạn 5 — Hậu xử lý
> 📄 **Code:** `src/postprocessor.py` — class `Postprocessor`, hàm `process()`
> 🔗 **Gọi bởi:** `src/in_context_learning_module.py` (Stage 4 trong pipeline)

Kết quả LLM đi qua:
1. **Safety check** — regex `_UNSAFE_PATTERNS` + kiểm tra nội dung blocked → raise `SafetyViolationError` nếu vi phạm.
2. **Truncation** — cắt ngắn nếu vượt `MAX_RESPONSE_CHARS = 10_000`.
3. **Format** — trả về dict chuẩn `{answer, char_count, truncated}`.
4. **Logging** — ghi vào `app.log` qua `src/utils.py`.

---

### Điều phối toàn bộ pipeline
> 📄 **Code:** `src/in_context_learning_module.py` — `InContextLearningModule.generate_response_from_conversation()`
> 🌐 **Expose qua API:** `src/main.py` — các endpoint `POST /api/qa`, `/api/summarize`, `/api/extract`

`InContextLearningModule` là orchestrator duy nhất điều phối cả 4 giai đoạn theo thứ tự: **Preprocess → Enrich (RAG) → LLM → Postprocess**, đo thời gian toàn pipeline và trả về `{answer, metadata}`.

---

## Sơ đồ tóm tắt

```
PDF Upload  ──────────────────────────────────────────────────────┐
[main.py: POST /api/upload_and_ingest]                            │
    │                                                             │
    ▼                                                             ▼
[vector_db_ingestor.py: run_ingestion()]            src/documents/{doc_id}/
  - PyPDF2 extract text                               ├── faiss_index.idx
  - chunk (600c, overlap 150c)                        ├── mapping.npy
  - SentenceTransformer embed                         ├── chunks.json
  - FAISS IndexFlatL2 save                            ├── pages.json
                                                      └── metadata.json

User Query
[main.py: POST /api/qa | /api/summarize | /api/extract]
    │
    ▼
[in_context_learning_module.py: generate_response_from_conversation()]
    │
    ├── Stage 1: [preprocessor.py: preprocess_conversation()]
    │              clean text · assign request_id · validate
    │
    ├── Stage 2: [prompt_enricher.py: enrich()]
    │   │
    │   ├── RAG Tầng 1: [vector_db_ingestor.py: retrieve_similar_text()]
    │   │                 embed query → FAISS top-15 chunks
    │   │
    │   ├── RAG Tầng 2: [prompt_enricher.py: _retrieve_rag_context()]
    │   │                 CrossEncoder rerank → top-5 chunks
    │   │
    │   ├── CoT:         [chain_service.py: _COT_STEPS]
    │   │                 4-step reasoning instructions
    │   │
    │   └── ICL:         [prompt_enricher.py: _FEW_SHOT_EXAMPLES]
    │                     24 few-shot examples (QA/Summary/Extraction)
    │
    ├── Stage 3: [in_context_learning_module.py: self.llm.invoke()]
    │              Gemini 2.5 Flash  ←  enriched prompt
    │
    └── Stage 4: [postprocessor.py: process()]
                   safety check · truncate · format · log
                       │
                       ▼
                  JSON Response  →  Client
```

---

*Hệ thống hỗ trợ 9 API endpoint: upload, QA, tóm tắt, trích xuất thông tin, và quản lý tài liệu (CRUD) — tất cả đều tận dụng cùng pipeline RAG + LLM nêu trên.*
