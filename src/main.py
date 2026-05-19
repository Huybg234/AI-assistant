import PyPDF2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import os
import traceback
import io
from starlette import status
from src.in_context_learning_module import InContextLearningModule
from src.postprocessor import SafetyViolationError
from src.preprocessor import Preprocessor
from src.vector_db_ingestor import (
    run_ingestion,
    list_documents,
    get_document,
    get_document_pdf_path,
    delete_document,
    update_document_chunks,
    DOCS_DIR,
)
from src.utils import log_request

from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# --- Setup ---
app = FastAPI(
    title="AI Document Assistant API",
    description="API for interacting with documents using AI.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
module = InContextLearningModule()
preprocessor = Preprocessor()

# --- Pydantic Models ---

class DocumentInteractionRequest(BaseModel):
    conversation: List[Dict[str, str]]
    conversation_type: str  # "documentQA", "documentSummary", "documentExtraction"
    doc_ids: Optional[List[str]] = None  # restrict Q&A to specific documents (None = all)

class ResponseMetadata(BaseModel):
    request_id: str
    timestamp: str
    processing_time_ms: int
    truncated: bool
    char_count: int

class QAResponse(BaseModel):
    answer: str
    metadata: ResponseMetadata

class SummaryResponse(BaseModel):
    summary: str
    metadata: ResponseMetadata

class ExtractionResponse(BaseModel):
    extracted_data: List[str]
    metadata: ResponseMetadata

class DocumentMetadata(BaseModel):
    doc_id: str
    filename: str
    uploaded_at: str
    num_pages: int
    num_chunks: int
    last_edited_at: Optional[str] = None

class DocumentDetail(BaseModel):
    metadata: DocumentMetadata
    pages: List[str]
    chunks: List[str]

class EditChunksRequest(BaseModel):
    chunks: List[str]

# --- Helper Functions ---

def extract_text_from_pdf(file_stream: io.BytesIO) -> str:
    try:
        reader = PyPDF2.PdfReader(file_stream)
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        return preprocessor.clean_text(text)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

def _build_metadata(raw_meta: Dict[str, Any]) -> ResponseMetadata:
    return ResponseMetadata(
        request_id=raw_meta["request_id"],
        timestamp=raw_meta["timestamp"],
        processing_time_ms=raw_meta["processing_time_ms"],
        truncated=raw_meta["truncated"],
        char_count=raw_meta["char_count"],
    )

def _handle_safety_error(exc: SafetyViolationError) -> None:
    raise HTTPException(status_code=422, detail=f"Safety check failed: {exc}")

def _get_document_text(doc_id: str) -> str:
    """Lấy toàn bộ text của document đã ingest từ pages.json."""
    try:
        detail = get_document(doc_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    pages = detail.get("pages", [])
    text = "\n\n".join(p for p in pages if p and p.strip())
    if not text.strip():
        raise HTTPException(status_code=422, detail=f"Document '{doc_id}' contains no extractable text.")
    return preprocessor.clean_text(text)

# --- API Endpoints ---

@app.get("/api/health")
def health_check():
    """Returns service status and the number of ingested documents."""
    import os
    doc_count = 0
    if os.path.isdir(DOCS_DIR):
        doc_count = sum(
            1 for e in os.listdir(DOCS_DIR)
            if os.path.isfile(os.path.join(DOCS_DIR, e, "metadata.json"))
        )
    return {"status": "ok", "document_count": doc_count}

# ── Document management ──────────────────────────────────────────────────────

@app.get("/api/documents", response_model=List[DocumentMetadata])
def list_all_documents():
    """List all ingested documents with their metadata."""
    return list_documents()

@app.get("/api/documents/{doc_id}", response_model=DocumentDetail)
def view_document(doc_id: str):
    """
    View the full content of an ingested document:
    metadata, extracted pages (raw text per page), and chunks used for RAG.
    """
    try:
        detail = get_document(doc_id)
        return DocumentDetail(
            metadata=DocumentMetadata(**detail["metadata"]),
            pages=detail["pages"],
            chunks=detail["chunks"],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/documents/{doc_id}/pdf")
def serve_document_pdf(doc_id: str):
    """Serve the original PDF file for in-browser viewing."""
    try:
        pdf_path = get_document_pdf_path(doc_id)
        detail   = get_document(doc_id)
        filename = detail["metadata"]["filename"]
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.put("/api/documents/{doc_id}/chunks")
def edit_document_chunks(doc_id: str, request: EditChunksRequest):
    """
    Replace the chunks of an existing document and re-build its FAISS index.

    Use this to correct OCR errors, remove irrelevant sections, or otherwise
    curate the content that the Q&A pipeline will retrieve from.

    Body: { "chunks": ["chunk text 1", "chunk text 2", ...] }
    """
    if not request.chunks:
        raise HTTPException(status_code=400, detail="'chunks' list must not be empty.")
    try:
        update_document_chunks(doc_id, request.chunks)
        return {"message": f"Document '{doc_id}' updated successfully. FAISS index rebuilt with {len(request.chunks)} chunks."}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{doc_id}")
def remove_document(doc_id: str):
    """Permanently delete a document and its FAISS index."""
    try:
        delete_document(doc_id)
        return {"message": f"Document '{doc_id}' deleted successfully."}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ── Core AI endpoints ────────────────────────────────────────────────────────

@app.post("/api/upload_and_ingest")
async def upload_and_ingest_pdf(file: UploadFile = File(..., description="PDF file to process")):
    """
    Uploads a PDF, extracts its text, chunks it, and ingests it into the vector database.
    Returns a doc_id that can be used to target this document in Q&A, view, edit, or delete.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported.")

    print("Received PDF for ingestion...")
    try:
        file_content = await file.read()
        doc_id = run_ingestion(file_content, file.filename)
        print(f"Successfully ingested document: {file.filename} → doc_id={doc_id}")
        return {
            "message": f"Document '{file.filename}' ingested successfully.",
            "doc_id": doc_id,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")

@app.post("/api/qa", response_model=QAResponse)
def answer_question(request: DocumentInteractionRequest):
    """
    Answers a question based on ingested documents.
    Pipeline: Preprocess → Enrich (Few-Shot CoT + RAG + Reranking) → LLM → Postprocess.

    - If doc_ids is provided, RAG is restricted to those specific documents.
    - If doc_ids is omitted, RAG searches across all ingested documents.
    """
    if request.conversation_type != "documentQA":
        raise HTTPException(status_code=400, detail="Invalid conversation_type. Use 'documentQA'.")

    import os
    if not os.path.isdir(DOCS_DIR) or not any(
        os.path.isfile(os.path.join(DOCS_DIR, e, "faiss_index.idx"))
        for e in os.listdir(DOCS_DIR)
    ):
        raise HTTPException(status_code=400, detail="No document ingested. Use /api/upload_and_ingest first.")

    _, pre_meta = preprocessor.preprocess_conversation(request.conversation)
    log_request(pre_meta, "/api/qa")

    try:
        result = module.generate_response_from_conversation(
            conversation=request.conversation,
            type=request.conversation_type,
            doc_ids=request.doc_ids,
        )
        return QAResponse(answer=result["answer"], metadata=_build_metadata(result["metadata"]))
    except SafetyViolationError as e:
        _handle_safety_error(e)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize", response_model=SummaryResponse)
async def summarize_document(
    doc_id: Optional[str] = Form(None, description="ID of an already-ingested document"),
    file: Optional[UploadFile] = File(None, description="PDF file to summarize"),
):
    """
    Summarizes a document. Supply either `doc_id` (already-ingested) or a PDF `file`.
    Pipeline: Preprocess → Enrich (Few-Shot CoT) → LLM → Postprocess.
    """
    if doc_id and file:
        raise HTTPException(status_code=400, detail="Provide either 'doc_id' or 'file', not both.")
    if not doc_id and not file:
        raise HTTPException(status_code=400, detail="Either 'doc_id' or 'file' is required.")

    try:
        if doc_id:
            document_text = _get_document_text(doc_id)
        else:
            if file.content_type != "application/pdf":
                raise HTTPException(status_code=400, detail="Only PDF files are supported.")
            file_content = await file.read()
            document_text = extract_text_from_pdf(io.BytesIO(file_content))
            if not document_text:
                raise HTTPException(status_code=422, detail="Could not extract text from the PDF.")
            document_text = preprocessor.clean_text(document_text)

        conversation = [{"role": "user", "content": f"Please summarize the following document:\n\n{document_text}"}]
        _, pre_meta = preprocessor.preprocess_conversation(conversation)
        log_request(pre_meta, "/api/summarize")

        result = module.generate_response_from_conversation(
            conversation=conversation,
            type="documentSummary",
            doc_id=doc_id,
        )
        return SummaryResponse(summary=result["answer"], metadata=_build_metadata(result["metadata"]))

    except SafetyViolationError as e:
        _handle_safety_error(e)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extract", response_model=ExtractionResponse)
async def extract_information(
    request: str = Form(...),
    doc_id: Optional[str] = Form(None, description="ID of an already-ingested document"),
    file: Optional[UploadFile] = File(None),
):
    """
    Extracts specific information from a document based on a user's request.
    Supply either `doc_id` (already-ingested) or a PDF `file`.
    The *request* form field must be a valid JSON string with a 'query' key,
    e.g. {"query": "Extract all project names"}.
    """
    if doc_id and file:
        raise HTTPException(status_code=400, detail="Provide either 'doc_id' or 'file', not both.")
    if not doc_id and not file:
        raise HTTPException(status_code=400, detail="Either 'doc_id' or 'file' is required.")

    try:
        try:
            request_data = preprocessor.validate_json(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        query = request_data.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Request JSON must contain a non-empty 'query' field.")
        query = preprocessor.clean_text(query)

        if doc_id:
            document_text = _get_document_text(doc_id)
        else:
            if file.content_type != "application/pdf":
                raise HTTPException(status_code=400, detail="Only PDF files are supported.")
            file_content = await file.read()
            document_text = extract_text_from_pdf(io.BytesIO(file_content))
            if not document_text:
                raise HTTPException(status_code=422, detail="Could not extract text from the PDF.")
            document_text = preprocessor.clean_text(document_text)

        conversation = [{"role": "user", "content": f"Request: {query}\n\nText:\n{document_text}"}]
        _, pre_meta = preprocessor.preprocess_conversation(conversation)
        log_request(pre_meta, "/api/extract")

        result = module.generate_response_from_conversation(
            conversation=conversation,
            type="documentExtraction",
            doc_id=doc_id,
        )
        extracted_list = [item.strip() for item in result["answer"].split("\n") if item.strip()]
        return ExtractionResponse(
            extracted_data=extracted_list,
            metadata=_build_metadata(result["metadata"]),
        )

    except SafetyViolationError as e:
        _handle_safety_error(e)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize", response_model=SummaryResponse)
async def summarize_document(file: UploadFile = File(..., description="PDF file to summarize")):
    """
    Summarizes the content of an uploaded PDF.
    Pipeline: Preprocess → Enrich (Few-Shot CoT) → LLM → Postprocess.

    Response includes the summary and request metadata.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported.")

    try:
        file_content = await file.read()
        document_text = extract_text_from_pdf(io.BytesIO(file_content))

        if not document_text:
            raise HTTPException(status_code=422, detail="Could not extract text from the PDF.")

        conversation = [{"role": "user", "content": f"Please summarize the following document:\n\n{document_text}"}]

        _, pre_meta = preprocessor.preprocess_conversation(conversation)
        log_request(pre_meta, "/api/summarize")

        result = module.generate_response_from_conversation(
            conversation=conversation,
            type="documentSummary",
        )
        return SummaryResponse(summary=result["answer"], metadata=_build_metadata(result["metadata"]))

    except SafetyViolationError as e:
        _handle_safety_error(e)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extract", response_model=ExtractionResponse)
async def extract_information(request: str = Form(...), file: UploadFile = File(...)):
    """
    Extracts specific information from a PDF based on a user's request.
    The *request* form field must be a valid JSON string with a 'query' key,
    e.g. ``{"query": "Extract all project names"}``.
    Pipeline: Preprocess (JSON validation + text cleaning) → Enrich → LLM → Postprocess.

    Response includes the extracted list and request metadata.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported.")

    try:
        # Stage 1 – Preprocess: validate JSON and clean the query
        try:
            request_data = preprocessor.validate_json(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        query = request_data.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Request JSON must contain a non-empty 'query' field.")

        query = preprocessor.clean_text(query)

        file_content = await file.read()
        document_text = extract_text_from_pdf(io.BytesIO(file_content))

        if not document_text:
            raise HTTPException(status_code=422, detail="Could not extract text from the PDF.")

        conversation = [{"role": "user", "content": f"Request: {query}\n\nText:\n{document_text}"}]

        _, pre_meta = preprocessor.preprocess_conversation(conversation)
        log_request(pre_meta, "/api/extract")

        result = module.generate_response_from_conversation(
            conversation=conversation,
            type="documentExtraction",
        )
        extracted_list = [item.strip() for item in result["answer"].split("\n") if item.strip()]
        return ExtractionResponse(
            extracted_data=extracted_list,
            metadata=_build_metadata(result["metadata"]),
        )

    except SafetyViolationError as e:
        _handle_safety_error(e)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
