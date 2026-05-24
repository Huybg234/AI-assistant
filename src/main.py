import PyPDF2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse, Response
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import os
import traceback
import io
from starlette import status
from src.pipeline.orchestrator import InContextLearningModule
from src.pipeline.postprocessor import SafetyViolationError
from src.pipeline.preprocessor import Preprocessor
from src.ingestion.ingestor import (
    run_ingestion,
    list_documents,
    get_document,
    get_document_file_path,
    get_document_pdf_path,
    delete_document,
    update_document_chunks,
)
from src.ingestion.file_reader import extract_pages_from_file, _extract_text_from_doc_ole
from src.utils import DOCS_DIR, SUPPORTED_EXTENSIONS, MIME_TO_EXT, log_request

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
    file_type: Optional[str] = None
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

_ACCEPTED_TYPES_MSG = f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"

def _validate_upload_file(file: UploadFile) -> str:
    """Validate uploaded file extension. Returns the file extension (e.g. '.pdf')."""
    import os as _os
    ext = _os.path.splitext(file.filename or "")[1].lower()
    if not ext and file.content_type:
        ext = MIME_TO_EXT.get(file.content_type, "")
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext or file.content_type}'. {_ACCEPTED_TYPES_MSG}",
        )
    return ext

def extract_text_from_file(file_stream: io.BytesIO, filename: str = "", content_type: str = "") -> str:
    """Extract plain text from any supported file format."""
    try:
        file_bytes = file_stream.read()
        pages = extract_pages_from_file(file_bytes, filename=filename, content_type=content_type)
        text = "\n\n".join(p for p in pages if p and p.strip())
        return preprocessor.clean_text(text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error extracting text from file: {str(e)}")

# Keep legacy name for any internal callers
def extract_text_from_pdf(file_stream: io.BytesIO) -> str:
    return extract_text_from_file(file_stream, filename="document.pdf")

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
def serve_document_file(doc_id: str):
    """Serve the original uploaded file for in-browser viewing or download."""
    try:
        file_path, mime_type = get_document_file_path(doc_id)
        detail   = get_document(doc_id)
        filename = detail["metadata"]["filename"]
        return FileResponse(
            path=file_path,
            media_type=mime_type,
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/documents/{doc_id}/preview", response_class=HTMLResponse)
def preview_document(doc_id: str):
    """
    Returns an HTML page suitable for embedding in an iframe.
    - DOCX → mammoth converts directly from original file to styled HTML
    - DOC / TXT / RTF → re-extracted from original file, wrapped in readable HTML
    - PDF → not handled here; frontend uses /pdf endpoint directly
    """
    try:
        file_path, mime_type = get_document_file_path(doc_id)
        detail   = get_document(doc_id)
        meta     = detail["metadata"]
        filename = meta["filename"]
        # Detect file type: prefer metadata field, fall back to filename extension
        file_type = (meta.get("file_type") or "").lower().strip(".")
        if not file_type:
            import os as _os
            ext = _os.path.splitext(filename.lower())[1].lstrip(".")
            file_type = ext
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    def _wrap_html(title: str, body_html: str) -> str:
        safe_title = title.replace("<", "&lt;").replace(">", "&gt;")
        return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{safe_title}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: #fff; }}
  body {{
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
    line-height: 1.75;
    color: #1a1a1a;
    padding: 24px 36px 48px;
    max-width: 900px;
    margin: 0 auto;
  }}
  h1 {{ font-size: 1.6em; color: #0d1b6e; border-bottom: 2px solid #c5cae9; padding-bottom: 6px; margin-top: 1.4em; }}
  h2 {{ font-size: 1.35em; color: #1a237e; margin-top: 1.2em; }}
  h3 {{ font-size: 1.15em; color: #283593; margin-top: 1em; }}
  h4, h5, h6 {{ color: #303f9f; margin-top: 0.8em; }}
  p {{ margin: 0.5em 0; }}
  ul, ol {{ padding-left: 1.6em; margin: 0.5em 0; }}
  li {{ margin: 0.2em 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 13px; }}
  td, th {{ border: 1px solid #ccc; padding: 7px 11px; text-align: left; vertical-align: top; }}
  th {{ background: #e8eaf6; font-weight: 600; }}
  tr:nth-child(even) td {{ background: #f5f5f5; }}
  strong, b {{ color: #111; }}
  em, i {{ color: #333; }}
  a {{ color: #1565c0; }}
  pre.txt {{
    white-space: pre-wrap;
    word-break: break-word;
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 16px 20px;
    font-size: 13px;
    font-family: "Consolas", "Courier New", monospace;
    line-height: 1.65;
  }}
  .file-header {{
    background: #e8eaf6;
    border-left: 4px solid #3949ab;
    padding: 8px 14px;
    border-radius: 4px;
    margin-bottom: 18px;
    font-size: 12px;
    color: #3949ab;
  }}
</style>
</head>
<body>
<div class="file-header">📄 {safe_title}</div>
{body_html}
</body>
</html>"""

    try:
        with open(file_path, "rb") as f:
            raw_bytes = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot read file: {e}")

    # ── DOCX: mammoth → rich HTML ─────────────────────────────────────────────
    if file_type == "docx":
        try:
            import mammoth, io as _io
            result = mammoth.convert_to_html(_io.BytesIO(raw_bytes))
            body = result.value or "<p><em>(Không có nội dung)</em></p>"
            return HTMLResponse(content=_wrap_html(filename, body), media_type="text/html; charset=utf-8")
        except Exception:
            traceback.print_exc()
            # fall through to plain-text rendering

    # ── DOC: try mammoth first (handles XML-based .doc), then clean OLE fallback ─
    if file_type == "doc":
        text = ""
        try:
            import mammoth, io as _io
            result = mammoth.extract_raw_text(_io.BytesIO(raw_bytes))
            text = (result.value or "").strip()
        except Exception:
            pass
        if not text:
            text = _extract_text_from_doc_ole(raw_bytes).strip()
        if not text:
            text = "(Không thể đọc nội dung file .doc. Hãy thử chuyển sang định dạng .docx.)"
        import html as _html
        body = f"<pre class='txt'>{_html.escape(text)}</pre>"
        return HTMLResponse(content=_wrap_html(filename, body), media_type="text/html; charset=utf-8")

    # ── TXT: decode and display ───────────────────────────────────────────────
    if file_type == "txt":
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw_bytes.decode("utf-16")
            except Exception:
                text = raw_bytes.decode("latin-1", errors="replace")
        import html as _html
        body = f"<pre class='txt'>{_html.escape(text)}</pre>"
        return HTMLResponse(content=_wrap_html(filename, body), media_type="text/html; charset=utf-8")

    # ── RTF: striprtf → plain text ────────────────────────────────────────────
    if file_type == "rtf":
        try:
            from striprtf.striprtf import rtf_to_text
            try:
                rtf_str = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                rtf_str = raw_bytes.decode("latin-1", errors="replace")
            text = rtf_to_text(rtf_str)
        except Exception:
            text = raw_bytes.decode("latin-1", errors="replace")
        import html as _html
        body = f"<pre class='txt'>{_html.escape(text)}</pre>"
        return HTMLResponse(content=_wrap_html(filename, body), media_type="text/html; charset=utf-8")

    # ── Unknown / fallback: display stored pages as plain text ───────────────
    pages = detail.get("pages", [])
    plain = "\n\n".join(p for p in pages if p and p.strip()) or "(Không có nội dung)"
    import html as _html
    body = f"<pre class='txt'>{_html.escape(plain)}</pre>"
    return HTMLResponse(content=_wrap_html(filename, body), media_type="text/html; charset=utf-8")

@app.get("/api/documents/{doc_id}/as-pdf")
def document_as_pdf(doc_id: str):
    """
    Convert any supported document to PDF and return it for inline viewing.
    The result is cached as converted.pdf inside the document directory.
    PDF files are returned directly without conversion.
    """
    try:
        file_path, mime_type = get_document_file_path(doc_id)
        detail   = get_document(doc_id)
        meta     = detail["metadata"]
        filename = meta["filename"]
        file_type = (meta.get("file_type") or "").lower().strip(".")
        if not file_type:
            file_type = os.path.splitext(filename.lower())[1].lstrip(".")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # PDFs are served directly
    if file_type == "pdf":
        return FileResponse(
            path=file_path,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
        )

    # Check for cached conversion (only serve if file is a valid non-empty PDF)
    doc_dir = os.path.dirname(file_path)
    cache_path = os.path.join(doc_dir, "converted.pdf")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:
        cached = open(cache_path, "rb").read()
        if cached[:4] == b"%PDF":
            return Response(
                content=cached,
                media_type="application/pdf",
                headers={"Content-Disposition": f'inline; filename="{os.path.splitext(filename)[0]}.pdf"'},
            )
        # Invalid cache — delete and reconvert
        try:
            os.unlink(cache_path)
        except OSError:
            pass

    # Convert and cache
    try:
        from src.conversion.converter import convert_to_pdf
        with open(file_path, "rb") as f:
            raw_bytes = f.read()
        pdf_bytes = convert_to_pdf(raw_bytes, filename, file_type)
        if not pdf_bytes or pdf_bytes[:4] != b"%PDF":
            raise ValueError(f"Conversion produced invalid PDF (got {len(pdf_bytes)} bytes)")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Cannot convert file to PDF: {e}")
    try:
        with open(cache_path, "wb") as f:
            f.write(pdf_bytes)
    except Exception:
        pass  # cache failure is non-fatal
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{os.path.splitext(filename)[0]}.pdf"'},
    )


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
async def upload_and_ingest_document(file: UploadFile = File(..., description="Document file to process (PDF, DOCX, DOC, TXT, RTF)")):
    """
    Uploads a document, extracts its text, chunks it, and ingests it into the vector database.
    Supported formats: PDF, DOCX, DOC, TXT, RTF.
    Returns a doc_id that can be used to target this document in Q&A, view, edit, or delete.
    """
    _validate_upload_file(file)

    print(f"Received file for ingestion: {file.filename}")
    try:
        file_content = await file.read()
        doc_id = run_ingestion(file_content, file.filename, content_type=file.content_type or "")
        print(f"Successfully ingested document: {file.filename} → doc_id={doc_id}")
        return {
            "message": f"Document '{file.filename}' ingested successfully.",
            "doc_id": doc_id,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
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
    file: Optional[UploadFile] = File(None, description="Document file to summarize (PDF, DOCX, DOC, TXT, RTF)"),
):
    """
    Summarizes a document. Supply either `doc_id` (already-ingested) or a document `file`.
    Supported file formats: PDF, DOCX, DOC, TXT, RTF.
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
            _validate_upload_file(file)
            file_content = await file.read()
            document_text = extract_text_from_file(
                io.BytesIO(file_content),
                filename=file.filename or "",
                content_type=file.content_type or "",
            )
            if not document_text:
                raise HTTPException(status_code=422, detail="Could not extract text from the file.")
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
    Supply either `doc_id` (already-ingested) or a document `file` (PDF, DOCX, DOC, TXT, RTF).
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
            _validate_upload_file(file)
            file_content = await file.read()
            document_text = extract_text_from_file(
                io.BytesIO(file_content),
                filename=file.filename or "",
                content_type=file.content_type or "",
            )
            if not document_text:
                raise HTTPException(status_code=422, detail="Could not extract text from the file.")
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
