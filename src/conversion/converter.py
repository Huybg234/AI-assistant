"""
converter.py — Convert any supported document format to PDF bytes.

Conversion chain per format:
  DOCX        → Word COM (ExportAsFixedFormat) → PDF, fallback: python-docx text → fpdf2
  DOC         → OLE text extraction → fpdf2
  TXT / RTF   → text extraction → fpdf2
  PDF         → pass-through (already PDF)
"""

import io
import os
import tempfile
import traceback
from typing import Union

# Windows system fonts path — Arial supports Vietnamese
_WINDOWS_FONTS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
]
_LINUX_FONTS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


def _find_unicode_font() -> str:
    """Return path to a TTF font that supports Unicode/Vietnamese, or ''."""
    for path in _WINDOWS_FONTS + _LINUX_FONTS:
        if os.path.isfile(path):
            return path
    return ""


def _text_to_pdf_bytes(text: str, title: str = "") -> bytes:
    """Render plain text to PDF bytes using fpdf2 with a Unicode-capable font."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Try TTF font for full Unicode support; fall back to built-in Helvetica
    font_name = "Helvetica"
    try:
        font_path = _find_unicode_font()
        if font_path:
            pdf.add_font("unifont", fname=font_path)
            font_name = "unifont"
    except Exception:
        font_name = "Helvetica"

    w = pdf.w - pdf.l_margin - pdf.r_margin  # effective page width, avoids cursor drift

    def _safe_cell(s: str) -> str:
        """Strip characters that cannot be encoded by the active font."""
        if font_name == "Helvetica":
            return s.encode("latin-1", errors="replace").decode("latin-1")
        return s

    # Title header
    if title:
        pdf.set_font(font_name, size=14)
        pdf.set_text_color(26, 35, 126)
        pdf.multi_cell(w, 9, _safe_cell(title))
        pdf.ln(3)
        pdf.set_draw_color(197, 202, 233)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(5)

    pdf.set_font(font_name, size=11)
    pdf.set_text_color(26, 26, 26)

    for line in text.splitlines():
        pdf.set_x(pdf.l_margin)   # reset x before every line
        if line.strip():
            pdf.multi_cell(w, 6, _safe_cell(line))
        else:
            pdf.ln(3)

    output = pdf.output()
    if isinstance(output, (bytes, bytearray)):
        return bytes(output)
    return output.encode("latin-1")


def _word_com_to_pdf(file_path: str, out_pdf: str) -> None:
    """Use Word COM automation directly to convert any Word document to PDF.
    Supports both .doc and .docx.
    """
    import win32com.client
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0          # suppress all dialogs
    word.AutomationSecurity = 1     # msoAutomationSecurityLow — allow macros

    abs_in  = os.path.abspath(file_path)
    abs_out = os.path.abspath(out_pdf)

    try:
        doc = word.Documents.Open(
            abs_in,
            ConfirmConversions=False,
            ReadOnly=False,
            AddToRecentFiles=False,
        )
        try:
            # ExportAsFixedFormat: works in Word 2007+ for both .doc and .docx
            doc.ExportAsFixedFormat(
                OutputFileName=abs_out,
                ExportFormat=17,          # wdExportFormatPDF
                OpenAfterExport=False,
                OptimizeFor=0,            # wdExportOptimizeForPrint
                Range=0,                  # wdExportAllDocument
                IncludeDocProps=True,
                BitmapMissingFonts=True,
            )
        except Exception:
            # Fallback: SaveAs2 with wdFormatPDF
            doc.SaveAs2(abs_out, FileFormat=17)
        doc.Close(0)
    finally:
        word.Quit()


def _docx_to_pdf_bytes(file_path: str) -> bytes:
    """Convert DOCX/DOC file to PDF using Word COM automation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_pdf = os.path.join(tmp_dir, "output.pdf")
        _word_com_to_pdf(file_path, out_pdf)
        with open(out_pdf, "rb") as f:
            return f.read()


def _docx_bytes_to_pdf(data: bytes, suffix: str = ".docx") -> bytes:
    """Write bytes to a temp file, convert with Word, return PDF bytes."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return _docx_to_pdf_bytes(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def convert_to_pdf(
    data: bytes,
    filename: str,
    file_type: str = "",
) -> bytes:
    """
    Convert document bytes to PDF bytes.

    Parameters
    ----------
    data      : raw file bytes
    filename  : original filename (used for title and extension detection)
    file_type : 'pdf', 'docx', 'doc', 'txt', 'rtf' (falls back to filename ext)

    Returns
    -------
    bytes — PDF file content
    """
    if not file_type:
        file_type = os.path.splitext(filename.lower())[1].lstrip(".")

    # ── PDF: already PDF ──────────────────────────────────────────────────────
    if file_type == "pdf":
        return data

    # ── DOCX: Word COM conversion (most faithful layout) ─────────────────────
    if file_type == "docx":
        try:
            return _docx_bytes_to_pdf(data, suffix=".docx")
        except Exception:
            # Word COM failed → fallback to python-docx text extraction
            try:
                import docx as _docx
                doc = _docx.Document(io.BytesIO(data))
                text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except Exception:
                text = "(Không thể chuyển đổi file sang PDF.)"
            return _text_to_pdf_bytes(text, title=filename)

    # ── DOC: mammoth first (for XML-format .doc), then OLE text extraction ────
    # Word COM is NOT used for .doc — it causes UnicodeDecodeError via win32com
    # charmap codec when processing .doc files through COM on some Windows setups.
    if file_type == "doc":
        text = ""
        try:
            import mammoth as _mammoth
            result = _mammoth.extract_raw_text(io.BytesIO(data))
            text = (result.value or "").strip()
        except Exception:
            pass
        if not text:
            try:
                from src.ingestion.file_reader import _extract_text_from_doc_ole
                text = _extract_text_from_doc_ole(data).strip()
            except Exception:
                pass
        if not text:
            text = "(Không thể đọc nội dung file .doc.)"
        return _text_to_pdf_bytes(text, title=filename)

    # ── TXT: decode text ──────────────────────────────────────────────────────
    if file_type == "txt":
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = data.decode("utf-16")
            except Exception:
                text = data.decode("latin-1", errors="replace")
        return _text_to_pdf_bytes(text, title=filename)

    # ── RTF: striprtf → text ──────────────────────────────────────────────────
    if file_type == "rtf":
        try:
            from striprtf.striprtf import rtf_to_text
            try:
                rtf_str = data.decode("utf-8")
            except UnicodeDecodeError:
                rtf_str = data.decode("latin-1", errors="replace")
            text = rtf_to_text(rtf_str)
        except Exception:
            text = data.decode("latin-1", errors="replace")
        return _text_to_pdf_bytes(text, title=filename)

    raise ValueError(f"Unsupported file type for PDF conversion: '{file_type}'")
