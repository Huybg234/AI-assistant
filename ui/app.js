// ── Config ──────────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000";

// ── State ────────────────────────────────────────────────────────────────────
let documents = [];
let qaConversation = [];
let fileQueue = []; // { id, file, status, docId?, error? }

// ── Utilities ────────────────────────────────────────────────────────────────
function buildMetaFooter(meta) {
  if (!meta) return "";
  return `
    <div class="result-meta">
      <span>🆔 ${meta.request_id ?? "—"}</span>
      <span>⏱ ${meta.processing_time_ms ?? "—"} ms</span>
      <span>📝 ${meta.char_count ?? "—"} chars</span>
      <span>⏰ ${meta.timestamp ? new Date(meta.timestamp).toLocaleString("vi-VN") : "—"}</span>
    </div>`;
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

function getStatusLabel(status) {
  switch (status) {
    case "pending":   return "⏳ Chờ upload";
    case "uploading": return "📤 Đang upload...";
    case "success":   return "✅ Thành công";
    case "error":     return "❌ Lỗi";
    default:          return status;
  }
}

// ── Health check ─────────────────────────────────────────────────────────────
async function checkHealth() {
  const dot  = document.getElementById("health-indicator");
  const text = document.getElementById("health-text");
  try {
    const r = await fetch(`${API_BASE}/api/health`);
    if (r.ok) {
      const d = await r.json();
      dot.className    = "health-dot ok";
      text.textContent = `API OK · ${d.document_count} tài liệu`;
    } else throw new Error();
  } catch {
    dot.className    = "health-dot error";
    text.textContent = "Không kết nối được";
  }
}

// ── Navigation ───────────────────────────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(`tab-${tab}`).classList.add("active");
  document.querySelector(`[data-tab="${tab}"]`).classList.add("active");
  if (tab === "documents") loadDocuments();
  if (["qa", "summarize", "extract"].includes(tab)) populateDocSelects();
}

document.querySelectorAll(".nav-btn").forEach(btn => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

// ── Upload (multi-file) ───────────────────────────────────────────────────────
const dropZone  = document.getElementById("drop-zone");
const pdfInput  = document.getElementById("pdf-input");

dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  addFilesToQueue(e.dataTransfer.files);
});

pdfInput.addEventListener("change", () => {
  if (pdfInput.files.length) addFilesToQueue(pdfInput.files);
  pdfInput.value = "";
});

function addFilesToQueue(fileList) {
  const files = Array.from(fileList);
  const valid   = files.filter(f => f.type === "application/pdf");
  const invalid = files.length - valid.length;

  valid.forEach(file => {
    const id = `fq-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    fileQueue.push({ id, file, status: "pending" });
  });

  renderFileQueue();

  if (invalid > 0) {
    const toast = document.createElement("div");
    toast.className = "alert alert-warning alert-dismissible fade show mt-2 py-2";
    toast.style.fontSize = "13px";
    toast.innerHTML = `<i class="bi bi-exclamation-triangle me-1"></i>${invalid} file không phải PDF đã bị bỏ qua.
      <button type="button" class="btn-close btn-close-sm" data-bs-dismiss="alert"></button>`;
    document.getElementById("file-queue").prepend(toast);
  }
}

function renderFileQueue() {
  const container = document.getElementById("file-queue");
  const actions   = document.getElementById("upload-actions");
  const countEl   = document.getElementById("queue-count");

  if (fileQueue.length === 0) {
    container.innerHTML = "";
    actions.classList.add("d-none");
    return;
  }

  actions.classList.remove("d-none");
  const pendingCount = fileQueue.filter(f => f.status === "pending").length;
  countEl.textContent = pendingCount;

  container.innerHTML = fileQueue.map(item => `
    <div class="file-queue-item" id="fqi-${item.id}">
      <i class="bi bi-file-earmark-pdf file-pdf-icon"></i>
      <span class="file-name" title="${item.file.name}">${item.file.name}</span>
      <span class="file-size">${formatFileSize(item.file.size)}</span>
      <span class="file-status-badge ${item.status}" id="fqs-${item.id}"
        ${item.error ? `title="${item.error}"` : ""}>${getStatusLabel(item.status)}</span>
      ${item.status === "pending"
        ? `<button class="btn btn-sm btn-outline-secondary py-0 px-2" onclick="removeFromQueue('${item.id}')">
             <i class="bi bi-x"></i></button>`
        : ""}
    </div>`).join("");
}

function removeFromQueue(id) {
  fileQueue = fileQueue.filter(f => f.id !== id);
  renderFileQueue();
}

function clearQueue() {
  fileQueue = fileQueue.filter(f => f.status === "uploading");
  renderFileQueue();
}

async function uploadAllFiles() {
  const btn = document.getElementById("upload-all-btn");
  btn.disabled = true;

  const pending = fileQueue.filter(f => f.status === "pending");
  for (const item of pending) {
    item.status = "uploading";
    updateFileStatusBadge(item);

    try {
      const formData = new FormData();
      formData.append("file", item.file);

      const r = await fetch(`${API_BASE}/api/upload_and_ingest`, { method: "POST", body: formData });
      const d = await r.json();

      if (r.ok) {
        item.status = "success";
        item.docId  = d.doc_id;
      } else {
        item.status = "error";
        item.error  = d.detail ?? JSON.stringify(d);
      }
    } catch (err) {
      item.status = "error";
      item.error  = err.message;
    }

    updateFileStatusBadge(item);
  }

  btn.disabled = false;
  // Refresh pending count
  const countEl = document.getElementById("queue-count");
  countEl.textContent = fileQueue.filter(f => f.status === "pending").length;
  checkHealth();
}

function updateFileStatusBadge(item) {
  const badge = document.getElementById(`fqs-${item.id}`);
  if (!badge) return;
  badge.className = `file-status-badge ${item.status}`;
  badge.textContent = getStatusLabel(item.status);
  if (item.error) badge.title = item.error;

  // Remove delete button when no longer pending
  if (item.status !== "pending") {
    const row = document.getElementById(`fqi-${item.id}`);
    row && row.querySelector("button")?.remove();
  }
}

document.getElementById("upload-all-btn").addEventListener("click", uploadAllFiles);
document.getElementById("clear-queue-btn").addEventListener("click", clearQueue);

// ── Documents ────────────────────────────────────────────────────────────────
async function loadDocuments() {
  const list = document.getElementById("doc-list");
  list.innerHTML = `<div class="empty-msg"><div class="spinner-border spinner-border-sm text-primary me-2"></div>Đang tải...</div>`;
  try {
    const r = await fetch(`${API_BASE}/api/documents`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    documents = await r.json();
    renderDocList();
    populateDocSelects();
  } catch (err) {
    list.innerHTML = `<div class="empty-msg text-danger"><i class="bi bi-exclamation-circle me-1"></i>Lỗi tải tài liệu: ${err.message}</div>`;
  }
}

function renderDocList() {
  const list = document.getElementById("doc-list");
  if (!documents.length) {
    list.innerHTML = `<div class="empty-msg"><i class="bi bi-inbox fs-1 d-block mb-2 text-muted"></i>Chưa có tài liệu nào. Hãy upload PDF trước.</div>`;
    return;
  }
  list.innerHTML = documents.map(doc => `
    <div class="doc-card">
      <span class="doc-name"><i class="bi bi-file-earmark-pdf text-danger me-2"></i>${doc.filename}</span>
      <div class="doc-actions">
        <button class="btn btn-sm btn-primary rounded-pill" onclick="viewPdf('${doc.doc_id}', '${doc.filename.replace(/'/g,"\\'")}')">
          <i class="bi bi-eye me-1"></i>Xem PDF
        </button>
        <button class="btn btn-sm btn-outline-danger rounded-pill" onclick="deleteDocument('${doc.doc_id}', '${doc.filename.replace(/'/g,"\\'")}')">
          <i class="bi bi-trash"></i>
        </button>
      </div>
    </div>`).join("");
}

async function viewDocument(docId) {
  const modal = document.getElementById("doc-modal");
  document.getElementById("modal-title").textContent  = "Đang tải...";
  document.getElementById("modal-meta").innerHTML     = "";
  document.getElementById("modal-pages").textContent  = "";
  document.getElementById("modal-chunks").textContent = "";
  modal.classList.remove("hidden");

  try {
    const r = await fetch(`${API_BASE}/api/documents/${docId}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    const m = d.metadata;

    document.getElementById("modal-title").textContent = m.filename;
    document.getElementById("modal-meta").innerHTML = `
      <div class="meta-item"><div class="label">Doc ID</div><div class="value">${m.doc_id}</div></div>
      <div class="meta-item"><div class="label">Số trang</div><div class="value">${m.num_pages}</div></div>
      <div class="meta-item"><div class="label">Số chunks</div><div class="value">${m.num_chunks}</div></div>
      <div class="meta-item"><div class="label">Upload lúc</div><div class="value">${new Date(m.uploaded_at).toLocaleString("vi-VN")}</div></div>`;

    document.getElementById("modal-pages").textContent  = d.pages.join("\n\n--- Trang tiếp ---\n\n") || "(Không có nội dung)";
    document.getElementById("modal-chunks").textContent = d.chunks.map((c, i) => `[Chunk ${i + 1}]\n${c}`).join("\n\n") || "(Không có chunks)";
  } catch (err) {
    document.getElementById("modal-title").textContent = `Lỗi: ${err.message}`;
  }
}

async function deleteDocument(docId, filename) {
  if (!confirm(`Xóa tài liệu "${filename}"? Hành động này không thể hoàn tác.`)) return;
  try {
    const r = await fetch(`${API_BASE}/api/documents/${docId}`, { method: "DELETE" });
    if (r.ok) {
      await loadDocuments();
      checkHealth();
    } else {
      const d = await r.json();
      alert(`Lỗi: ${d.detail ?? "Không thể xóa tài liệu."}`);
    }
  } catch (err) {
    alert(`Lỗi kết nối: ${err.message}`);
  }
}

document.getElementById("refresh-docs").addEventListener("click", loadDocuments);
document.getElementById("close-modal").addEventListener("click", () =>
  document.getElementById("doc-modal").classList.add("hidden"));
document.getElementById("doc-modal").addEventListener("click", e => {
  if (e.target === document.getElementById("doc-modal"))
    document.getElementById("doc-modal").classList.add("hidden");
});

// ── PDF Viewer ────────────────────────────────────────────────────────────────
function viewPdf(docId, filename) {
  const modal       = document.getElementById("pdf-modal");
  const iframe      = document.getElementById("pdf-iframe");
  const loading     = document.getElementById("pdf-loading");
  const errorBox    = document.getElementById("pdf-error");
  const title       = document.getElementById("pdf-modal-title");
  const downloadBtn = document.getElementById("pdf-download-btn");
  const fallbackBtn = document.getElementById("pdf-fallback-btn");

  const pdfUrl = `${API_BASE}/api/documents/${docId}/pdf`;

  title.textContent = filename;
  downloadBtn.href  = pdfUrl;
  downloadBtn.setAttribute("download", filename);
  fallbackBtn.href  = pdfUrl;

  // Reset state
  iframe.classList.add("hidden");
  errorBox.classList.add("hidden");
  loading.classList.remove("hidden");
  iframe.src = "";

  modal.classList.remove("hidden");

  // Load PDF in iframe
  iframe.onload = () => {
    loading.classList.add("hidden");
    iframe.classList.remove("hidden");
  };
  iframe.onerror = () => {
    loading.classList.add("hidden");
    errorBox.classList.remove("hidden");
  };

  iframe.src = pdfUrl;

  // Fallback timeout — if iframe doesn't load in 8s, show error
  setTimeout(() => {
    if (!loading.classList.contains("hidden")) {
      loading.classList.add("hidden");
      errorBox.classList.remove("hidden");
    }
  }, 8000);
}

document.getElementById("close-pdf-modal").addEventListener("click", () => {
  const modal  = document.getElementById("pdf-modal");
  const iframe = document.getElementById("pdf-iframe");
  modal.classList.add("hidden");
  iframe.src = ""; // stop loading / free memory
});

document.getElementById("pdf-modal").addEventListener("click", e => {
  if (e.target === document.getElementById("pdf-modal")) {
    const iframe = document.getElementById("pdf-iframe");
    iframe.src = "";
    document.getElementById("pdf-modal").classList.add("hidden");
  }
});

// ── Populate doc selects ─────────────────────────────────────────────────────
function populateDocSelects() {
  // Summarize / Extract single selects
  ["sum-doc-select", "ext-doc-select"].forEach(id => {
    const sel = document.getElementById(id);
    sel.innerHTML = '<option value="">-- Chọn tài liệu --</option>' +
      documents.map(doc => `<option value="${doc.doc_id}">${doc.filename}</option>`).join("");
  });

  // Q&A chip-based multi-doc filter
  renderDocChips();
}

let selectedDocIds = new Set(); // empty = "all"

function renderDocChips() {
  const container = document.getElementById("doc-chips");
  const emptyMsg  = document.getElementById("doc-chips-empty");

  // Remove old doc chips (keep #chip-all and #doc-chips-empty)
  container.querySelectorAll(".doc-chip:not(#chip-all)").forEach(el => el.remove());

  if (!documents.length) {
    emptyMsg.classList.remove("d-none");
    updateFilterSummary();
    return;
  }

  emptyMsg.classList.add("d-none");

  documents.forEach(doc => {
    const chip = document.createElement("div");
    chip.className  = "doc-chip";
    chip.dataset.id = doc.doc_id;
    chip.title      = doc.filename;
    chip.innerHTML  = `
      <i class="bi bi-file-earmark-pdf chip-pdf-icon"></i>
      <span>${doc.filename}</span>
      <span class="ms-1 opacity-50" style="font-size:11px">${doc.num_pages}tr</span>`;
    chip.addEventListener("click", () => toggleDocChip(chip, doc.doc_id));
    container.appendChild(chip);
  });

  updateFilterSummary();
}

function toggleDocChip(chip, docId) {
  const allChip = document.getElementById("chip-all");

  if (selectedDocIds.has(docId)) {
    selectedDocIds.delete(docId);
    chip.classList.remove("active");
  } else {
    selectedDocIds.add(docId);
    chip.classList.add("active");
    allChip.classList.remove("active");
  }

  if (selectedDocIds.size === 0) {
    allChip.classList.add("active");
  }

  updateFilterSummary();
}

function selectAllDocs() {
  selectedDocIds.clear();
  document.querySelectorAll(".doc-chip:not(#chip-all)").forEach(c => c.classList.remove("active"));
  document.getElementById("chip-all").classList.add("active");
  updateFilterSummary();
}

function clearAllDocs() {
  selectedDocIds.clear();
  document.querySelectorAll(".doc-chip").forEach(c => c.classList.remove("active"));
  document.getElementById("chip-all").classList.remove("active");
  updateFilterSummary();
}

function getSelectedDocIds() {
  return [...selectedDocIds];
}

function updateFilterSummary() {
  const text = document.getElementById("doc-filter-summary-text");
  const n    = selectedDocIds.size;

  if (n === 0) {
    text.innerHTML = `Đang tìm trong <strong>tất cả ${documents.length} tài liệu</strong>`;
  } else if (n === 1) {
    const id  = [...selectedDocIds][0];
    const doc = documents.find(d => d.doc_id === id);
    text.innerHTML = `Đang tìm trong <strong>${doc ? doc.filename : id}</strong>`;
  } else {
    const names = [...selectedDocIds]
      .map(id => documents.find(d => d.doc_id === id)?.filename ?? id)
      .join(", ");
    text.innerHTML = `Đang tìm trong <strong>${n} tài liệu</strong>: ${names}`;
  }
}

// Chip-all click → select all
document.getElementById("chip-all").addEventListener("click", selectAllDocs);

// ── Q&A ──────────────────────────────────────────────────────────────────────
const chatBox = document.getElementById("chat-box");
const qaInput = document.getElementById("qa-input");

function appendMessage(role, text) {
  const el = document.createElement("div");
  el.className  = `chat-msg ${role}`;
  el.textContent = text;
  chatBox.appendChild(el);
  chatBox.scrollTop = chatBox.scrollHeight;
  return el;
}

document.getElementById("qa-send").addEventListener("click", sendQuestion);
qaInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuestion(); }
});

async function sendQuestion() {
  const question = qaInput.value.trim();
  if (!question) return;

  qaInput.value = "";
  appendMessage("user", question);
  qaConversation.push({ role: "user", content: question });

  const thinking = appendMessage("thinking", "Đang suy nghĩ...");
  const selectedIds = getSelectedDocIds(); // [] = all docs

  try {
    const r = await fetch(`${API_BASE}/api/qa`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation: qaConversation,
        conversation_type: "documentQA",
        doc_ids: selectedIds.length ? selectedIds : null,
      }),
    });
    const d = await r.json();
    chatBox.removeChild(thinking);

    if (r.ok) {
      appendMessage("assistant", d.answer);
      qaConversation.push({ role: "assistant", content: d.answer });
    } else {
      appendMessage("assistant", `❌ Lỗi: ${d.detail ?? JSON.stringify(d)}`);
      qaConversation.pop();
    }
  } catch (err) {
    chatBox.removeChild(thinking);
    appendMessage("assistant", `❌ Không kết nối được server: ${err.message}`);
    qaConversation.pop();
  }
}

// ── Summarize ────────────────────────────────────────────────────────────────
document.getElementById("sum-btn").addEventListener("click", async () => {
  const docId  = document.getElementById("sum-doc-select").value;
  const file   = document.getElementById("sum-file").files[0];
  const result = document.getElementById("sum-result");

  if (!docId && !file) {
    result.innerHTML = '<p class="text-danger">❌ Vui lòng chọn tài liệu hoặc upload file PDF.</p>';
    result.classList.remove("hidden"); return;
  }
  if (docId && file) {
    result.innerHTML = '<p class="text-danger">❌ Chỉ cung cấp một trong hai: tài liệu hoặc file.</p>';
    result.classList.remove("hidden"); return;
  }

  result.innerHTML = '<p class="text-primary"><span class="spinner-border spinner-border-sm me-2"></span>Đang tóm tắt...</p>';
  result.classList.remove("hidden");

  try {
    const form = new FormData();
    if (docId) form.append("doc_id", docId);
    if (file)  form.append("file", file);

    const r = await fetch(`${API_BASE}/api/summarize`, { method: "POST", body: form });
    const d = await r.json();

    if (r.ok) {
      result.innerHTML = `<h3>📝 Tóm tắt</h3><div class="result-text">${d.summary}</div>${buildMetaFooter(d.metadata)}`;
    } else {
      result.innerHTML = `<p class="text-danger">❌ ${d.detail ?? JSON.stringify(d)}</p>`;
    }
  } catch (err) {
    result.innerHTML = `<p class="text-danger">❌ Lỗi kết nối: ${err.message}</p>`;
  }
});

// ── Extract ──────────────────────────────────────────────────────────────────
document.getElementById("ext-btn").addEventListener("click", async () => {
  const query  = document.getElementById("extract-query").value.trim();
  const docId  = document.getElementById("ext-doc-select").value;
  const file   = document.getElementById("ext-file").files[0];
  const result = document.getElementById("ext-result");

  if (!query) {
    result.innerHTML = '<p class="text-danger">❌ Vui lòng nhập yêu cầu trích xuất.</p>';
    result.classList.remove("hidden"); return;
  }
  if (!docId && !file) {
    result.innerHTML = '<p class="text-danger">❌ Vui lòng chọn tài liệu hoặc upload file PDF.</p>';
    result.classList.remove("hidden"); return;
  }
  if (docId && file) {
    result.innerHTML = '<p class="text-danger">❌ Chỉ cung cấp một trong hai: tài liệu hoặc file.</p>';
    result.classList.remove("hidden"); return;
  }

  result.innerHTML = '<p class="text-primary"><span class="spinner-border spinner-border-sm me-2"></span>Đang trích xuất...</p>';
  result.classList.remove("hidden");

  try {
    const form = new FormData();
    form.append("request", JSON.stringify({ query }));
    if (docId) form.append("doc_id", docId);
    if (file)  form.append("file", file);

    const r = await fetch(`${API_BASE}/api/extract`, { method: "POST", body: form });
    const d = await r.json();

    if (r.ok) {
      const items = d.extracted_data ?? [];
      result.innerHTML = `
        <h3>🔍 Kết quả trích xuất (${items.length} mục)</h3>
        <ul class="extracted-list">${items.map(i => `<li>${i}</li>`).join("")}</ul>
        ${buildMetaFooter(d.metadata)}`;
    } else {
      result.innerHTML = `<p class="text-danger">❌ ${d.detail ?? JSON.stringify(d)}</p>`;
    }
  } catch (err) {
    result.innerHTML = `<p class="text-danger">❌ Lỗi kết nối: ${err.message}</p>`;
  }
});

// ── Init ─────────────────────────────────────────────────────────────────────
document.getElementById("filter-select-all").addEventListener("click", selectAllDocs);
document.getElementById("filter-clear-all").addEventListener("click", clearAllDocs);

document.getElementById("qa-clear-btn").addEventListener("click", () => {
  qaConversation = [];
  document.getElementById("chat-box").innerHTML = "";
});

document.getElementById("sum-clear-btn").addEventListener("click", () => {
  const r = document.getElementById("sum-result");
  r.innerHTML = "";
  r.classList.add("hidden");
});

document.getElementById("ext-clear-btn").addEventListener("click", () => {
  const r = document.getElementById("ext-result");
  r.innerHTML = "";
  r.classList.add("hidden");
});

document.getElementById("sum-file").addEventListener("change", e => {
  document.getElementById("sum-file-name").textContent = e.target.files[0]?.name ?? "Chưa chọn file";
});
document.getElementById("ext-file").addEventListener("change", e => {
  document.getElementById("ext-file-name").textContent = e.target.files[0]?.name ?? "Chưa chọn file";
});

checkHealth();
setInterval(checkHealth, 30000);
loadDocuments();
