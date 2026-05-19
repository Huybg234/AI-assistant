# UI - AI Document Assistant

Giao diện web đơn giản để sử dụng **AI Document Assistant API**.

## Cách chạy

### Bước 1: Khởi động API backend
```bash
# Từ thư mục gốc D:\AI-assistant
uvicorn src.main:app --reload --port 8000
```

### Bước 2: Mở UI
Mở file `ui/index.html` trực tiếp trong trình duyệt, hoặc dùng server tĩnh:

```bash
# Python (từ thư mục ui/)
cd ui
python -m http.server 3000
# Truy cập: http://localhost:3000
```

> ⚠️ Đảm bảo API đang chạy tại `http://localhost:8000` trước khi mở UI.

## Tính năng

| Tab | Chức năng |
|-----|-----------|
| ⬆️ Upload PDF | Tải lên và ingest file PDF vào vector database |
| 📚 Tài liệu | Xem danh sách, chi tiết nội dung, và xóa tài liệu |
| 💬 Hỏi & Đáp | Chat hỏi đáp với tài liệu (RAG), hỗ trợ lịch sử hội thoại |
| 📝 Tóm tắt | Tóm tắt tài liệu đã ingest hoặc upload PDF mới |
| 🔍 Trích xuất | Trích xuất thông tin theo yêu cầu cụ thể |

## Cấu trúc file

```
ui/
├── index.html   # Giao diện chính
├── style.css    # Stylesheet
├── app.js       # Logic JavaScript
└── README.md    # Tài liệu này
```

## Thay đổi URL API

Nếu API chạy ở địa chỉ khác, sửa dòng đầu trong `app.js`:
```js
const API_BASE = "http://localhost:8000";
```
