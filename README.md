# MultiAgents-RAG-FnB
# MultiAgents-RAG-FnB

Hệ thống chatbot tư vấn cho nhà hàng (Food & Beverage) sử dụng kiến trúc **Multi-Agent RAG (Retrieval-Augmented Generation)** với các thành phần:

* **Router Agent**: phân loại câu hỏi người dùng.
* **Consultant Agent**: truy vấn menu và tài liệu từ Neo4j.
* **FAQ Agent**: trả lời các câu hỏi thường gặp.
* **Ignore Agent**: xử lý các câu hỏi ngoài phạm vi.
* **Session Store**: lưu lịch sử hội thoại.
* **Neo4j Knowledge Graph**: lưu trữ dữ liệu menu, FAQ và tài liệu.
* **FastAPI**: cung cấp REST API.

---

# Clone Repository

```bash
git clone https://github.com/Buicongbang04/MultiAgents-RAG-FnB.git
cd MultiAgents-RAG-FnB
```

---

# Cài đặt Dependencies

## Tạo môi trường Python

```bash
conda create -n fiai python=3.11 -y
conda activate fiai
```

## Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

# Chạy Hệ Thống

## Khởi động SGLang

```bash
bash scripts/SGLang_Server.sh
```

## Chạy APP

```bash
bash scripts/restart_system.sh
```

## Truy cập

* API: [http://localhost:8001](http://localhost:8001)
* Swagger UI: [http://localhost:8001/docs](http://localhost:8001/docs)
* Neo4j Browser: [http://localhost:7474](http://localhost:7474)

---

# Kiến Trúc Hệ Thống

```text
User
  ↓
FastAPI (/chat)
  ↓
Router Agent
  ├── Consultant Agent → Neo4j
  ├── FAQ Agent → Neo4j
  └── Ignore Agent
  ↓
Session Store
  ↓
Response
```

---

# Luồng Hoạt Động

1. Người dùng gửi câu hỏi tới API `/chat`.
2. FastAPI nhận request và chuyển đến Router Agent.
3. Router Agent phân loại câu hỏi thành:

   * Consultant
   * FAQ
   * Ignore
4. Agent tương ứng xử lý câu hỏi.
5. Nếu cần, Agent truy vấn dữ liệu từ Neo4j.
6. Kết quả được lưu vào Session Store.
7. FastAPI trả phản hồi cho người dùng.

---

# Quy Trình Hoạt Động

## 1. Router Agent

Phân loại câu hỏi bằng mô hình fine-tuned để xác định đúng tác vụ.

Ví dụ:

* "Quán có phở bò không?" → Consultant
* "Quán mở cửa lúc mấy giờ?" → FAQ
* "Hôm nay thời tiết thế nào?" → Ignore

## 2. Consultant Agent

Truy xuất:

* Menu món ăn
* Giá cả
* Mô tả món
* Tài liệu hỗ trợ tư vấn

## 3. FAQ Agent

Trả lời các câu hỏi như:

* Giờ mở cửa
* Chính sách giao hàng
* Hình thức thanh toán

## 4. Ignore Agent

Từ chối lịch sự các câu hỏi ngoài phạm vi hệ thống.

---

# Ví Dụ Hội Thoại

## Ví dụ 1: Tư vấn menu

**User:**

> Quán có món cà phê sữa đá không?

**System:**

> Dạ quán hiện có món Cà phê sữa đá với giá 35.000 VNĐ.

---

## Ví dụ 2: Câu hỏi FAQ

**User:**

> Quán mở cửa lúc mấy giờ?

**System:**

> Dạ quán mở cửa từ 7:00 sáng đến 10:00 tối mỗi ngày.

---

## Ví dụ 3: Câu hỏi ngoài phạm vi

**User:**

> Hôm nay thời tiết ở TP.HCM thế nào?

**System:**

> Xin lỗi, tôi chỉ hỗ trợ tư vấn thông tin liên quan đến nhà hàng.

---

# Công Nghệ Sử Dụng

* FastAPI
* Neo4j
* Docker
* OpenAI API
* Pytest

---

# Tác Giả

**Bùi Công Bằng**

GitHub: [https://github.com/Buicongbang04](https://github.com/Buicongbang04)
