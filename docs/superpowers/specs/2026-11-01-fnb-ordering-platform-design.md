# Thiết kế: Nền tảng đặt món F&B có AI tư vấn — từ chatbot demo thành sản phẩm hoàn chỉnh

Ngày: 2026-11-01 · Trạng thái: bản nháp chờ duyệt
Nguồn: trao đổi với chủ dự án + `Ke_hoach_mo_rong_du_an_FnB_AI.docx` + đánh giá thực tế khi chạy hệ thống hiện tại.

Quy ước: **[Chủ dự án]** = quyết định của chủ dự án. **[Quyết định của người viết]** = lựa chọn mặc định do người viết đặt, chủ dự án chưa xác nhận riêng, có thể đổi.

---

## 1. Ý định và lý do

Dự án hiện tại là chatbot multi-agent + Graph RAG cho quán cà phê (dữ liệu Highlands giả lập). Nó trả lời được menu/FAQ/tư vấn và có giỏ hàng trong RAM, nhưng chưa tạo đơn thật, chưa có phía nhân viên, và trải nghiệm hội thoại còn nhiều lỗi.

**Người cần thuyết phục là một chủ quán thật [Chủ dự án]** — không phải người phỏng vấn kỹ thuật. Họ đánh giá bằng cảm giác sử dụng:

1. **Thao tác dễ**: khách gọi món, chọn size/tuỳ chỉnh, sửa đơn, chốt đơn mà không phải nghĩ cách nói cho bot hiểu.
2. **Bot nói chuyện như một nhân viên có kinh nghiệm**: lắng nghe, hiểu nhu cầu, gợi ý đúng lúc, tôn trọng quyết định của khách. Mục tiêu là giúp khách chọn được món phù hợp thoải mái và nhanh, **không phải bán được nhiều nhất** [Chủ dự án].

Kế hoạch trong file docx (PostgreSQL, tool calling, xác nhận trước khi tạo đơn, idempotency, phân quyền, audit, đánh giá) vẫn giữ — nhưng phục vụ trải nghiệm trên, không phải mục tiêu tự thân. Phần n8n/webhook ra ngoài bị bỏ (mục 4).

**Ưu tiên tuyệt đối [Chủ dự án]**: nếu chỉ kịp một phần, phần đó là **bot hỗ trợ khách đặt món mượt và trò chuyện/tư vấn tự nhiên**.

## 2. Hiện trạng (đo được khi chạy ngày 2026-11-01)

Chạy trên RTX 3060 12GB: Neo4j + BGE-M3 + reranker, cả `llm.backend: mock` lẫn Qwen2.5-7B-AWQ qua SGLang. Kịch bản thử ở `/tmp/fnb_probe.py` (nên đưa vào repo thành bộ đánh giá, mục 9).

Lỗi quan sát được — phần 1 phải xử lý hết:

| # | Hiện tượng | Ví dụ |
|---|---|---|
| H1 | Giỏ hàng làm rơi món đang chờ xác nhận | "thêm 2 trà vải size M" → "xem đơn"/"chốt đơn" → trà vải biến mất, không báo |
| H2 | Trạng thái order "dính" món cũ | "Tôi cần 2 bánh chocolate giao ngày mai" → bot hỏi size Bạc xỉu; câu prompt injection bị coi là đang đặt Bạc xỉu |
| H3 | Một câu nhiều món chỉ lấy một món; LLM gộp nhầm giá | "1 bạc xỉu và 2 trà vải" → báo giá Bạc xỉu cho cả hai |
| H4 | LLM để lộ nhãn điều khiển nội bộ | `CONFIRM "Dạ, em ghi nhận..."`, `(ASK_SIZE) "..."` |
| H5 | Bot làm như đã ghi tuỳ chỉnh nhưng hệ thống không lưu | "bạc xỉu ít đá không đường size M" |
| H6 | Tư vấn bỏ qua tiêu chí mới / cache trả câu cũ | (mock) "ít ngọt" vẫn gợi ý Chocolate Freeze; "rẻ hơn" trả y câu trước từ cache trong 25ms |
| H7 | Giọng không nhất quán | lúc "em", lúc "chúng tôi/Bạn" |
| H8 | Mời thêm món lặp lại sau mỗi lần thêm | "Anh/chị cần gọi thêm gì nữa không ạ?" — đúng điều chủ dự án ghét nhất |
| H9 | Chậm | 3–9s/lượt với Qwen 7B local, concurrency 1, UI không stream thật; cold start 35–39s |
| H10 | Vận hành | `tests/` không có test pytest nào chạy; thư viện ML không ghim version (datasets/pyarrow/huggingface_hub vỡ); vector index phải tạo tay; 7B + BGE-M3 + reranker fp32 không vừa 12GB như README nói |

Điểm mạnh giữ lại: cấu hình YAML có schema, Graph RAG (keyword + vector + mở rộng quan hệ + reranker), router nhiều tầng có fallback, FAQ chính xác, chặn món ngoài menu, tầng dialogue state tất định trước router, giá/tổng tiền do code tính.

## 3. Mục tiêu

### Phía khách (phần 1 — trọng tâm)
- Một **ứng dụng web duy nhất, responsive** cho cả ba kiểu khách [Chủ dự án]:
  - **QR tại bàn**: mã QR mang sẵn số bàn.
  - **Quầy / kiosk**: màn hình lớn, khách nhận **số thứ tự**.
  - **Online**: **mang đi** (tới lấy) hoặc **giao hàng**; nhập tên + số điện thoại, giao hàng thêm địa chỉ; có thể **hẹn giờ** (vd. ngày mai 9h).
  - Màn hình đầu chọn "Tại bàn / Mang đi / Giao hàng" (QR tại bàn/kiosk tự điền chế độ). Phần trò chuyện dùng chung; chỉ bước chốt đơn hỏi thông tin khác nhau.
- Khách **gõ hoặc bấm**: thẻ món (hình, giá theo size), nút chọn size, nút tuỳ chỉnh, giỏ hàng luôn nhìn thấy, nút "Thêm món"/"Chốt đơn". Bấm và gõ cùng sửa **một giỏ hàng duy nhất**.
- **Tuỳ chỉnh món bằng lựa chọn cố định để bấm** [Chủ dự án], lưu theo từng dòng trong đơn và hiện cho nhân viên pha chế. Bot hiểu được khi khách gõ ("ít đá, không đường") và chuyển thành lựa chọn tương ứng.
  - [Quyết định của người viết] Mặc định: Đường: không / ít / bình thường / nhiều. Đá: không / ít / bình thường. Nóng/đá chỉ hiện với món có cả hai dạng. Không tính thêm tiền. Admin cấu hình nhóm lựa chọn nào áp cho món nào (bánh: không có).
- **Món "tạm hết"**: là một trạng thái của món; khi bật thì món **không đặt được** [Chủ dự án]. Bot không gợi ý món đó, khách gọi đúng món đó thì bot nói rõ và gợi ý món tương tự. [Quyết định của người viết] Món đang trong giỏ chưa chốt mà bị tạm hết thì bot báo và gợi ý đổi; đơn đã chốt không tự đổi gì. Tạm hết áp cho cả món, không theo size.
- **Thanh toán**: không xử lý thanh toán thật. [Quyết định của người viết] Khách chọn cách trả (tiền mặt / chuyển khoản–MoMo / trả tại quầy) lúc chốt; nhân viên bấm "Đã thanh toán".
- **Danh tính khách**: [Quyết định của người viết] khách không có tài khoản. Sau khi chốt, khách nhận một **đường link riêng** (token khó đoán, lưu trên trình duyệt) để xem đơn của mình. Không ai xem được đơn của người khác.
- Sau khi chốt: **thẻ trạng thái đơn tự cập nhật** trong khung chat (đã nhận → đang pha → sẵn sàng…) [Chủ dự án]; khách cũng hỏi bot "đơn tới đâu rồi" được.

### Cách bot tư vấn — nguyên tắc từ chủ dự án [Chủ dự án]
Đây là tiêu chí nghiệm thu chính của phần 1:

1. **Hỏi trước, gợi ý sau.** Khi khách phân vân, không liệt kê một loạt món. Hỏi câu đơn giản: cà phê hay trà trái cây? ngọt nhiều hay ít? thanh mát hay béo? (ngân sách nếu khách nhắc).
2. **Hỏi vừa đủ**: mỗi lượt **một câu hỏi ngắn**, dựa trên câu trả lời trước.
3. **Gợi ý có lý do**: hiểu đủ rồi thì gợi ý **1–2 món**, mỗi món một lý do ngắn gắn với điều khách vừa nói. Dựa trên khẩu vị/ngân sách/nhu cầu, **không dựa vào món bán chạy**.
4. Khách vẫn phân vân → hỏi thêm **một** câu hoặc đưa **một lựa chọn an toàn**.
5. **Không hỏi lại thông tin đã biết**: khẩu vị/ràng buộc khách đã nói (ít đường, không cà phê, ngân sách, món đã từ chối) được nhớ suốt cuộc trò chuyện và áp vào mọi gợi ý sau — kể cả mặc định tuỳ chỉnh khi thêm món.
6. **Biết dừng**: khách chọn xong → xác nhận món, **không mời thêm** topping/bánh/món khác. Khách đã từ chối → không mời lại. Không có câu "Anh/chị cần gọi thêm gì nữa không ạ?" lặp lại; thay bằng nút "Thêm món"/"Chốt đơn".
7. **Giọng**: lịch sự, tự nhiên, vừa phải; xưng "em", gọi "anh/chị" nhất quán; không chào/cảm ơn mỗi câu; emoji rất hạn chế.

Ngoài ra (rút từ H1–H8): bot không bao giờ báo sai giá/size/tình trạng món; không để lộ nhãn nội bộ; không tuyên bố đã ghi điều hệ thống không lưu; xử lý được nhiều món trong một câu; không bị "dính" món cũ khi khách đổi chủ đề; nội dung kiểu prompt injection không làm thay đổi đơn hay chính sách.

### Phía nhân viên (phần 2)
- **Màn hình hàng đợi** các đơn hiện tại, **thông báo** khi có đơn mới (âm thanh + nổi bật) [Chủ dự án].
- **Thứ tự theo điểm ưu tiên**, khách tại quán (quầy, QR tại bàn) ưu tiên hơn online [Chủ dự án].
  - [Quyết định của người viết, chủ dự án đồng ý hướng] Điểm = trọng số theo kênh + điểm tăng dần theo thời gian chờ (chống đơn online bị đẩy xuống mãi). Trọng số chỉnh trong admin.
  - Đơn hẹn giờ chỉ vào hàng đợi trước giờ nhận một khoảng chuẩn bị (mặc định 20 phút, chỉnh được).
  - Nhân viên có thể kéo một đơn lên đầu bằng tay.
- **Nhận đơn không trùng** [Chủ dự án]: bấm "Nhận đơn" → đơn gắn tên người đó và chuyển "Đang làm"; hai người bấm gần như cùng lúc thì chỉ người trước thành công (kiểm tra nguyên tử phía server), người sau thấy "Đơn đã được [tên] nhận"; mọi màn hình cập nhật tức thì. [Quyết định của người viết] Có "Trả lại hàng đợi"; admin chuyển đơn cho người khác.
- **Vòng đời đơn** [Quyết định của người viết, chủ dự án đồng ý]: `Chờ xác nhận` (bản nháp) → `Mới` → `Đang làm` → `Sẵn sàng` → (`Đang giao` — chỉ giao hàng) → `Hoàn tất`; nhánh `Đã huỷ`. Server chặn chuyển sai bước.
- Bấm "Tạm hết"/"Có lại" cho món ngay trên màn hình nhân viên.

### Trang quản lý cho chủ quán (phần 3)
- Thêm/sửa/ẩn món, giá theo size, hình, mô tả, danh mục, tag khẩu vị, nhóm tuỳ chỉnh [Chủ dự án đồng ý].
- Sửa thông tin quán, FAQ, chính sách.
- Chỉnh trọng số ưu tiên hàng đợi, thời gian chuẩn bị đơn hẹn giờ.
- Quản lý tài khoản nhân viên.
- Thay đổi phải tới bot **ngay**, không cần khởi động lại (mục 6.3).

### Báo cáo (phần 4)
- Cuối ngày hệ thống tự tạo báo cáo, hiện **thông báo trong app**; **không gửi gì ra ngoài web/app** [Chủ dự án].
- Số liệu do code tính từ database: doanh thu, số đơn, món bán chạy, doanh thu theo kênh, giờ cao điểm, thời gian chờ trung bình. [Quyết định của người viết, suy từ chữ "đánh giá"] Kèm một đoạn nhận xét ngắn do LLM viết chỉ dựa trên đúng các số liệu đó. Xem lại các ngày trước.

## 4. Ngoài phạm vi và điều cấm

**Ngoài phạm vi**: thanh toán thật; gửi thông báo/báo cáo ra ngoài (n8n, Telegram, Zalo, email); POS/ERP; nhiều chi nhánh; đếm tồn kho nguyên liệu; tài khoản khách / tích điểm; khẳng định "production-ready".

**Điều cấm (dù chạy đúng kỹ thuật vẫn coi là thất bại)**:
- Bot cố bán thêm, mời lại sau khi khách từ chối, hoặc lặp câu mời thêm món.
- Bot hỏi lại điều khách đã nói.
- Báo sai giá, gợi ý/nhận đơn món đang tạm hết, tự chọn size giùm khách.
- Lộ nhãn/prompt nội bộ cho khách.
- Hai nhân viên cùng làm một đơn; khách xem được đơn người khác; một lần chốt tạo ra hai đơn.
- Số liệu báo cáo hoặc số liệu đánh giá bịa / không tái lập được.

## 5. Ràng buộc

- **Dữ liệu**: giữ menu/FAQ/chính sách Highlands giả lập [Chủ dự án]. Hình món: ảnh minh hoạ/placeholder, thay được trong admin.
- **Ngôn ngữ**: [Quyết định của người viết] giao diện tiếng Việt; bot trả lời theo ngôn ngữ khách dùng (hệ thống hiện đã có dữ liệu Vi + En).
- **Công nghệ** [Chủ dự án giao người viết chọn, đã đồng ý]: **Next.js (React)** cho giao diện; **giữ FastAPI**; **PostgreSQL** cho dữ liệu giao dịch; **WebSocket** cho cập nhật tức thì; **Docker Compose** bật cả hệ thống bằng một lệnh. Giữ Neo4j + Graph RAG cho tư vấn/FAQ. Chainlit bị thay khi UI mới chạy được.
- **LLM** [Chủ dự án chấp nhận cloud]: một lớp nhà cung cấp chung; **cloud là mặc định khi demo**, **local SGLang/Qwen vẫn chạy được** bằng cấu hình. Dù dùng cloud, giá/size/tình trạng món/tổng tiền vẫn do code lấy từ database. Embedding và reranker chạy local. Nhà cung cấp cloud: xem mục 10.
- **Phần cứng demo**: máy có RTX 3060 12GB; khi dùng cloud LLM, GPU chỉ cần cho embedding/reranker (hoặc chạy CPU).

## 6. Cách làm đã chốt

### 6.1 Nguồn sự thật
- **PostgreSQL là nguồn sự thật** cho món, biến thể size + giá, nhóm tuỳ chỉnh, trạng thái món (đang bán / tạm hết / ẩn), FAQ, chính sách, thông tin quán, giỏ hàng nháp, đơn, dòng đơn, sự kiện đơn, tài khoản, cấu hình hàng đợi, báo cáo, audit log.
- **Neo4j là bản sao phục vụ tìm kiếm/tư vấn** (tách dữ liệu giao dịch khỏi knowledge graph — đúng quyết định mở trong docx mục 11).
- Bot đọc tình trạng món và giá **từ PostgreSQL tại thời điểm trả lời**, không từ RAG hay cache.
- Giỏ hàng nháp lưu ở PostgreSQL theo phiên khách (không chỉ trong RAM) để UI-bấm và chat dùng chung, không mất khi backend khởi động lại.

### 6.2 Tách "quyết định" và "diễn đạt"
- Mọi thao tác thay đổi giỏ/đơn đi qua các **thao tác có kiểu, kiểm tra ở server** (tương đương bộ công cụ trong docx: tìm món, kiểm tra còn bán, thêm/sửa/bỏ dòng, cập nhật khẩu vị, xem giỏ, yêu cầu chốt, tạo đơn, tra trạng thái đơn). Server kiểm tra món tồn tại, đang bán, size hợp lệ, tuỳ chỉnh hợp lệ, số lượng hợp lệ ([Quyết định của người viết] 1–20/dòng).
- **Tạo đơn chỉ xảy ra sau khi khách xác nhận rõ ràng** trên bản tóm tắt đơn (nút xác nhận, hoặc đồng ý rõ ràng với chính bản tóm tắt đó); có **khoá idempotency** để bấm hai lần / retry không tạo hai đơn.
- LLM lo hiểu câu khách và diễn đạt câu trả lời. Câu trả lời gửi tới khách **không chứa nhãn nội bộ**; phần có cấu trúc (thẻ món, giỏ, nút) do UI dựng từ dữ liệu server trả về, không do LLM viết ra.
- [Quyết định của người viết] **Chính sách hội thoại do code điều khiển**: code quyết định bước tiếp theo (hỏi câu nào về khẩu vị, khi nào đủ để gợi ý, gợi ý 1–2 món nào từ dữ liệu, khi nào dừng / không mời thêm, khi nào hỏi size/tuỳ chỉnh còn thiếu). LLM chỉ (a) hiểu câu khách thành dữ liệu có cấu trúc và (b) diễn đạt bước code đã chọn. Lý do: các quy tắc mục 3 phải đúng với cả LLM local 7B, không chỉ cloud.
- Hồ sơ khẩu vị của phiên (loại đồ uống, độ ngọt, đá, nóng/lạnh, ngân sách, món đã từ chối, đã từ chối mời thêm) được lưu có cấu trúc và **dùng để lọc/sắp xếp truy vấn** (vd. lọc giá trong Cypher/SQL), không chỉ nhét vào prompt.

### 6.3 Đồng bộ khi admin sửa
Sửa trong admin → cập nhật Neo4j (node + quan hệ), tính lại embedding cho phần đổi, xoá cache liên quan, làm mới từ vựng menu đang cache trong process. Bot dùng giá/trạng thái mới ngay ở lượt kế tiếp.

### 6.4 Cập nhật tức thì
WebSocket đẩy: đơn mới / đổi trạng thái / nhận đơn tới màn hình nhân viên; trạng thái đơn tới khách (theo token đơn); món tạm hết tới các phiên khách đang mở; thông báo báo cáo tới admin.

### 6.5 Bảo mật và tin cậy (từ docx, giữ nguyên)
Đăng nhập nhân viên/admin; phân quyền staff/admin (khách = token đơn); secret qua biến môi trường; không ghi số điện thoại/địa chỉ đầy đủ vào log; kiểm tra đầu vào bằng Pydantic; máy trạng thái đơn ở server; idempotency; retry có giới hạn; audit log cho mọi thay đổi đơn/món/cấu hình.

## 7. Chia phần và thứ tự

Một tài liệu thiết kế chung; lập kế hoạch và làm **theo từng phần**, phần sau bắt đầu khi phần trước đã demo được.

| Phần | Nội dung | Xong khi |
|---|---|---|
| **0. Nền móng** | Ghim version thư viện; pytest thật chạy được; Docker Compose (Postgres, Neo4j, backend, frontend; SGLang là profile tuỳ chọn); tạo vector index tự động; schema Postgres + migration + nạp dữ liệu Highlands vào Postgres rồi đồng bộ sang Neo4j; lớp LLM cloud/local | `docker compose up` + một lệnh nạp dữ liệu là chạy; test xanh |
| **1. Khách đặt món + tư vấn** (ưu tiên) | Web khách (3 chế độ), chat stream thật, thẻ món/nút/giỏ, tuỳ chỉnh, tạm hết (bật/tắt tạm qua API/seed), đơn lưu Postgres khi xác nhận, link theo dõi đơn, bot theo nguyên tắc mục 3; sửa H1–H9 | Bộ kịch bản đánh giá (mục 9) đạt; demo kịch bản mục 8 trơn tru |
| **2. Nhân viên** | Đăng nhập, hàng đợi ưu tiên, thông báo, nhận đơn nguyên tử, chuyển trạng thái, trả lại/chuyển đơn, nút tạm hết; khách thấy trạng thái tức thì | Hai trình duyệt nhân viên cùng bấm nhận một đơn → chỉ một thành công |
| **3. Admin** | CRUD món/giá/hình/tuỳ chỉnh/FAQ/thông tin quán, cấu hình hàng đợi, tài khoản nhân viên, đồng bộ sang bot | Đổi giá trong admin → bot báo giá mới ngay lượt sau |
| **4. Báo cáo** | Báo cáo cuối ngày + nhận xét AI + thông báo trong app + lịch sử | Báo cáo khớp số liệu tính tay trên dữ liệu mẫu |

## 8. Kịch bản demo cho chủ quán

1. Khách quét QR bàn 5 → "chưa biết uống gì" → bot hỏi từng câu ngắn → gợi ý 1–2 món có lý do → khách chọn, bấm size, bấm "ít đường" → xác nhận, **không bị mời thêm**.
2. Khách gõ "thêm 1 bạc xỉu và 2 trà vải size M, ít đá" → giỏ có đủ hai dòng đúng giá, bot chỉ hỏi phần còn thiếu (size bạc xỉu).
3. Khách gọi món đang tạm hết → bot nói rõ, gợi ý món tương tự.
4. Khách chốt → bản tóm tắt → xác nhận → bấm hai lần vẫn chỉ một đơn → thẻ trạng thái xuất hiện.
5. (Phần 2) Màn hình nhân viên kêu đơn mới; đơn bàn 5 xếp trên một đơn online đặt trước đó; nhân viên A nhận, B bấm sau bị từ chối; chuyển "Sẵn sàng" → điện thoại khách cập nhật.
6. Khách online: "Tôi cần 2 bánh tiramisu giao ngày mai 9h" → bot hỏi tên/SĐT/địa chỉ → đơn hẹn giờ không vào hàng đợi cho tới gần giờ.
7. (Phần 3–4) Chủ quán đổi giá một món → bot báo giá mới; mở báo cáo cuối ngày.

Ca lỗi phải xử lý đúng (từ docx + H1–H8): món tạm hết; thiếu thông tin; số lượng không hợp lệ; tạo đơn hai lần; prompt injection trong câu khách hoặc nội dung FAQ; khách mở link đơn người khác / không có token; nhân viên không đủ quyền.

## 9. Đánh giá

- **Bộ kịch bản hội thoại cố định** trong repo (mở rộng từ kịch bản đã chạy ở mục 2), chạy được tự động, mỗi kịch bản có kiểm tra tất định: trạng thái giỏ/đơn cuối cùng, không có nhãn nội bộ trong câu trả lời, số câu hỏi mỗi lượt ≤ 1, không có câu mời thêm sau khi khách chọn xong/từ chối, không hỏi lại khẩu vị đã biết, giá đúng.
- Chỉ số (từ docx): tỉ lệ hoàn thành tác vụ, thao tác không hợp lệ bị server từ chối, tuân thủ xác nhận trước khi tạo đơn, latency p50/p95, lỗi API, tỉ lệ đơn trùng. Báo cáo kèm cấu hình đo; không đặt số liệu giả.
- [Quyết định của người viết] "Đạt": các kiểm tra tất định (giỏ/đơn, giá, không nhãn nội bộ, không tạo trùng, xác nhận trước khi tạo đơn) đạt 100% mọi lần chạy trên **cả** backend cloud và local. Kiểm tra về giọng/độ tự nhiên (chấm bằng LLM-judge) đạt ≥ 90% qua 3 lần chạy, bắt buộc trên cloud, chỉ báo cáo (không bắt buộc) trên local.
- [Quyết định của người viết] Mục tiêu latency khi dùng cloud: thao tác bấm nút (giỏ/size/tuỳ chỉnh) không chờ LLM; chữ đầu tiên của câu trả lời chat hiện trong khoảng ~1.5s ở p50.

## 10. Quyết định sau vòng review (chủ dự án giao người viết quyết, 2026-11-01)

Chủ dự án: **không có hạn chót; làm chuẩn chỉnh theo góc nhìn senior dev + kỹ sư LLM/NLP/agent** [Chủ dự án]. Vì vậy lập kế hoạch chi tiết cho phần 0–1 trước; phần 2–4 được lập kế hoạch riêng khi phần trước đã demo được (thiết kế chung vẫn là tài liệu này). Các câu còn lại được giao cho người viết:

- **LLM cloud** [Quyết định của người viết]: Anthropic Claude qua SDK chính thức `anthropic` (async); model mặc định `claude-opus-5-5`, đổi qua cấu hình (`llm.model`). Dùng structured outputs cho bước hiểu câu (NLU → JSON đúng schema), streaming cho câu trả lời, prompt caching cho phần system prompt cố định; chỉnh `effort` theo từng loại lệnh gọi (NLU/diễn đạt thấp, báo cáo cao hơn) và đo latency thực tế. Backend local giữ client SGLang (OpenAI-compatible) hiện có; hai backend nằm sau cùng một interface. API key do chủ dự án cung cấp qua biến môi trường (`ANTHROPIC_API_KEY` hoặc profile `ant auth login`); không có key thì chạy local.
- **Giao hàng và đặt trước** [Quyết định của người viết]: được phép **viết lại các mục FAQ/chính sách Highlands mâu thuẫn** (faq_0010, faq_0011 chỉ giao qua app; faq_0028 không nhận đặt trước) cho khớp sản phẩm. Quy tắc: quán tự giao; **phí giao cố định** (mặc định 15.000đ, chỉnh trong admin) cộng vào tổng và hiện trên bản tóm tắt; không giới hạn khu vực / đơn tối thiểu khi demo; hẹn trước tối đa 2 ngày, giờ nhận phải nằm trong 7:00–22:00; nhân viên chuyển `Đang giao` → `Hoàn tất`.
- **Môi trường demo** [Quyết định của người viết]: mặc định chạy trên máy có RTX 3060, điện thoại/tablet/laptop cùng mạng LAN quét QR vào thẳng máy đó. Chuẩn bị sẵn cấu hình tunnel HTTPS (để chủ quán mở từ xa và để trình duyệt điện thoại cho phép âm thanh/WebSocket ổn định). URL gốc trong mã QR lấy từ cấu hình. Kiosk và màn hình nhân viên là thiết bị riêng mở cùng web.
- **Sau khi chốt đơn** [Quyết định của người viết]: khách tự huỷ được khi đơn còn `Mới` (chưa ai nhận); sau đó chỉ nhân viên/admin huỷ (bắt buộc ghi lý do, vào audit log). Không sửa đơn đã chốt — muốn đổi thì huỷ và đặt lại. Bàn gọi thêm lần nữa → **đơn mới** gắn cùng số bàn, không gộp hoá đơn; màn hình nhân viên nhóm các đơn cùng bàn cạnh nhau.

## 11. Báo cáo quyết định [Chủ dự án]

Mọi quyết định thiết kế hoặc kỹ thuật phát sinh trong lúc làm (kể cả những mục ở phần 12 "để người làm quyết định") đều phải **báo lại cho chủ dự án kèm lý do** và các phương án đã cân nhắc. Ghi lại thành ADR ngắn trong `docs/adr/` (một file mỗi quyết định: bối cảnh, quyết định, lý do, hệ quả) và tóm tắt trong báo cáo cuối mỗi task/phần. Quyết định làm thay đổi hành vi khách/nhân viên/chủ quán nhìn thấy thì hỏi trước khi làm.

## 12. Để người làm quyết định (vẫn phải báo lại theo mục 11)

Cấu trúc thư mục frontend, thư viện component, ORM/migration, thư viện xác thực; giữ hay gộp các agent hiện có (router/consultant/FAQ/order) miễn đạt nguyên tắc mục 3 và ràng buộc mục 6; dùng tool calling gốc của LLM hay hành động JSON có kiểu (phải chạy được với cả backend cloud và local); chiến lược cache (không được làm hỏng câu tinh chỉnh tiêu chí — H6); chi tiết công thức ưu tiên ngoài các ràng buộc ở mục 3; cách thông báo âm thanh trên trình duyệt.
