# Báo cáo Part 0 — Nền tảng

Nhánh `feat/part0-foundation`. Báo cáo này dành cho chủ dự án: đã thay đổi gì, đo được gì, quyết định nào đã đưa ra, và cái gì chưa kiểm chứng.

## 1. Đã thay đổi gì

- **Môi trường tái lập được**: `pyproject.toml` + `uv.lock` (Python 3.11, khoá cho Linux), phần nặng (torch/CUDA) tách vào extra `ml`; pin lại `trl==0.11.4` như script huấn luyện gốc.
- **PostgreSQL là nguồn sự thật** cho catalog (món, size, nhóm tuỳ chọn, FAQ, chunk chính sách, cài đặt quán), dùng SQLAlchemy 2 async + Alembic. Seed từ CSV, chạy lại không tạo trùng.
- **Neo4j trở thành bản chiếu (projection)** của Postgres: `app/knowledge/graph_writer.py` dựng lại cùng hình dạng graph cũ (id `menu_0001`, `faq_0001`, `doc_0001`...). Món `sold_out`/`hidden` được chiếu thành `available=false` và retriever lọc bỏ.
- **Một lệnh bootstrap**: `uv run python -m scripts.bootstrap --reset` = migrate, seed, chiếu graph, tạo vector index, embed (21 món, 55 biến thể, 30 FAQ, 20 chunk).
- **LLM**: thêm `generate_structured()`, backend Claude (`AnthropicClient`, qua gateway OmniRoute, key đọc từ `ANTHROPIC_API_KEY`, thiếu key thì app dừng ngay lúc khởi động và nêu rõ tên biến), LLM giả có kịch bản cho test, SGLang giữ nguyên để chạy offline.
- **Hiệu năng/vận hành**: embedding và rerank chạy ngoài event loop (thread), fp16 trên GPU, warmup lúc khởi động (lỗi warmup chỉ cảnh báo, không làm app chết), `/health` có `postgres` và trả `degraded` thay vì lỗi 500.
- **Nội dung**: FAQ giao hàng cập nhật (quán tự giao, phí 15.000đ).
- **Hạ tầng dev/CI**: `docker-compose.yml` (postgres, neo4j, cặp dịch vụ test tách port + tmpfs, SGLang là profile tuỳ chọn), `Dockerfile`, `.env.example`, workflow GitHub Actions (unit + integration), các script kiểm tra cũ chuyển vào `scripts/checks/`.
- **Probe hội thoại**: `scripts/checks/conversation_probe.py` (nguyên văn các phiên a/b/c/d của bản đánh giá nền, thêm `--base-url`, `--only`).

## 2. Số đo

| Hạng mục | Kết quả | Nguồn |
|---|---|---|
| VRAM backend (không SGLang), embedding + reranker fp16 | khoảng 2,3 – 2,4 GB (đo lại ở Task 9: tiến trình uvicorn 2.518 MiB sau khi chạy probe) | Task 7, Task 9 |
| Warmup lúc khởi động | 15,2 s | Task 7 |
| `/chat` đầu tiên / thứ hai (sau warmup) | 2,19 s / 0,65 s | Task 7 |
| Claude qua OmniRoute: generate | khoảng 4,6 – 5 s | Task 6 |
| Claude qua OmniRoute: TTFT (stream) | khoảng 1,8 s | Task 6 |
| Claude qua OmniRoute: structured output | khoảng 2 s | Task 6 |
| Test | 24 unit pass; 10 integration pass (`pytest -m integration`, cần compose) | Task 9 |

### Kết quả probe (Task 9)

Tất cả 17 lượt của mỗi lần chạy trả HTTP 200, không có lỗi trong log backend.

**Intent** (giống hệt bản nền ở spec mục 2, cả hai backend): a = order x6; b = faq x4; c = consultant x3 rồi order; d = order x4 rồi ignore.

**Câu "Có giao hàng không?"** trả lời đúng nội dung mới (quán tự giao, 15.000đ), cả mock (395 ms) lẫn anthropic (2.452 ms).

**Mock** (`LLM__BACKEND=mock`): độ trễ mỗi lượt 0 – 790 ms (lượt chạy bằng state machine 0 ms).

**Anthropic** (`LLM__BACKEND=anthropic`, Claude opus qua OmniRoute): mọi lượt đi qua LLM mất khoảng 2,2 – 12,1 s (trung vị khoảng 5 s); lượt chạy state machine 0 – 1 ms. Chậm nhất: "Có gì rẻ hơn không?" 12,1 s và injection 11,8 s. Không có lỗi 500 và không có `LLMUnavailableError` trong lần chạy này (gateway ổn định); lỗi này vẫn chưa được bắt ở các nơi gọi (xem mục 6).

**Lỗi H1–H9 còn lại (đúng như dự kiến, Part 1 sẽ sửa)** quan sát được trong probe:

- H1: "thêm 2 trà vải size M" rồi "xem đơn"/"chốt đơn" chỉ còn 1 Bạc xỉu, trà vải biến mất không báo (cả hai backend).
- H2: "2 bánh chocolate giao ngày mai" vẫn bị hỏi size Bạc xỉu; câu prompt injection vẫn bị coi là đang đặt Bạc xỉu (mock).
- H3: "1 bạc xỉu và 2 trà vải" chỉ xử lý Bạc xỉu trước (mock báo giá chỉ Bạc xỉu).
- H5: "ít đá không đường" được bot nhắc lại như đã ghi nhận nhưng hệ thống không lưu tuỳ chỉnh.
- H6 (mock): "ít ngọt" và "rẻ hơn" trả lại đúng câu gợi ý cũ (vẫn có Chocolate Freeze).
- H7/H8: lời mời "Anh/chị cần gọi thêm gì nữa không ạ?" lặp sau mỗi lần thêm; giọng điệu chưa thống nhất giữa các câu.
- H9: với Claude, độ trễ 2 – 12 s mỗi lượt, chưa stream.
- Quan sát thêm với Claude: ở phiên b lượt 4 và phiên d, Claude tự "đính chính" thông tin cũ không có thật hoặc đổi số lượng (1 ly thành 2 ly rồi lại 1 ly). Đây là hậu quả của việc trạng thái order chưa chuẩn (H1/H2/H5), cần Part 1.
- H4 (lộ nhãn điều khiển nội bộ) không xuất hiện trong lần chạy này.

## 3. ADR 0001 – 0010

| ADR | Quyết định | Vì sao (ngắn) |
|---|---|---|
| 0001 | Ghi mọi quyết định thành ADR | Giữ lại lý do, nhẹ, tránh tranh luận lại |
| 0002 | PostgreSQL là nguồn sự thật, Neo4j là bản chiếu | Giao dịch ACID, ràng buộc ở DB; Neo4j xoá dựng lại được không mất dữ liệu |
| 0003 | uv + pyproject + lockfile | Cài đặt giống nhau mọi máy; extras tách phần GPU nặng |
| 0004 | SQLAlchemy async + Alembic | Khớp FastAPI, có kiểu, `alembic check` bắt lệch model/migration |
| 0005 | Giao diện LLM, Claude là mặc định | Một model opus đủ nhanh và đơn giản hơn nhiều model nhỏ; structured output thay parse JSON mong manh; giữ SGLang cho offline |
| 0006 | Chiến lược test (unit + integration + LLM giả) | Nhanh và tất định, integration bắt lỗi DB/Cypher thật; không dùng LLM thật làm cổng merge |
| 0007 | Topology docker compose | Một lệnh dựng hạ tầng; dịch vụ test tách riêng không xoá dữ liệu dev; SGLang là profile tuỳ chọn |
| 0008 | Nhóm tuỳ chọn và luật seed | Chỉ cà phê không tag nóng/đá cần hỏi nhiệt độ; giá nguyên VND theo size; tuỳ chọn là dữ liệu để admin chỉnh |
| 0009 | Đẩy ML chặn ra thread + warmup + fp16 | Event loop không bị khoá; nửa VRAM; request đầu của khách không trả giá khởi động |
| 0010 | Dựng lại projection Neo4j toàn bộ | Bật/tắt "tạm hết" rẻ và đảo ngược được; luôn khớp Postgres; dữ liệu nhỏ nên rẻ |

## 4. Đã bỏ qua / chưa kiểm chứng

- **Build image Docker của backend**: chưa kiểm chứng. Bước export các layer torch/CUDA mất hơn 50 phút trên máy này. Đường chạy dev (`uv run`) đã kiểm chứng; dịch vụ `backend` trong compose có thể chưa khởi động được cho tới khi image được làm gọn.
- **VRAM khi chạy chung LLM 7B cục bộ + embedding + reranker**: chưa đo vì bạn muốn GPU trống (không chạy SGLang). Chỉ đo được backend một mình. ADR 0009 ghi rõ là chưa đo.
- **CI chạy thật trên GitHub**: chưa chạy (chưa push). Các lệnh trong workflow được chạy tương đương ở local.
- **Probe với SGLang/Qwen 7B**: không chạy, vì cùng lý do GPU.

## 5. Các quyết định đã đưa ra

Mỗi dòng: quyết định, lý do, và chi phí nếu quyết định sai.

1. Mục `database:` trong YAML thêm ở Task 3 cùng schema của nó, không phải Task 2. Lý do: bộ nạp cấu hình cấm khoá lạ, thêm sớm sẽ làm hỏng mọi import. Chi phí nếu sai: không có.
2. Task 2 dừng (không xoá) container `neo4j-fnb` do controller tạo để nhường cổng 7474/7687 cho neo4j của compose. Lý do: dữ liệu dựng lại hoàn toàn được bằng bootstrap. Chi phí nếu sai: chạy lại bootstrap.
3. Làm việc trực tiếp trên nhánh `feat/part0-foundation` thay vì worktree riêng. Lý do: SGLang, venv, docker nằm trong thư mục này và dịch vụ gắn cổng cố định; nhánh đã cách ly khỏi `main`. Chi phí nếu sai: không có đối với `main`.
4. ADR và báo cáo cho người dùng viết bằng tiếng Việt (code, comment, commit message bằng tiếng Anh). Lý do: bạn đọc tiếng Việt và muốn thấy từng quyết định kèm lý do. Chi phí nếu sai: phải dịch lại.
5. Test `test_config` của Task 1 qua ngay không có bước RED (do kế hoạch bắt buộc). Lý do: test ghim hợp đồng cấu hình hiện có (`extra=forbid`), hành vi đã tồn tại nên không thể có RED, test vẫn kiểm tra hành vi thật. Chi phí nếu sai: không có.
6. Ghim `trl` về 0.11.4 (từ 0.29.1 do lock tự trôi) thay vì sửa script huấn luyện. Lý do: đó là phiên bản script được viết và chạy; chuyển đổi nằm ngoài phạm vi Part 0. Chi phí nếu sai: chưa dùng được tính năng trl mới tới khi nâng cấp có chủ đích.
7. `uv.lock` giới hạn cho Linux (`sys_platform=='linux'`). Lý do: cần để cài `trl==0.11.4` với numpy 2.2.1; dự án chỉ nhắm Linux + CUDA. Chi phí nếu sai: dev macOS/Windows không `uv sync` được, phải mở rộng lock sau.
8. Sửa Dockerfile không dùng `uv run --no-dev` lúc khởi động mà gọi thẳng binary trong venv. Lý do: `uv run` đồng bộ chính xác và sẽ gỡ extra `ml` (torch) khi container khởi động. Chi phí nếu sai: không có.
9. `.dockerignore` giữ lại `data/mock`. Lý do: bootstrap/seed đọc thư mục này bên trong container. Chi phí nếu sai: image lớn hơn một chút.
10. Món `hidden`/`sold_out` vẫn là node với `available=false` (không xoá), nhưng các truy vấn MenuItem của retriever phải lọc `coalesce(m.available,true)`. Lý do: bot không bao giờ gợi ý món không bán được, và giữ node tránh embed lại khi bật/tắt. Chi phí nếu sai: index phình nhẹ.
11. Truy cập Claude qua gateway OmniRoute của bạn: thêm `llm.anthropic.base_url` (YAML, không phải bí mật; rỗng = mặc định Anthropic), model `auto/claude-opus-5.5`, key đọc từ `ANTHROPIC_API_KEY`. Lý do: đã kiểm tra thật ngày 2026-11-01 (model trả về claude-opus-5-5, khoảng 2 s, structured output chạy được). Chi phí nếu sai: đổi model id/base_url trong YAML.
12. Đọc `ANTHROPIC_API_KEY` tường minh và truyền `api_key` + `base_url` trực tiếp cho `AsyncAnthropic`; bỏ `load_dotenv` lúc import. Lý do: tránh rò credential không liên quan sang gateway bên thứ ba. Chi phí nếu sai: không có.
13. Bước đo VRAM của Task 7 chạy không có SGLang. Lý do: LLM mặc định giờ là cloud và bạn muốn GPU trống. Chi phí nếu sai: chưa đo việc LLM 7B cục bộ chạy chung với các model (đã ghi trong ADR).
14. Warmup lỗi thì chỉ cảnh báo log và vẫn phục vụ (model nạp lười ở request đầu). Lý do: nguyên tắc "không bao giờ chết", giống reranker tự hạ cấp. Chi phí nếu sai: GPU cấu hình sai chỉ bị phát hiện ở request đầu thay vì lúc khởi động (log/health vẫn báo).
15. Image Docker của backend không build trong Task 9, ghi là chưa kiểm chứng và khuyến nghị làm gọn ở Part 1. Lý do: export layer torch/CUDA hơn 50 phút. Chi phí nếu sai: dịch vụ `backend` trong compose có thể chưa chạy được tới khi sửa; đường dev đã kiểm chứng.
16. Task 9 chạy probe với mock rồi với anthropic (OmniRoute), backend chỉ bật tạm rồi tắt, không chạy SGLang. Lý do: giữ GPU trống theo yêu cầu của bạn. Chi phí nếu sai: không có.

## 6. Việc chuyển sang Part 1

1. **Lớp fallback cho `LLMUnavailableError`** trước khi chuyển mặc định demo sang anthropic: hiện các nơi gọi chưa bắt lỗi này, nếu gateway sập thì request trả 500.
2. **`effort` / `max_tokens` theo từng lời gọi** khi model luôn bật thinking (hiện cấu hình chung cho mọi lời gọi; lượt đơn giản vẫn tốn 2 – 12 s).
3. **`price`/`total` của đơn hàng dùng BigInteger** (tránh tràn int32 khi tổng tiền lớn).
4. **Làm gọn image backend** (torch CPU hoặc multi-stage) để build được trong thời gian hợp lý.
5. Sửa các lỗi H1–H9 ở mục 2 (đặc biệt H1, H2, H5 vì làm Claude tự mâu thuẫn trong hội thoại).
