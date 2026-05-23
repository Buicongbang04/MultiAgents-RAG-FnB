# MultiAgents RAG FnB

Robot tư vấn F&B (Highlands Coffee) sử dụng Multi-Agent + Graph RAG hoàn toàn Local.

## Thử ngay

| Intent | Ví dụ |
|--------|-------|
| **Đặt hàng** | "Cho anh 1 bạc xỉu đá size L" |
| **Tư vấn** | "Có gì ngon rẻ không?" |
| **FAQ** | "Wifi quán là gì?", "Mấy giờ đóng cửa?" |
| **Ignore** | "Hôm nay thời tiết thế nào?" |

## Luồng xử lý

```
Query → Router (Qwen2.5) → Agent → Graph RAG (Neo4j) → Generator → Response
```
