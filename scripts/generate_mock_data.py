import csv
import json
import random
from pathlib import Path


OUT_DIR = Path("data/mock")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MENU_CSV = OUT_DIR / "menu.csv"
FAQ_CSV = OUT_DIR / "faq.csv"
DOCS_JSONL = OUT_DIR / "docs.jsonl"

DRINK_BASES = [
    ("Bạc xỉu", "Bac Xiu", "coffee", ["coffee", "milk", "sweet"]),
    ("Cà phê sữa đá", "Iced Milk Coffee", "coffee", ["coffee", "milk", "iced"]),
    ("Cà phê đen đá", "Iced Black Coffee", "coffee", ["coffee", "strong", "iced"]),
    ("Latte", "Latte", "coffee", ["coffee", "milk", "soft"]),
    ("Cappuccino", "Cappuccino", "coffee", ["coffee", "milk foam"]),
    ("Trà sen vàng", "Golden Lotus Tea", "tea", ["tea", "lotus", "fresh"]),
    ("Trà đào cam sả", "Peach Lemongrass Tea", "tea", ["tea", "peach", "lemongrass"]),
    ("Trà vải", "Lychee Tea", "tea", ["tea", "lychee", "sweet"]),
    ("Freeze trà xanh", "Green Tea Freeze", "freeze", ["green tea", "ice blended"]),
    ("Chocolate Freeze", "Chocolate Freeze", "freeze", ["chocolate", "ice blended"]),
    ("Bánh mì que", "Bread Stick", "food", ["bread", "snack"]),
    ("Bánh phô mai", "Cheese Cake", "food", ["cake", "cheese"]),
]

SIZES = ["S", "M", "L"]
PRICE_BY_CATEGORY = {
    "coffee": (29000, 59000),
    "tea": (39000, 65000),
    "freeze": (49000, 79000),
    "food": (25000, 55000),
}


FAQ_TOPICS = [
    ("wifi", "Wifi của quán là Highlands_Guest, mật khẩu là highlands123."),
    ("opening_hours", "Quán mở cửa từ 7:00 đến 22:00 hằng ngày."),
    ("payment", "Quán hỗ trợ tiền mặt, thẻ ngân hàng, Momo và chuyển khoản."),
    ("invoice", "Khách hàng có thể yêu cầu xuất hóa đơn VAT tại quầy."),
    ("parking", "Quán có khu vực giữ xe phía trước, tùy từng chi nhánh."),
    ("delivery", "Quán có hỗ trợ đặt món mang đi và giao hàng qua đối tác giao nhận."),
    ("promotion", "Khuyến mãi có thể thay đổi theo từng thời điểm và từng chi nhánh."),
    ("takeaway", "Tất cả món nước đều có thể đặt mang đi."),
    ("size", "Một số món có size S, M hoặc L tùy theo danh mục."),
    ("allergy", "Khách hàng nên báo nhân viên nếu có dị ứng với sữa, hạt hoặc thành phần đặc biệt."),
]


DOC_POLICIES = [
    "Nhân viên cần xác nhận lại tên món, size và số lượng trước khi hoàn tất đơn hàng.",
    "Không được tự ý báo giá nếu thông tin giá không có trong hệ thống menu.",
    "Khi khách hỏi thông tin không có trong dữ liệu, trợ lý cần nói rõ là chưa tìm thấy thông tin.",
    "Với câu hỏi tư vấn, trợ lý nên đề xuất tối đa ba lựa chọn để khách dễ quyết định.",
    "Với câu hỏi về wifi, giờ mở cửa, thanh toán hoặc hóa đơn, trợ lý phải ưu tiên dữ liệu FAQ.",
    "Nếu khách nói không rõ hoặc chỉ chào hỏi, trợ lý nên phản hồi ngắn gọn và hỏi khách cần hỗ trợ gì.",
    "Khi khách hỏi món ít ngọt, ưu tiên đề xuất trà hoặc cà phê có thể tùy chỉnh đường.",
    "Khi khách hỏi món tỉnh táo, ưu tiên đề xuất các món có cà phê.",
    "Khi khách hỏi món mát hoặc phù hợp trời nóng, ưu tiên đề xuất trà trái cây hoặc freeze.",
    "Khi khách hỏi món ăn nhẹ, ưu tiên đề xuất bánh hoặc snack trong menu.",
]



def make_price(category: str, size: str) -> int:
    low, high = PRICE_BY_CATEGORY[category]
    base = random.randrange(low, high + 1, 5000)
    if size == "M":
        base += 5000
    elif size == "L":
        base += 10000
    return base


def generate_menu(n: int = 120) -> None:
    rows = []
    idx = 1

    while len(rows) < n:
        name_vi, name_en, category, ingredients = random.choice(DRINK_BASES)
        size = random.choice(SIZES)

        variant = random.choice([
            "",
            " ít ngọt",
            " đá",
            " nóng",
            " đặc biệt",
            " truyền thống",
            " size " + size,
        ])

        item_name_vi = f"{name_vi}{variant}".strip()
        item_name_en = name_en

        rows.append({
            "id": f"menu_{idx:04d}",
            "name_vi": item_name_vi,
            "name_en": item_name_en,
            "price": make_price(category, size),
            "size": size,
            "category": category,
            "ingredients": "|".join(ingredients),
            "description": f"{item_name_vi} thuộc nhóm {category}, phù hợp cho khách muốn lựa chọn nhanh tại quán.",
            "tags": "|".join(ingredients + [category, size.lower()]),
            "available": "true",
        })
        idx += 1

    with MENU_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def generate_faq(n: int = 160) -> None:
    rows = []
    for i in range(1, n + 1):
        topic, answer = random.choice(FAQ_TOPICS)

        question_templates = [
            f"{topic} của quán là gì?",
            f"Cho tôi hỏi về {topic}",
            f"Thông tin {topic} như thế nào?",
            f"Can you tell me about {topic}?",
            f"What is the {topic} policy?",
        ]

        question = random.choice(question_templates)

        rows.append({
            "id": f"faq_{i:04d}",
            "topic": topic,
            "question": question,
            "answer": answer,
            "language": "vi" if i % 5 != 0 else "en",
            "source_file": "mock_faq.csv",
        })

    with FAQ_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def generate_docs(n: int = 60) -> None:
    with DOCS_JSONL.open("w", encoding="utf-8") as f:
        for i in range(1, n + 1):
            text = random.choice(DOC_POLICIES)
            record = {
                "id": f"doc_{i:04d}",
                "source_file": "mock_internal_policy.jsonl",
                "chunk_index": i - 1,
                "language": "vi",
                "text": text,
                "entities": extract_mock_entities(text),
                "metadata": {
                    "doc_type": "internal_policy",
                    "version": "mock_v1",
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_mock_entities(text: str) -> list[dict]:
    candidates = {
        "wifi": "policy",
        "giờ mở cửa": "policy",
        "thanh toán": "policy",
        "hóa đơn": "policy",
        "menu": "business_object",
        "giá": "business_object",
        "cà phê": "menu_category",
        "trà": "menu_category",
        "freeze": "menu_category",
        "bánh": "menu_category",
    }

    entities = []
    lowered = text.lower()
    for name, entity_type in candidates.items():
        if name in lowered:
            entities.append({
                "name": name,
                "type": entity_type,
                "normalized_name": name.lower(),
            })
    return entities

def main() -> None:
    random.seed(42)

    generate_menu(120)
    generate_faq(160)
    generate_docs(60)

    print(f"Created: {MENU_CSV}")
    print(f"Created: {FAQ_CSV}")
    print(f"Created: {DOCS_JSONL}")
    print("Mock data summary:")
    print("- MenuItems: 120")
    print("- FAQ chunks: 160")
    print("- Document chunks: 60")


if __name__ == "__main__":
    main()