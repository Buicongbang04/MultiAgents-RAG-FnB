INTENT_LABELS = {
    "order": 0,
    "consultant": 1,
    "faq": 2,
    "ignore": 3,
}


SEED_DATA = {
    "order": [
        "Cho anh một ly bạc xỉu đá",
        "Lấy cho tôi cà phê đen",
        "Gọi thêm bánh mì",
        "Cho em một trà sen vàng",
        "Mình muốn latte size M",
        "Order cappuccino ít ngọt",
        "Can I order an iced coffee?",
        "One caramel freeze please",
        "I want a latte",
        "Get me a cappuccino",
    ],

    "consultant": [
        "Có món nào ngon không?",
        "Gợi ý giúp tôi món dễ uống",
        "Tôi thích ít ngọt thì uống gì?",
        "Có gì hợp trời nóng không?",
        "Món nào bán chạy?",
        "What drink do you recommend?",
        "Any low sugar drink?",
        "Suggest something good",
        "What is popular here?",
        "Any drink for hot weather?",
    ],

    "faq": [
        "Wifi quán là gì?",
        "Quán mở cửa mấy giờ?",
        "Có giao hàng không?",
        "Thanh toán momo được không?",
        "Có size nào vậy?",
        "What time do you close?",
        "Do you have wifi?",
        "Can I pay with Momo?",
        "Do you deliver?",
        "What sizes are available?",
    ],

    "ignore": [
        "hello",
        "haha",
        "ừm",
        "ok",
        "alo",
        "hmmm",
        "...",
        "lol",
        "hihi",
        "hey",
    ],
}


HARD_SAMPLES = [
    {
        "text": "Có gì ngon rẻ không?",
        "intent": "consultant",
    },
    {
        "text": "Cho tôi xem món nào dễ uống",
        "intent": "consultant",
    },
    {
        "text": "Tính tiền giúp tôi",
        "intent": "order",
    },
    {
        "text": "Có bán trà sen không nhỉ",
        "intent": "order",
    },
    {
        "text": "Có đồ uống nào hợp trời mưa không?",
        "intent": "consultant",
    },
    {
        "text": "Wifi bên mình sao nhỉ?",
        "intent": "faq",
    },
]