
from app.utils.tts_preprocess import preprocess_tts_text

samples = [
    "Bạc xỉu đá size M giá 49.000đ.",
    "Tổng cộng là 54000 VND, đã gồm VAT.",
]

for s in samples:
    print(preprocess_tts_text(s))
