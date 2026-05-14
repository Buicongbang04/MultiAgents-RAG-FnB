import re

from app.core.config import get_settings


CLAUSE_END_PATTERN = re.compile(r"([,.?!;:。！？؛،]+)")


class ClauseSplitter:
    """
    Tách text thành clause để sau này nối TTS.

    MVP:
    - Nhận token/text fragment.
    - Buffer lại.
    - Khi gặp dấu câu thì emit clause.
    - Flush phần còn lại khi stream kết thúc.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.buffer = ""

    def push(self, text: str) -> list[str]:
        self.buffer += text

        clauses = []

        while True:
            match = CLAUSE_END_PATTERN.search(self.buffer)
            if not match:
                break

            end_idx = match.end()
            clause = self.buffer[:end_idx].strip()
            self.buffer = self.buffer[end_idx:].lstrip()

            if len(clause) >= self.settings.clause_min_chars:
                clauses.append(clause)

        return clauses

    def flush(self) -> list[str]:
        remaining = self.buffer.strip()
        self.buffer = ""

        if remaining:
            return [remaining]

        return []