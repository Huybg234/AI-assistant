import re
from typing import List

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 150) -> List[str]:
    """
    Split *text* into overlapping chunks aligned to sentence boundaries.
    """
    sentences = _SENTENCE_END.split(text)
    chunks: List[str] = []
    current: List[str] = []
    current_len: int = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        s_len = len(sentence) + 1

        if current_len + s_len > chunk_size and current:
            chunks.append(" ".join(current))
            overlap_buf: List[str] = []
            overlap_len: int = 0
            for s in reversed(current):
                if overlap_len + len(s) + 1 > overlap:
                    break
                overlap_buf.insert(0, s)
                overlap_len += len(s) + 1
            current = overlap_buf
            current_len = overlap_len

        current.append(sentence)
        current_len += s_len

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]
