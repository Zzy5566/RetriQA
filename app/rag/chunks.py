"""切块：把长文本切成适合检索的 chunk（对标 MimirQ 的 app/parsing/chunking/）。

用递归分隔符切分，模仿 langchain 的 RecursiveCharacterTextSplitter，但零依赖：
优先按段落（\n\n）切，不行再按句号/空格切，最后硬切。带 overlap 保上下文连续。
"""
from __future__ import annotations

_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ". ", " ", ""]


def _split_on(separator: str, text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    if separator:
        parts = text.split(separator)
    else:
        parts = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    current = ""
    for part in parts:
        candidate = current + separator + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # overlap：把上一块的尾巴带过来，保证跨块语义不撕裂
            start = max(0, len(part) - overlap)
            current = separator + part if start == 0 and current else part[start:]
    if current:
        chunks.append(current)
    return chunks


def split_text(text: str, *, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """递归切块：优先用大分隔符，装不下一块时切分后对小段递归处理。"""
    if len(text) <= chunk_size:
        return [text]

    for sep in _SEPARATORS[:-1]:
        if sep and sep in text:
            pieces = _split_on(sep, text, chunk_size, overlap)
            # 结果里仍有超长块 -> 用更小的分隔符递归
            if all(len(p) <= chunk_size for p in pieces):
                return pieces
            flattened: list[str] = []
            for p in pieces:
                if len(p) <= chunk_size:
                    flattened.append(p)
                else:
                    flattened.extend(split_text(p, chunk_size=chunk_size, overlap=overlap))
            return flattened

    return _split_on("", text, chunk_size, overlap)