def chunk_text(text: str, size: int = 50, overlap: int = 10) -> list[str]:
    """Split `text` into chunks of words.

    Rules (see README for examples):
    - Words are separated by whitespace (use str.split()).
    - Each chunk has at most `size` words, joined with a single space.
    - Consecutive chunks share exactly `overlap` words.
    - Stop when a chunk reaches the last word. No extra trailing chunk.
    - Empty or whitespace-only text returns [].
    - Raise ValueError if size <= 0, overlap < 0, or overlap >= size.
    """
    if size <= 0:
        raise ValueError("The parameter 'size' must be positive number")
    if overlap < 0 or overlap >= size:
        raise ValueError("The parameter 'overlap' must be non-negative and strictly less than size")

    words = text.split()
    if not words:
        return []

    chunks = []
    step = size - overlap
    start = 0

    while start < len(words):
        chunk = words[start : start + size]
        chunks.append(" ".join(chunk))
        if start + size >= len(words):
            break
        start += step

    return chunks
