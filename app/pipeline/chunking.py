from typing import List


class TextChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        chunk_size: max characters per chunk
        chunk_overlap: characters shared between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks using recursive splitting."""
        # Clean the text
        text = text.strip()
        if not text:
            return []

        # If text fits in one chunk, return as-is
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        splits = self._recursive_split(text)

        # Merge splits into chunks of target size with overlap
        current_chunk = ""
        for split in splits:
            # If adding this split would exceed chunk_size
            if len(current_chunk) + len(split) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap from end of previous
                    overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                    current_chunk = overlap_text + split
                else:
                    # Single split is bigger than chunk_size — force add it
                    chunks.append(split[:self.chunk_size].strip())
                    current_chunk = ""
            else:
                current_chunk += split

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _recursive_split(self, text: str) -> List[str]:
        """Split text by trying separators in order of priority."""
        for separator in self.separators:
            if separator in text:
                splits = text.split(separator)
                # Re-attach separator to each split (except last)
                result = []
                for i, split in enumerate(splits):
                    if i < len(splits) - 1:
                        result.append(split + separator)
                    else:
                        result.append(split)
                return result
        # No separator found — return text as-is
        return [text]