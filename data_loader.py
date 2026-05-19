from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()


class DocumentParser:
    """Service to load and split documents into chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.reader = PDFReader()
        self.splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def parse_pdf(self, path: str) -> list[str]:
        docs = self.reader.load_data(file=path)
        texts = [d.text for d in docs if getattr(d, "text", None)]
        chunks = []
        for t in texts:
            chunks.extend(self.splitter.split_text(t))
        return chunks


class TextEmbedder:
    """Service to generate vector embeddings for text chunks."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # HuggingFaceEmbeddings returns a list of embedding vectors (list of list of float)
        return self.embeddings.embed_documents(texts)


# ── BACKWARD COMPATIBILITY WRAPPERS ──────────────────────────────────────────

_default_parser = DocumentParser()
_default_embedder = TextEmbedder()


def load_and_chunk_pdf(path: str) -> list[str]:
    """Backward compatible wrapper for load_and_chunk_pdf."""
    return _default_parser.parse_pdf(path)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Backward compatible wrapper for embed_texts."""
    return _default_embedder.embed_texts(texts)