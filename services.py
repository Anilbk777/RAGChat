import logging
import os
import uuid
from pathlib import Path

import inngest
from inngest.experimental import ai
from fastapi import UploadFile

from custom_types import (
    RAGError,
    PDFEmptyError,
    PDFTooBigError,
    PDFNotReadableError,
    PDFNoContentError,
    EmbeddingError,
    VectorStoreError,
    QueryEmbeddingError,
    NoContextFoundError,
    LLMError,
    RAGChunkAndSrc,
    RAGUpsertResult,
    RAGSearchResult,
)
from data_loader import DocumentParser, TextEmbedder
from vector_db import QdrantStorage

logger = logging.getLogger("uvicorn")

# ── CONSTANTS ────────────────────────────────────────────────────────────────

MAX_PDF_SIZE_MB = 50
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024
MIN_CHUNK_LENGTH = 20  # chars — below this a chunk is considered garbage
MIN_CHUNKS_REQUIRED = 1


# ── ORCHESTRATION SERVICE ────────────────────────────────────────────────────

class RAGService:
    """Orchestrates PDF text loading, embedding, and vector database operations."""

    def __init__(
        self,
        parser: DocumentParser,
        embedder: TextEmbedder,
        storage: QdrantStorage,
    ):
        self.parser = parser
        self.embedder = embedder
        self.storage = storage

    def ingest(self, pdf_path: str, source_id: str) -> int:
        """Loads, chunks, generates embeddings, and saves a PDF to vector storage."""
        filename = Path(pdf_path).name

        try:
            chunks = self.parser.parse_pdf(pdf_path)
        except FileNotFoundError:
            raise RAGError(f"PDF file not found on disk: {pdf_path}", 404)
        except PermissionError:
            raise RAGError(f"Permission denied reading: {pdf_path}", 403)
        except Exception as e:
            raise PDFNotReadableError(str(e))

        self.validate_chunks(chunks, filename)

        try:
            vecs = self.embedder.embed_texts(chunks)
        except Exception as e:
            raise EmbeddingError(str(e))

        if not vecs or len(vecs) != len(chunks):
            raise EmbeddingError(
                "Embedding count mismatch — got fewer vectors than chunks."
            )

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]
        payloads = [
            {"source": source_id, "text": chunks[i]} for i in range(len(chunks))
        ]

        try:
            self.storage.upsert(ids, vecs, payloads)
        except Exception as e:
            raise VectorStoreError(str(e))

        return len(chunks)

    def search(self, question: str, top_k: int = 5) -> RAGSearchResult:
        """Generates embedding for the query and searches the vector store."""
        try:
            query_vecs = self.embedder.embed_texts([question])
        except Exception as e:
            raise QueryEmbeddingError(str(e))

        if not query_vecs:
            raise QueryEmbeddingError("Embedding returned no vectors.")

        query_vec = query_vecs[0]

        try:
            found = self.storage.search(query_vec, top_k)
        except Exception as e:
            raise VectorStoreError(f"Search failed: {e}")

        contexts = found.get("contexts", [])
        sources = found.get("sources", [])

        if not contexts:
            raise NoContextFoundError()

        return RAGSearchResult(contexts=contexts, sources=sources)

    @staticmethod
    def validate_chunks(chunks: list[str], filename: str) -> None:
        """Ensure extracted chunks are meaningful."""
        if not chunks:
            raise PDFNoContentError()

        meaningful = [c for c in chunks if len(c.strip()) >= MIN_CHUNK_LENGTH]
        if len(meaningful) < MIN_CHUNKS_REQUIRED:
            raise PDFNoContentError()

        logger.info(
            f"[{filename}] Extracted {len(chunks)} chunks ({len(meaningful)} meaningful)."
        )


# ── INNGEST CLIENT ───────────────────────────────────────────────────────────

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logger,
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


# ── HELPERS ──────────────────────────────────────────────────────────────────

def validate_pdf_file(file: UploadFile, raw_bytes: bytes) -> None:
    """Run all pre-processing validations on the uploaded file."""
    is_pdf_content_type = (file.content_type or "").lower() == "application/pdf"
    is_pdf_extension = (file.filename or "").lower().endswith(".pdf")
    if not (is_pdf_content_type or is_pdf_extension):
        raise RAGError(
            f"Only PDF files are accepted. Received content-type: '{file.content_type}'.",
            415,
        )

    if len(raw_bytes) == 0:
        raise PDFEmptyError()

    if not raw_bytes.startswith(b"%PDF"):
        raise RAGError(
            "The uploaded file does not appear to be a valid PDF "
            "(missing PDF header). It may be corrupted or renamed.",
            422,
        )

    size_mb = len(raw_bytes) / (1024 * 1024)
    if len(raw_bytes) > MAX_PDF_SIZE_BYTES:
        raise PDFTooBigError(size_mb)


# ── INNGEST FUNCTIONS ────────────────────────────────────────────────────────

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):

    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        filename = Path(pdf_path).name

        logger.info(f"[Ingest] Loading PDF: {filename}")
        
        service = RAGService(DocumentParser(), TextEmbedder(), QdrantStorage())
        try:
            chunks = service.parser.parse_pdf(pdf_path)
        except FileNotFoundError:
            raise RAGError(f"PDF file not found on disk: {pdf_path}", 404)
        except PermissionError:
            raise RAGError(f"Permission denied reading: {pdf_path}", 403)
        except Exception as e:
            raise PDFNotReadableError(str(e))

        service.validate_chunks(chunks, filename)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        logger.info(f"[Ingest] Embedding {len(chunks)} chunks for: {source_id}")

        service = RAGService(DocumentParser(), TextEmbedder(), QdrantStorage())
        try:
            vecs = service.embedder.embed_texts(chunks)
        except Exception as e:
            raise EmbeddingError(str(e))

        if not vecs or len(vecs) != len(chunks):
            raise EmbeddingError(
                "Embedding count mismatch — got fewer vectors than chunks."
            )

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]
        payloads = [
            {"source": source_id, "text": chunks[i]} for i in range(len(chunks))
        ]

        try:
            service.storage.upsert(ids, vecs, payloads)
        except Exception as e:
            raise VectorStoreError(str(e))

        logger.info(
            f"[Ingest] Successfully upserted {len(chunks)} vectors for: {source_id}"
        )
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run(
        "load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc
    )
    ingested = await ctx.step.run(
        "embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult
    )
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):

    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        logger.info(f"[Query] Embedding question: {question[:80]}...")
        service = RAGService(DocumentParser(), TextEmbedder(), QdrantStorage())
        return service.search(question, top_k)

    question = ctx.event.data.get("question", "").strip()
    top_k = int(ctx.event.data.get("top_k", 5))

    if not question:
        raise RAGError("Question cannot be empty.", 400)

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult,
    )

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
    )

    try:
        res = await ctx.step.ai.infer(
            "llm-answer",
            adapter=adapter,
            body={
                "max_tokens": 1024,
                "temperature": 0.2,
                "messages": [
                    {
                        "role": "system",
                        "content": "You answer questions using only the provided context. "
                        "If the context does not contain enough information, say so honestly.",
                    },
                    {"role": "user", "content": user_content},
                ],
            },
        )
    except Exception as e:
        raise LLMError(str(e))

    try:
        answer = res["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"Unexpected response shape from LLM: {e}")

    if not answer:
        raise LLMError("LLM returned an empty answer.")

    logger.info(
        f"[Query] Answer generated ({len(answer)} chars) from {len(found.contexts)} contexts."
    )

    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts),
    }
