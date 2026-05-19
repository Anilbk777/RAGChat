from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
import logging
import os

logger = logging.getLogger("uvicorn")


class QdrantStorage:
    """Service to interact with the Qdrant vector database."""

    def __init__(
        self,
        url: str = None,
        collection: str = "docs",
        dim: int = 768,
    ):
        if url is None:
            url = os.getenv("QDRANT_URL", "http://localhost:6333")
            
        try:
            self.client = QdrantClient(url=url)
            self.collection = collection
            self.dim = dim
            # Ensure the collection is automatically set up
            self._ensure_collection()
        except UnexpectedResponse as e:
            raise ConnectionError(f"Qdrant returned an error during init: {e}")
        except Exception as e:
            raise ConnectionError(f"Could not connect to Qdrant at {url}: {e}")

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't already exist."""
        try:
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=self.dim, distance=Distance.COSINE
                    ),
                )
                logger.info(
                    f"[Qdrant] Created collection '{self.collection}' (dim={self.dim})"
                )
            else:
                logger.debug(f"[Qdrant] Collection '{self.collection}' already exists.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to ensure collection '{self.collection}' exists: {e}"
            )

    def create_collection(self) -> None:
        """Public method kept for backwards compatibility."""
        self._ensure_collection()

    def delete_collection(self) -> None:
        """Delete the current collection."""
        try:
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(self.collection)
                logger.info(f"[Qdrant] Deleted collection '{self.collection}'")
        except Exception as e:
            raise RuntimeError(f"Failed to delete collection '{self.collection}': {e}")

    def recreate_collection(self) -> None:
        """Recreate the current collection (drops and recreates)."""
        self.delete_collection()
        self._ensure_collection()

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        """Upsert document embedding points with payloads into the vector store."""
        # ── GUARDS ───────────────────────────────────────────────────────────
        if not ids:
            raise ValueError(
                "upsert() received an empty ids list. "
                "This usually means embed_texts() returned no vectors — "
                "check that your PDF has extractable text."
            )
        if not vectors:
            raise ValueError(
                "upsert() received an empty vectors list. "
                "embed_texts() may have failed or returned nothing."
            )
        if not payloads:
            raise ValueError("upsert() received an empty payloads list.")

        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError(
                f"upsert() length mismatch — "
                f"ids: {len(ids)}, vectors: {len(vectors)}, payloads: {len(payloads)}. "
                "embed_texts() likely returned fewer vectors than chunks."
            )

        for i, vec in enumerate(vectors):
            if not vec:
                raise ValueError(
                    f"Vector at index {i} is empty. "
                    "The embedding model returned a blank vector for that chunk."
                )
            if len(vec) != self.dim:
                raise ValueError(
                    f"Vector at index {i} has {len(vec)} dimensions "
                    f"but this collection expects {self.dim}. "
                    "Update QdrantStorage(dim=...) to match your embedding model."
                )

        # ── BUILD & SEND ─────────────────────────────────────────────────────
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]

        logger.info(f"[Qdrant] Upserting {len(points)} points into '{self.collection}'")

        try:
            self.client.upsert(collection_name=self.collection, points=points)
            logger.info(f"[Qdrant] Upsert complete — {len(points)} points stored.")
        except UnexpectedResponse as e:
            raw = e.content.decode() if hasattr(e, "content") else str(e)
            raise RuntimeError(
                f"Qdrant rejected the upsert (HTTP {e.status_code}): {raw}"
            )
        except Exception as e:
            raise RuntimeError(f"Qdrant upsert failed unexpectedly: {e}")

    def search(self, query_vector: list[float], top_k: int = 5) -> dict:
        """Search for the top_k most similar documents."""
        # ── GUARDS ───────────────────────────────────────────────────────────
        if not query_vector:
            raise ValueError(
                "search() received an empty query vector. "
                "embed_texts() may have failed on your question."
            )
        if len(query_vector) != self.dim:
            raise ValueError(
                f"Query vector has {len(query_vector)} dimensions "
                f"but collection expects {self.dim}."
            )

        logger.info(f"[Qdrant] Searching top-{top_k} in '{self.collection}'")

        try:
            results = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
        except UnexpectedResponse as e:
            raw = e.content.decode() if hasattr(e, "content") else str(e)
            raise RuntimeError(f"Qdrant search failed (HTTP {e.status_code}): {raw}")
        except Exception as e:
            raise RuntimeError(f"Qdrant search failed unexpectedly: {e}")

        contexts = []
        sources = set()

        for point in results.points:
            payload = point.payload or {}
            text = payload.get("text", "").strip()
            source = payload.get("source", "")

            if text:
                contexts.append(text)
            if source:
                sources.add(source)

        logger.info(
            f"[Qdrant] Search returned {len(contexts)} contexts "
            f"from sources: {sources or 'none'}"
        )

        return {"contexts": contexts, "sources": list(sources)}


if __name__ == "__main__":
    db = QdrantStorage(dim=768)
    db.recreate_collection()
    print("Collection recreated successfully.")
