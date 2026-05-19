from pydantic import BaseModel, validator

class RAGError(Exception):

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class PDFEmptyError(RAGError):
    def __init__(self):
        super().__init__(
            "The PDF file is empty (0 bytes). Please upload a valid PDF.", 400
        )


class PDFTooBigError(RAGError):
    def __init__(self, size_mb: float):
        super().__init__(
            f"PDF is {size_mb:.1f} MB which exceeds the 50 MB limit.",
            413,
        )


class PDFNotReadableError(RAGError):
    def __init__(self, detail: str = ""):
        msg = "Could not extract text from this PDF."
        if detail:
            msg += f" Reason: {detail}"
        msg += " The file may be scanned, image-only, password-protected, or corrupted."
        super().__init__(msg, 422)


class PDFNoContentError(RAGError):
    def __init__(self):
        super().__init__(
            "The PDF was parsed but no readable text could be extracted. "
            "It may contain only images or scanned pages with no OCR layer.",
            422,
        )


class EmbeddingError(RAGError):
    def __init__(self, detail: str = ""):
        msg = "Failed to generate embeddings for the PDF content."
        if detail:
            msg += f" Detail: {detail}"
        super().__init__(msg, 502)


class VectorStoreError(RAGError):
    def __init__(self, detail: str = ""):
        msg = "Failed to save embeddings to the vector database."
        if detail:
            msg += f" Detail: {detail}"
        super().__init__(msg, 502)


class QueryEmbeddingError(RAGError):
    def __init__(self, detail: str = ""):
        msg = "Failed to embed your question."
        if detail:
            msg += f" Detail: {detail}"
        super().__init__(msg, 502)


class NoContextFoundError(RAGError):
    def __init__(self):
        super().__init__(
            "No relevant context was found in the ingested documents for your question. "
            "Try rephrasing, or make sure the relevant PDF has been ingested.",
            404,
        )


class LLMError(RAGError):
    def __init__(self, detail: str = ""):
        msg = "The language model failed to generate an answer."
        if detail:
            msg += f" Detail: {detail}"
        super().__init__(msg, 502)


class RAGChunkAndSrc(BaseModel):
    chunks: list[str]
    source_id: str = None


class RAGUpsertResult(BaseModel):
    ingested: int


class RAGSearchResult(BaseModel):
    contexts: list[str]
    sources: list[str]


class RAGQueryResult(BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

    @validator("question")
    def question_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be blank.")
        return v.strip()

    @validator("top_k")
    def top_k_must_be_valid(cls, v):
        if not (1 <= v <= 20):
            raise ValueError("top_k must be between 1 and 20.")
        return v