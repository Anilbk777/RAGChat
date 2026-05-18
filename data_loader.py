from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os
import numpy as np
load_dotenv()


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",huggingfacehub_api_token=os.getenv("HUGGING_FACE_API_KEY"))

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[np.ndarray]:
    response = embeddings.embed_documents(texts)
    return response