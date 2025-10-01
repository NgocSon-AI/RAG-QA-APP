import os
from typing import List
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import docx
from PyPDF2 import PdfReader

# ---------------- File reading ----------------
def read_file(file) -> str:
    text = ""
    try:
        fname = file.name.lower()
        if fname.endswith(".pdf"):
            reader = PdfReader(file)
            text = "".join([page.extract_text() or "" for page in reader.pages])
        elif fname.endswith(".docx"):
            doc = docx.Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif fname.endswith(".txt") or fname.endswith(".csv"):
            text = file.read().decode("utf-8")
    except Exception as e:
        print(f"Không đọc được file {file.name}: {e}")
    return text or ""

# ---------------- Text splitter ----------------
def split_text(text: str, chunk_size=500, chunk_overlap=50) -> list:
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text) or []

from langchain_core.embeddings import Embeddings

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


# ---------------- Embedding model selection ----------------
def get_embedding_model(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", openai_api_key=None):
    if openai_api_key:
        return OpenAIEmbeddings(openai_api_key=openai_api_key)
    else:
        return SentenceTransformerEmbeddings(model_name)

# ---------------- Vector store ----------------
def build_vectorstore(texts, embedding_model, use_faiss=True):
    if use_faiss:
        return FAISS.from_texts(texts, embedding_model)
    else:
        return Chroma.from_texts(texts, embedding_model)
