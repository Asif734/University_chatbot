# app/agents/rag_store.py
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

class RAGStore:
    def __init__(self, index_path="app/data/rag_index.faiss", meta_path="app/data/rag_meta.pkl", embedding_model_name="all-MiniLM-L6-v2"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.texts = []
        self.metadata = []
        self.index = None

        # Load embedding model locally
        self.embeddings = SentenceTransformer(embedding_model_name)

        # Load existing index if exists
        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)

    def _save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "wb") as f:
                pickle.dump(self.metadata, f)

    def split_text(self, text, chunk_size=500, overlap=50):
        # Basic text splitter
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def add_document(self, text: str = None, filename: str = "user_text"):
        if not text:
            return

        chunks = self.split_text(text)
        self.texts.extend(chunks)
        self.metadata.extend([{"source": filename}] * len(chunks))

        vectors = self.embeddings.encode(chunks, convert_to_numpy=True).astype("float32")

        if self.index is None:
            dim = vectors.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(vectors)
        else:
            self.index.add(vectors)

        self._save_index()

    def search(self, query: str, top_k=5):
        if self.index is None or len(self.texts) == 0:
            return []

        vector = self.embeddings.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(vector, top_k)
        results = []
        for idx in I[0]:
            results.append({"text": self.texts[idx], "metadata": self.metadata[idx]})
        return results
        # inside RAGStore class
    def query(self, query_text: str, top_k=5):
        """Alias for search, so older code still works."""
        return [r["text"] for r in self.search(query_text, top_k)]