import os
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorStore:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.texts = []  # List[str]
        self.metadata = []  # List[dict]
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)

    def add_texts(self, texts: List[str], metadatas: List[dict]):
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadata.extend(metadatas)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, dict]]:
        query_vec = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vec, top_k)
        results = [(self.texts[i], self.metadata[i]) for i in I[0]]
        return results
