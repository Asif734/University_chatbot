# agents/mental_health_agent.py
import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from .llm_interface import local_llm

class MentalHealthAgent:
    """
    A mental-health retrieval agent that can:
    - Load Q/A pairs from CSV
    - Build FAISS embeddings for fast semantic search
    - Fall back to local LLM if no confident match
    - Optionally detect risk flags (self-harm, severe anxiety)
    """

    def __init__(
        self,
        csv_file: str = r"C:\Users\Asif\VSCODE\University Chatbot\app\data\train.csv",
        index_path: str = r"C:\Users\Asif\VSCODE\University Chatbot\app\data\mh_index.faiss",
        meta_path: str = r"C:\Users\Asif\VSCODE\University Chatbot\app\data\mh_meta.pkl",
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3,
        threshold: float = 0.55,
        include_context_in_fallback: bool = True,
        fallback_context_k: int = 3,
    ):
        self.csv_file = csv_file
        self.index_path = index_path
        self.meta_path = meta_path
        self.top_k = max(1, top_k)
        self.threshold = float(threshold)
        self.include_context_in_fallback = include_context_in_fallback
        self.fallback_context_k = max(1, fallback_context_k)

        self.model = SentenceTransformer(embed_model_name)

        # Load or build index
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self._load_index_and_meta()
            print("MentalHealthAgent: Loaded FAISS index and metadata.")
        else:
            print("MentalHealthAgent: Building index from CSV (first run)...")
            self._build_index_from_csv()
            print("MentalHealthAgent: Index built and saved.")

    # ---------- Public API ----------

    def respond(self, message: str) -> (str, bool):
        """
        Returns assistant response and a risk flag.
        Risk flag = True if message indicates mental health crisis.
        """
        # Encode query
        q = self._encode_and_normalize([message])
        D, I = self.index.search(q, self.top_k)
        scores = D[0]
        idxs = I[0]

        best_score = float(scores[0])
        best_idx = int(idxs[0])

        if best_idx >= 0 and best_score >= self.threshold:
            # confident match
            response = self.pairs[best_idx]["assistant"]
        else:
            # fallback to LLM
            context_text = None
            if self.include_context_in_fallback:
                context_text = self._format_context(idxs, scores, k=min(self.fallback_context_k, len(self.pairs)))
            response = local_llm(self._fallback_prompt(message, context_text))

        risk_flag = self._detect_risk(message)
        return response, risk_flag

    # ---------- Internal helpers ----------

    def _build_index_from_csv(self):
        df = pd.read_csv(self.csv_file)
        if "text" not in df.columns:
            raise ValueError("MentalHealthAgent: CSV must contain a 'text' column.")

        pairs = []
        for raw in df["text"].dropna().astype(str).tolist():
            if "<HUMAN>:" in raw and "<ASSISTANT>:" in raw:
                human_part, rest = raw.split("<ASSISTANT>:", 1)
                human = human_part.replace("<HUMAN>:", "").strip()
                assistant = rest.strip()
                if human and assistant:
                    pairs.append({"human": human, "assistant": assistant})

        if not pairs:
            raise ValueError("MentalHealthAgent: No valid <HUMAN>/<ASSISTANT> pairs found.")

        self.pairs = pairs
        questions = [p["human"] for p in pairs]
        emb = self._encode_and_normalize(questions)

        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb.astype(np.float32))

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(index, self.index_path)

        meta = {"pairs": self.pairs, "dim": dim}
        with open(self.meta_path, "wb") as f:
            pickle.dump(meta, f)

        self.index = index

    def _load_index_and_meta(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            meta = pickle.load(f)
        self.pairs = meta["pairs"]

    def _encode_and_normalize(self, texts):
        vec = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if vec.ndim == 1:
            vec = vec[None, :]
        vec = vec.astype(np.float32)
        norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
        return vec / norms

    def _format_context(self, idxs, scores, k=3):
        items = []
        for rank in range(min(k, len(self.pairs))):
            idx = int(idxs[rank])
            if idx < 0:
                continue
            sim = float(scores[rank])
            q = self.pairs[idx]["human"]
            a = self.pairs[idx]["assistant"]
            items.append(f"[sim={sim:.2f}] Q: {q}\nA: {a}")
        return "\n\n".join(items) if items else None

    def _fallback_prompt(self, user_message: str, context_text: str | None):
        if context_text:
            return (
                "You are a supportive, empathetic mental health assistant. "
                "Use the following examples if relevant, but keep the reply concise, warm, and non-clinical. "
                "If risk is detected (self-harm, harm to others, or medical emergency), advise contacting local professionals.\n\n"
                f"Examples:\n{context_text}\n\n"
                f"User: {user_message}\nAssistant:"
            )
        else:
            return (
                "You are a supportive, empathetic mental health assistant. "
                "Respond concisely and kindly. If risk is detected (self-harm, harm to others, or medical emergency), "
                "advise contacting local professionals.\n\n"
                f"User: {user_message}\nAssistant:"
            )

    def _detect_risk(self, message: str) -> bool:
        """
        Simple keyword-based detection. Can be replaced with a small ML classifier.
        """
        keywords = ["suicide", "harm", "kill myself", "depressed", "hopeless"]
        return any(kw in message.lower() for kw in keywords)
