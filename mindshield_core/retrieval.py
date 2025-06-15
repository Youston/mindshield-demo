from __future__ import annotations

import tiktoken
from typing import List, Tuple, Dict, Any

class WindowRetriever:
    """Token-window retriever that slices indexed chunks and re-embeds windows.

    Parameters
    ----------
    index : Any
        Handle to a vector database collection (e.g. LanceDB table or Qdrant).
    window_size : int, default 256
        Number of tokens per chunk stored in the collection.
    margin : int, default 64
        Extra tokens to include before and after each keyword hit when extracting
        a contextual window.
    """

    def __init__(self, index: Any, window_size: int = 256, margin: int = 64):
        self.index = index
        self.window_size = window_size
        self.margin = margin
        self._enc = tiktoken.get_encoding("cl100k_base")

    # ---------------------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------------------
    def find_hits(self, text: str, keywords: List[str]) -> List[Tuple[int, int]]:
        """Return a list of (start_idx, end_idx) token spans where *any* keyword appears.

        Indices are token offsets in the full document.
        """
        tokens = self._enc.encode(text)
        token_str = [self._enc.decode([tok]) for tok in tokens]

        spans: List[Tuple[int, int]] = []
        for i, tok_str in enumerate(token_str):
            for kw in keywords:
                if kw.lower() in tok_str.lower():
                    start = max(i - self.margin, 0)
                    end = min(i + len(kw.split()) + self.margin, len(tokens))
                    spans.append((start, end))
        return spans

    def grab_window(self, doc_id: str, span: Tuple[int, int]) -> str:
        """Fetch tokens around *span* from the indexed doc and return decoded string."""
        rec = self.index.get(doc_id)  # assuming index has .get(doc_id) -> dict with 'text'
        if rec is None:
            return ""
        tokens = self._enc.encode(rec["text"])
        start, end = span
        window_tokens = tokens[start:end]
        return self._enc.decode(window_tokens)

    def _embed(self, text: str):
        """Embed a blob using OpenAI Ada or local model (placeholder)."""
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(model="text-embedding-3-small", input=text)
        return resp.data[0].embedding

    def retrieve(self, query: str, keywords: List[str], top_k: int = 3) -> List[Dict]:
        """Hybrid retrieval using keyword windows then semantic rerank."""
        hits: List[Dict] = []
        # naive: iterate docs in index.metadata
        for doc in self.index:
            spans = self.find_hits(doc["text"], keywords)
            for span in spans:
                blob = self.grab_window(doc["id"], span)
                hits.append({"blob": blob, "source": doc["id"], "span": span})
        # embed and rank
        q_emb = self._embed(query)
        for h in hits:
            b_emb = self._embed(h["blob"])
            # cosine similarity
            import numpy as np
            sim = np.dot(q_emb, b_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(b_emb))
            h["similarity"] = sim
        hits.sort(key=lambda x: x["similarity"], reverse=True)
        return hits[:top_k] 