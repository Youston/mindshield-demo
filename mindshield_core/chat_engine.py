from __future__ import annotations

from typing import List, Dict, Any

from openai import OpenAI

from .retrieval import WindowRetriever
from .logger import log_turn
from .modality import score_modalities, choose_modality

# Delay heavy imports and model load until first use to keep memory low
from functools import lru_cache

# We intentionally avoid importing KeyBERT / torch at module import time because
# Streamlit executes the script twice during startup and Render has a tight
# memory limit.  The helper below loads a much smaller model the first time
# keywords are requested and reuses it afterwards.

@lru_cache(maxsize=1)
def _get_kw_model():
    """Return (and cache) a lightweight KeyBERT model instance."""
    from keybert import KeyBERT  # import here to defer PyTorch / transformers

    # The *paraphrase-MiniLM-L3-v2* model (~45 MB) is ~3× smaller than the
    # earlier L6-v2 we used and still provides useful keyword extraction.
    return KeyBERT("sentence-transformers/paraphrase-MiniLM-L3-v2")

class ChatEngine:
    """Wraps LLM chat, keyword extraction, retrieval, and logging."""

    def __init__(self, retriever: WindowRetriever):
        self.retriever = retriever
        self.client = OpenAI()

    # ------------------------------------------------------------------
    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        kws = _get_kw_model().extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words="english", top_n=top_k)
        return [k[0] for k in kws]

    def chat(self, session_id: str, history: List[Dict[str, str]], user_msg: str) -> str:
        """Process a chat turn with retrieval and logging."""
        keywords = self._extract_keywords(user_msg)
        modality_scores = score_modalities(user_msg)
        chosen_modality = choose_modality(modality_scores)
        retrievals = self.retriever.retrieve(user_msg, keywords)

        evidence_blobs = "\n---\n".join([r["blob"] for r in retrievals])
        # cap to 800 tokens roughly by truncation
        evidence_blobs = evidence_blobs[:4000]

        prompt_messages = history + [
            {"role": "system", "content": f"Evidence for this user query:\n{evidence_blobs}"},
            {"role": "user", "content": user_msg},
        ]

        resp = self.client.chat.completions.create(
            model="gpt-4o", messages=prompt_messages
        )
        reply = resp.choices[0].message.content

        # Log
        log_turn(
            session_id,
            {
                "user_message": user_msg,
                "keywords": keywords,
                "modality_scores": modality_scores,
                "chosen_modality": chosen_modality,
                "retrieval_meta": retrievals,
                "llm_prompt": str(prompt_messages),
                "llm_response": reply,
                "safety_flag": "none",
            },
        )
        return reply 