from __future__ import annotations

from typing import List, Dict, Any

from keybert import KeyBERT
from openai import OpenAI

from .retrieval import WindowRetriever
from .logger import log_turn

kw_model = KeyBERT("all-MiniLM-L6-v2")

class ChatEngine:
    """Wraps LLM chat, keyword extraction, retrieval, and logging."""

    def __init__(self, retriever: WindowRetriever):
        self.retriever = retriever
        self.client = OpenAI()

    # ------------------------------------------------------------------
    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        kws = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words="english", top_n=top_k)
        return [k[0] for k in kws]

    def chat(self, session_id: str, history: List[Dict[str, str]], user_msg: str) -> str:
        """Process a chat turn with retrieval and logging."""
        keywords = self._extract_keywords(user_msg)
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
                "modality_scores": {},
                "chosen_modality": "",
                "retrieval_meta": retrievals,
                "llm_prompt": str(prompt_messages),
                "llm_response": reply,
                "safety_flag": "none",
            },
        )
        return reply 