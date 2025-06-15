from __future__ import annotations

import datetime as dt
from typing import Dict, Any

from sqlalchemy import create_engine, Table, Column, MetaData, String, DateTime, JSON
from sqlalchemy.engine import Engine
from sqlalchemy.sql import insert

from .privacy import redact_pii, hash_prompt

_DB_PATH = "data/chat_logs.db"
_METADATA = MetaData()

chat_logs = Table(
    "chat_logs",
    _METADATA,
    Column("session_id", String, primary_key=False),
    Column("timestamp_utc", DateTime),
    Column("user_message", String),
    Column("keywords", JSON),
    Column("modality_scores", JSON),
    Column("chosen_modality", String),
    Column("retrieval_meta", JSON),
    Column("llm_prompt_hash", String),
    Column("llm_response", String),
    Column("safety_flag", String),
    Column("therapist_fix", String, nullable=True),
)

_engine: Engine | None = None

def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(f"sqlite:///{_DB_PATH}", future=True)
        _METADATA.create_all(_engine)
    return _engine


def log_turn(session_id: str, data: Dict[str, Any]) -> None:
    """Persist a single chat turn after redacting PII."""
    engine = _get_engine()

    cleaned = data.copy()
    cleaned["user_message"] = redact_pii(cleaned.get("user_message", ""))
    prompt_text = cleaned.pop("llm_prompt", "")
    cleaned["llm_prompt_hash"] = hash_prompt(prompt_text)
    cleaned["timestamp_utc"] = dt.datetime.utcnow()

    with engine.begin() as conn:
        conn.execute(insert(chat_logs).values(session_id=session_id, **cleaned))

def fetch_recent_logs(limit: int = 100):
    """Return recent chat logs as list of dicts."""
    engine = _get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            chat_logs.select().order_by(chat_logs.c.timestamp_utc.desc()).limit(limit)
        )
        rows = [dict(r) for r in result]
    return rows 