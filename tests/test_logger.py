from mindshield_core.logger import log_turn, _get_engine
import os, sqlite3, json, datetime

def test_log_turn(tmp_path):
    db_file = tmp_path / "test.db"
    os.environ["CHAT_DB"] = str(db_file)
    # patch path
    from mindshield_core import logger as lg
    lg._DB_PATH = str(db_file)
    lg._engine = None

    log_turn("sess1", {
        "user_message": "Call me at 123-456-7890",
        "keywords": ["call"],
        "modality_scores": {},
        "chosen_modality": "",
        "retrieval_meta": [],
        "llm_prompt": "prompt",
        "llm_response": "hi",
        "safety_flag": "none",
    })

    conn = sqlite3.connect(db_file)
    cur = conn.execute("SELECT user_message FROM chat_logs WHERE session_id='sess1'")
    row = cur.fetchone()
    assert "[PHONE]" in row[0] 