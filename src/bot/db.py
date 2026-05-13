import csv
import json
import os
import sqlite3
from pathlib import Path

from telegram import Update

DB_PATH = os.getenv("DB_PATH", "/app/data/app.db")
TWISTERS_CSV = os.getenv("TWISTERS_CSV", "/app/data/tongue_twisters.csv")
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "/app/data/audio"))
ADMIN_IDS = {int(x) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip().isdigit()}
LETTER_TO_COL = {
    "л": "has_l",
    "р": "has_r",
    "с": "has_s",
    "т": "has_t",
    "ц": "has_c",
    "ч": "has_ch",
    "ш": "has_sh",
    "щ": "has_sch",
}


def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _load_twisters_csv():
    # читает CSV и возвращает список кортежей (text, has_l, ..., has_sch)
    col_map = {"л": 0, "р": 1, "с": 2, "т": 3, "ц": 4, "ч": 5, "ш": 6, "щ": 7}
    rows = []
    try:
        with open(TWISTERS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                flags = [int(row.get(k, 0)) for k in col_map]
                rows.append((row["text"], *flags))
    except FileNotFoundError:
        pass
    return rows


def init_db():
    # создаёт таблицы и заполняет скороговорки из CSV при первом запуске
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id INTEGER UNIQUE NOT NULL,
            role TEXT NOT NULL DEFAULT 'patient',
            name TEXT,
            selected_letters TEXT DEFAULT '[]',
            current_twister_id INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS twisters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            has_l INTEGER NOT NULL DEFAULT 0,
            has_r INTEGER NOT NULL DEFAULT 0,
            has_s INTEGER NOT NULL DEFAULT 0,
            has_t INTEGER NOT NULL DEFAULT 0,
            has_c INTEGER NOT NULL DEFAULT 0,
            has_ch INTEGER NOT NULL DEFAULT 0,
            has_sh INTEGER NOT NULL DEFAULT 0,
            has_sch INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS patient_therapist_access (
            patient_id INTEGER NOT NULL,
            therapist_id INTEGER NOT NULL,
            UNIQUE(patient_id, therapist_id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            twister_id INTEGER NOT NULL,
            selected_letters TEXT NOT NULL,
            audio_path TEXT,
            ml_score REAL,
            ml_payload TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute("SELECT COUNT(*) AS c FROM twisters")
    if cur.fetchone()["c"] == 0:
        seed = _load_twisters_csv()
        if seed:
            cur.executemany(
                """
                INSERT INTO twisters (text, has_l, has_r, has_s, has_t, has_c, has_ch, has_sh, has_sch)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                seed,
            )
    conn.commit()
    conn.close()


def get_or_create_user(update: Update):
    # регистрирует нового пользователя; admin_ids получают роль admin сразу
    tg_id = update.effective_user.id
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE telegram_id = ?", (tg_id,))
    user = cur.fetchone()
    if not user:
        role = "admin" if tg_id in ADMIN_IDS else "patient"
        cur.execute(
            "INSERT INTO users (telegram_id, role, name) VALUES (?, ?, ?)",
            (tg_id, role, None),
        )
        conn.commit()
        cur.execute("SELECT * FROM users WHERE telegram_id = ?", (tg_id,))
        user = cur.fetchone()
    conn.close()
    return user


def get_selected_letters(user) -> list[str]:
    # парсит JSON-список букв из строки БД
    return [x.lower() for x in json.loads(user["selected_letters"] or "[]")]


def pick_twister(letters):
    # случайная скороговорка по буквам
    conn = db_conn()
    cur = conn.cursor()
    twister = None
    if letters:
        where = " OR ".join([f"{LETTER_TO_COL[l]} = 1" for l in letters])
        cur.execute(f"SELECT * FROM twisters WHERE {where} ORDER BY RANDOM() LIMIT 1")
        twister = cur.fetchone()
    if not twister:
        cur.execute("SELECT * FROM twisters ORDER BY RANDOM() LIMIT 1")
        twister = cur.fetchone()
    conn.close()
    return twister
