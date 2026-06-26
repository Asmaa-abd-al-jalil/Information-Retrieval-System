import os
import sqlite3

DB_PATH = "data/documents.db"

os.makedirs("data", exist_ok=True)


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            raw_text TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def insert_documents(docs: dict):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executemany(
        "INSERT OR IGNORE INTO documents (doc_id, raw_text) VALUES (?, ?)",
        [(doc_id, text) for doc_id, text in docs.items()]
    )

    conn.commit()
    conn.close()


def get_documents_by_ids(doc_ids: list) -> dict:

    if not doc_ids:
        return {}

    conn = get_connection()
    cursor = conn.cursor()

    placeholders = ",".join("?" * len(doc_ids))

    cursor.execute(
        f"""
        SELECT doc_id, raw_text
        FROM documents
        WHERE doc_id IN ({placeholders})
        """,
        doc_ids
    )

    results = {
        row[0]: row[1]
        for row in cursor.fetchall()
    }

    conn.close()

    return results


def get_all_documents() -> dict:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT doc_id, raw_text FROM documents"
    )

    docs = {
        row[0]: row[1]
        for row in cursor.fetchall()
    }

    conn.close()

    return docs


def get_document_count() -> int:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM documents")

    count = cursor.fetchone()[0]

    conn.close()

    return count