import sqlite3
from dotenv import load_dotenv
import os

load_dotenv(override=True)

SQL_DB_NAME = os.getenv("SQL_DB_NAME")

def get_sqldb_connection():
    conn = sqlite3.connect(SQL_DB_NAME)
    return conn

def create_tables():
    conn = get_sqldb_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retrieval_experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

            embedding_model TEXT,
            retriever TEXT,
            reranker TEXT,
            prompt_version TEXT,

            chunk_size INTEGER,
            chunk_overlap INTEGER,
            RETRIEVAL_K INTEGER,

            mrr REAL,
            ndcg REAL,
            coverage REAL
        )
    """)

    conn.commit()
    conn.close()

def alter_tables():
    conn = get_sqldb_connection()
    cursor = conn.cursor()

    cursor.execute("""
        ALTER TABLE retrieval_experiments
        ADD COLUMN chunking_strategy TEXT
    """)

    conn.commit()
    conn.close()
    print("Column Added !!")

def populate_column(column):
    conn = get_sqldb_connection()
    cursor = conn.cursor()

    cursor.execute(f"""
        UPDATE retrieval_experiments
        SET {column} = 'RecursiveCharacterTextSplitter'
    """)

    conn.commit()
    conn.close()
    print(f"Column : {column} value populated  !!")    

if __name__ == "__main__":
    populate_column("chunking_strategy")