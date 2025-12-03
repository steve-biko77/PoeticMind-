import sqlite3
import pandas as pd
import os

DB_PATH = "database/poeticmind.db"
CSV_PATH = "data/cleaned/clean_poems.csv"

def create_database():
    if not os.path.exists("database"):
        os.makedirs("database")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Charger et exécuter le schéma SQL
    with open("database/schema.sql", "r", encoding="utf-8") as f:
        cursor.executescript(f.read())

    conn.commit()
    conn.close()
    print("📦 Base SQLite créée avec succès.")

def insert_data():
    df = pd.read_csv(CSV_PATH)

    conn = sqlite3.connect(DB_PATH)

    df.to_sql("poems", conn, if_exists="append", index=False)

    conn.close()
    print("📥 Données insérées dans la base.")

if __name__ == "__main__":
    create_database()
    insert_data()
    print("✔️ Base prête : database/poeticmind.db")
