import requests
import pandas as pd
import json
import os
from time import sleep

RAW_JSON = "data/raw/poems.json"
RAW_CSV = "data/raw/poems.csv"

BASE_URL = "https://poetrydb.org"


def get(endpoint):
    """Requête robuste avec gestion d'erreurs."""
    try:
        r = requests.get(BASE_URL + endpoint, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"⚠️ Endpoint {endpoint} renvoie {r.status_code}")
            return None
    except Exception as e:
        print(f"❌ Erreur requête : {e}")
        return None


def fetch_authors():
    """Récupère la liste des auteurs."""
    data = get("/author")
    if data and "authors" in data:
        return data["authors"]
    return []


def fetch_poems():
    """Scraping robuste via auteurs (plus fiable)."""
    print("📡 Récupération des auteurs...")
    authors = fetch_authors()
    
    print(f"✔️ {len(authors)} auteurs trouvés")
    all_poems = []

    for author in authors:
        print(f"➡️ Auteur : {author}")
        poems = get(f"/author/{author}")
        sleep(0.3)

        if isinstance(poems, list):
            for p in poems:
                if isinstance(p, dict) and "title" in p:
                    all_poems.append(p)

    print(f"\n📚 Total poèmes collectés : {len(all_poems)}")

    os.makedirs("data/raw", exist_ok=True)

    # Save JSON
    with open(RAW_JSON, "w", encoding="utf-8") as f:
        json.dump(all_poems, f, indent=4)

    # Normalize for CSV
    df = pd.json_normalize(all_poems)
    df.to_csv(RAW_CSV, index=False, encoding="utf-8")

    print("💾 Fichiers enregistrés dans data/raw/")
    print(f"📌 JSON : {RAW_JSON}")
    print(f"📌 CSV  : {RAW_CSV}")


if __name__ == "__main__":
    fetch_poems()
