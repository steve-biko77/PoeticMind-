import os
import pandas as pd
import json

# Détecte le dossier racine (PoeticMind/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_poems():
    path = os.path.join(BASE_DIR, "data", "cleaned", "clean_poems.csv")
    df_raw = pd.read_csv(path)
    df = df_raw[['title', 'clean_text', 'lemmatized']].dropna().reset_index(drop=True)
    return df

def load_lexicons():
    path = os.path.join(BASE_DIR, "data", "lexic", "lexicons.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
