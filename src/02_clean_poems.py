import pandas as pd
import os
import re
import nltk
import spacy
from nltk.corpus import stopwords

RAW_CSV = "data/raw/poems.csv"
CLEAN_CSV = "data/cleaned/clean_poems.csv"

# Préparation NLP
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    """Nettoyage de base : minuscules, suppression ponctuation & caractères spéciaux."""
    
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_and_lemmatize(text):
    """Tokenisation + lemmatisation + suppression stopwords."""
    
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc 
        if token.lemma_.isalpha() and token.lemma_ not in STOPWORDS
    ]
    return " ".join(tokens)


def poem_length(lines):
    """Compte des mots et lignes."""
    if isinstance(lines, list):
        joined = " ".join(lines)
    else:
        joined = str(lines)

    words = joined.split()
    return len(words), len(joined.split("\n"))


def build_clean_dataset():
    print("📥 Chargement du dataset brut...")
    df = pd.read_csv(RAW_CSV)

    # Certains datasets ont 'lines' sous forme de liste, d'autres string
    df["text"] = df["lines"].apply(lambda x: " ".join(eval(x)) if isinstance(x, str) and x.startswith("[") else str(x))

    print(" Nettoyage du texte brut...")
    df["clean_text"] = df["text"].apply(preprocess_text)

    print(" Tokenisation + lemmatisation...")
    df["lemmatized"] = df["clean_text"].apply(tokenize_and_lemmatize)

    print("📏 Calcul métriques...")
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df["line_count"] = df["text"].apply(lambda x: str(x).count("\n") + 1)

    print("💾 Sauvegarde...")
    os.makedirs("data/cleaned", exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False, encoding="utf-8")

    print("✔️ Dataset nettoyé disponible :")
    print(f" {CLEAN_CSV}")


if __name__ == "__main__":
    build_clean_dataset()
