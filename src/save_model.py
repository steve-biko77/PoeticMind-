# src/save_model.py
import joblib
import os
import shutil

# Détection automatique du dossier racine du projet (PoeticMind/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save_pipeline(vectorizer, X_poems, centroids, emotions, df,
                  relative_path="models/poeticmind_v2"):

    # Dossier de sauvegarde propre et indépendant de l’endroit d’exécution
    path = os.path.join(BASE_DIR, relative_path)

    os.makedirs(path, exist_ok=True)

    # Sauvegardes principales
    joblib.dump(vectorizer, os.path.join(path, "vectorizer.joblib"))
    joblib.dump(X_poems, os.path.join(path, "X_poems_sparse.joblib"))
    joblib.dump(centroids, os.path.join(path, "emotion_centroids.joblib"))
    joblib.dump(emotions, os.path.join(path, "emotion_labels.joblib"))

    # Métadonnées utiles pour supervision / debug
    meta_df = df[['title', 'lemmatized', 'predicted_emotion', 'confidence']]
    joblib.dump(meta_df, os.path.join(path, "metadata_poems.joblib"))

    # Copie du lexique émotionnel
    src_lex = os.path.join(BASE_DIR, "data", "lexic", "lexicons.json")
    dst_lex = os.path.join(path, "lexicons.json")
    shutil.copy(src_lex, dst_lex)

    print(f"⚡ Pipeline PoeticMind V2 sauvegardé dans : {path}")