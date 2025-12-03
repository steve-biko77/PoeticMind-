

# 📘 **PoeticMind V2 — Analyse Émotionnelle & Recommandation Poétique**

![Emotion Distribution](preview/preview.png)
PoeticMind V2 est un moteur d'analyse poétique combinant NLP classique et clustering émotionnel.
Il transforme des poèmes en vecteurs TF-IDF, détecte l’émotion dominante, calcule un score de confiance et génère des recommandations par proximité vectorielle.

Ce projet vise à créer un système capable de comprendre la structure émotionnelle d’un texte et d’offrir un outil d’exploration artistique assisté par IA.

---

## 🚀 **Fonctionnalités principales**

* **Nettoyage & normalisation** des poèmes
* **Tokenisation, lemmatisation, stopwords**
* **Extraction TF-IDF**
* **Clustering émotionnel (centroïdes)**
* **Prédiction : sentiment / émotion dominante**
* **Calcul de confiance**
* **Génération automatique de graphes**
* **Dashboard Preview (Plotly / Matplotlib)**
* **Sauvegarde complète du pipeline** (`vectorizer`, `centroids`, metadata, lexiques)

---

## 📂 **Arborescence du projet**

```
PoeticMind/
│
├── data/
│   ├── raw/              # Données brutes
│   ├── cleaned/          # Données nettoyées
│   └── lexic/            # Lexiques émotionnels
│
├── models/
│   └── poeticmind_v2/    # Vectorizer, matrices sparse, centroids, metadata
│
├── preview/
│   ├── plot_distributions.py
│   ├── plot_clusters.py
│   └── plot_embeddings.py
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── vectorizer.py
│   ├── clustering.py
│   ├── save_model.py
│   ├── visualization.py
│   └── train_poeticmind.py
│
├── requirements.txt
└── README.md
```

---

# 🛠️ **Installation**

### 1. Créer l’environnement virtuel

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

# 📥 **Préparation des données**

Place les données nettoyées ici :

```
data/cleaned/clean_poems.csv
```

Place les lexiques émotionnels ici :

```
data/lexic/lexicons.json
```

---

# 🧠 **Entraîner le modèle**

Depuis la racine du projet :

```bash
python -m src.train_poeticmind
```

Le script :

* charge les poèmes
* prétraite le texte
* vectorise TF-IDF
* calcule les centroïdes émotionnels
* prédit les émotions
* sauvegarde le pipeline dans `models/poeticmind_v2/`
* génère les graphes automatiques dans `preview/`

---

# 📊 **Visualisation & Dashboard (Preview)**

Lancer les visualisations :

```bash
python preview/plot_distributions.py
python preview/plot_clusters.py
python preview/plot_embeddings.py
```

Elles affichent :

* distribution des émotions
* heatmap des centroïdes
* nuage des poèmes vectorisés (PCA/TSNE)

---

# 📦 **Structure du modèle sauvegardé**

Dans `models/poeticmind_v2/` :

| Fichier                    | Description                      |
| -------------------------- | -------------------------------- |
| `vectorizer.joblib`        | TF-IDF fitted                    |
| `X_poems_sparse.joblib`    | Matrice TF-IDF                   |
| `emotion_centroids.joblib` | Centroïdes émotionnels           |
| `emotion_labels.joblib`    | Étiquettes d’émotions            |
| `metadata_poems.joblib`    | Poèmes + prédictions + confiance |
| `lexicons.json`            | Lexiques copiés automatiquement  |

---

# 🧪 **Utilisation du modèle dans un script Python**

```python
import joblib

vectorizer = joblib.load("models/poeticmind_v2/vectorizer.joblib")
centroids = joblib.load("models/poeticmind_v2/emotion_centroids.joblib")
labels = joblib.load("models/poeticmind_v2/emotion_labels.joblib")

text = "Ce soir le vent se perd dans ma mémoire brisée…"

vec = vectorizer.transform([text])

# calcul du score de similarité avec les centroïdes
import numpy as np

distances = np.linalg.norm(vec.toarray() - centroids, axis=1)
emotion = labels[distances.argmin()]

print("Émotion détectée :", emotion)
```

---

# 🗺️ **Roadmap (V3 → V4)**

🟦 **V3 (prochaine étape)**

* ajout de *poetic embeddings* (transformers)
* clustering hybride TF-IDF + embeddings
* dashboard interactif (Streamlit)

🟧 **V4 (objectif long terme)**

* modèle supervisé finement annoté
* classification multi-label
* génération poétique contrôlée par émotion
* assistant PoeticMind (chat + voix)

---

# 👤 **Auteur**

Projet développé par **Boss**
Jeune développeur & roboticien 📡
Objectif : maîtriser IA, technologies avancées et atteindre l’indépendance numérique.

