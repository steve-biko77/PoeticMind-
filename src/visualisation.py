# src/visualisation.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.manifold import TSNE
import plotly.graph_objects as go

def plot_confidence_distribution(df, save_path="../docs"):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(12,6))
    sns.histplot(df['confidence'], bins=30, kde=True, color="navy")
    plt.title("Distribution des confiances")
    plt.xlabel("Confiance")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confidence_distribution.png"))
    plt.show()

def plot_similarity_heatmap(similarities, emotions, save_path="../docs"):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(14,8))
    sns.heatmap(similarities[:50], cmap="viridis", xticklabels=emotions)
    plt.title("Similarités (50 premiers poèmes)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "similarity_heatmap.png"))
    plt.show()

def plot_emotion_distribution(df, save_path="../docs"):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(12,6))
    sns.countplot(data=df, x='predicted_emotion', order=df['predicted_emotion'].value_counts().index, hue='predicted_emotion', palette="husl", legend=False)
    plt.title("Distribution des émotions prédites")
    plt.xlabel("Émotion")
    plt.ylabel("Nombre de poèmes")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "emotion_distribution.png"))
    plt.show()

def plot_3d_clusters(X_poems, df, emotions, save_path="../docs"):
    os.makedirs(save_path, exist_ok=True)

    sample = min(3000, X_poems.shape[0])
    X_sample = X_poems[:sample].toarray()
    labels_sample = df['predicted_emotion'][:sample]

    tsne = TSNE(n_components=3, random_state=42, perplexity=40)
    X_3d = tsne.fit_transform(X_sample)

    fig = go.Figure()

    color_map = {
        'love_romantic': '#e91e63',
        'melancholy_nostalgia': '#9c27b0',
        'death_mortality': '#212121',
        'time_transience': '#607d8b',
        'nature_sublime': '#4caf50',
        'passion_erotic': '#ff5722',
        'solitude_exile': '#795548',
        'joy_ecstasy': '#ffeb3b',
        'spirituality_devotion': '#03a9f4'
    }

    for e in emotions:
        mask = labels_sample == e
        fig.add_trace(go.Scatter3d(
            x=X_3d[mask,0], y=X_3d[mask,1], z=X_3d[mask,2],
            mode="markers",
            name=e.replace("_"," ").title(),
            marker=dict(size=5, color=color_map.get(e, '#000000'), opacity=0.75)
        ))

    fig.write_html(os.path.join(save_path, "clusters_3D.html"))