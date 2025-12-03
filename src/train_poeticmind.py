from src.data_loader import load_poems, load_lexicons
from src.vectorization import build_vectorizer, vectorize_poems
from src.centroids import build_emotion_centroids
from src.inference import compute_similarities, compute_predictions
from src.visualisation import plot_confidence_distribution, plot_similarity_heatmap, plot_3d_clusters, plot_emotion_distribution
from src.save_model import save_pipeline

print("=== Chargement ===")
df = load_poems()
lexicons = load_lexicons()
emotions = list(lexicons.keys())

print("=== Vectorisation ===")
vectorizer = build_vectorizer()
X_poems = vectorize_poems(df, vectorizer)

print("=== Centroides ===")
centroids = build_emotion_centroids(vectorizer, lexicons)

print("=== Similarités + Prédictions ===")
similarities = compute_similarities(X_poems, centroids, emotions)
preds, conf = compute_predictions(similarities, emotions)
df["predicted_emotion"] = preds
df["confidence"] = conf

print("=== Performances ===")
print("Distribution des émotions (en %):")
print(df['predicted_emotion'].value_counts(normalize=True) * 100)
print(f"\nConfiance moyenne: {df['confidence'].mean():.2f}")
print(f"Confiance médiane: {df['confidence'].median():.2f}")
print(f"Confiance min/max: {df['confidence'].min():.2f} / {df['confidence'].max():.2f}")

print("=== Visualisations ===")
plot_confidence_distribution(df)
plot_similarity_heatmap(similarities, emotions)
plot_emotion_distribution(df)
plot_3d_clusters(X_poems, df, emotions)

print("=== Sauvegarde ===")
save_pipeline(vectorizer, X_poems, centroids, emotions, df)