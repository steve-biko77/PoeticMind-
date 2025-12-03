# src/inference.py
import numpy as np

def compute_similarities(X_poems, emotion_vectors, emotions):
    similarities = np.zeros((X_poems.shape[0], len(emotions)))

    for i, emotion in enumerate(emotions):
        vec = emotion_vectors[emotion]
        dot = X_poems.dot(vec.T)
        similarities[:, i] = dot.toarray().ravel()

    return similarities

def compute_predictions(similarities, emotions):
    row_sums = similarities.sum(axis=1).reshape(-1,1)
    probs = similarities / row_sums

    predictions = np.array(emotions)[np.argmax(similarities, axis=1)]
    confidence = np.max(probs, axis=1)

    return predictions, confidence