# src/centroids.py
from sklearn.preprocessing import normalize
import numpy as np

def build_emotion_centroids(vectorizer, lexicons):
    emotions = list(lexicons.keys())
    centroids = {}

    for emotion in emotions:
        text_title = emotion.replace('_', ' ').title()
        body = lexicons[emotion]["core_words"] + lexicons[emotion]["extended_words"]
        centroid_text = text_title + " " + " ".join(body)

        vec = vectorizer.transform([centroid_text])
        boost = lexicons[emotion]["embedding_weight"] * 8
        vec = vec * boost

        centroids[emotion] = vec

    return centroids