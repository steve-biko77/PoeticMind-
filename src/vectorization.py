# src/vectorization.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

def build_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        min_df=3,
        max_df=0.95,
        ngram_range=(1,2),
        sublinear_tf=True
    )

def vectorize_poems(df, vectorizer):
    corpus = df['title'].fillna("") + " " + df['lemmatized'].fillna("")
    vectorizer.fit(corpus)

    X_title = vectorizer.transform(df['title'].fillna(""))
    X_body  = vectorizer.transform(df['lemmatized'].fillna(""))

    X_poems = 0.60 * X_title + 0.40 * X_body
    X_poems = normalize(X_poems, norm='l2')

    return X_poems