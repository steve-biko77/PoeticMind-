# dashboard/app.py (VERSION BEAUTÉ ABSOLUE – À REMPLACER)
import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.manifold import TSNE

# ---------- NLTK ----------
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------- Config ----------
st.set_page_config(
    page_title="PoeticMind ⋅ Salon des Âmes",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🖤"
)

# ---------- CSS MAGIQUE (dark poetic luxury) ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap');

    :root {
        --bg: #0a0a0d;
        --card: #111114;
        --border: rgba(193,154,107,0.15);
        --gold: #c19a6b;
        --gold-light: #e2d0b4;
        --purple: #8e44ad;
        --text: #e8e6e3;
        --muted: #a09a8f;
    }

    .main > div {
        background: linear-gradient(135deg, #0a0a0d 0%, #121015 100%);
        color: var(--text);
    }

    /* Sidebar luxueuse */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #0c0c10, #080809);
        border-right: 1px solid var(--border);
    }
    .sidebar-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 2.4rem;
        font-weight: 600;
        background: linear-gradient(90deg, var(--gold), var(--gold-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1px;
    }

    /* Titres poétiques */
    .poetic-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.2rem;
        font-weight: 700;
        letter-spacing: 1px;
        color: #f0e6d2;
        text-align: center;
        margin: 20px 0;
        position: relative;
    }
    .poetic-title::after {
        content: '';
        position: absolute;
        bottom: -12px;
        left: 50%;
        transform: translateX(-50%);
        width: 120px;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--gold), transparent);
    }

    .subtitle {
        text-align: center;
        font-style: italic;
        color: var(--muted);
        font-size: 1.15rem;
        margin-bottom: 30px;
    }

    /* Cards en verre fumé */
    .glass-card {
        background: rgba(17, 17, 20, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.7);
        margin-bottom: 24px;
    }

    /* Boutons précieux */
    .stButton > button {
        background: linear-gradient(135deg, #c19a6b, #8e44ad);
        color: white !important;
        font-family: 'Cormorant Garamond', serif;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(193,154,107,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(193,154,107,0.4);
    }

    /* Résultat émotion */
    .emotion-result {
        font-size: 2.2rem;
        font-family: 'Playfair Display', serif;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, rgba(193,154,107,0.15), rgba(142,68,173,0.15));
        border-radius: 16px;
        border: 1px solid var(--gold);
        color: var(--gold-light);
        margin: 20px 0;
    }

    /* Plotly dark perfection */
    .js-plotly-plot .plotly { border-radius: 16px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------- Chargement modèle ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'poeticmind_v2')

@st.cache_resource
def load_model():
    vectorizer = joblib.load(os.path.join(MODEL_PATH, 'vectorizer.joblib'))
    centroids = joblib.load(os.path.join(MODEL_PATH, 'emotion_centroids.joblib'))
    emotions = joblib.load(os.path.join(MODEL_PATH, 'emotion_labels.joblib'))
    meta_df = joblib.load(os.path.join(MODEL_PATH, 'metadata_poems.joblib'))
    return vectorizer, centroids, emotions, meta_df

vectorizer, centroids, emotions, meta_df = load_model()

# ---------- Fonctions NLP ----------
def clean_and_lemmatize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(t) for t in tokens])

def vectorize_new_poem(title, body, vectorizer):
    X_title = vectorizer.transform([title])
    X_body = vectorizer.transform([body])
    X = 0.60 * X_title + 0.40 * X_body
    return normalize(X, norm='l2')

def predict_emotion(X, centroids, emotions):
    sims = np.array([X.dot(centroids[e].T).toarray()[0][0] for e in emotions])
    probs = sims / sims.sum()
    idx = np.argmax(sims)
    return emotions[idx], probs[idx]

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>PoeticMind</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#a09a8f; font-size:0.95rem; margin-top:-8px;'>Salon des émotions cachées</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", ["🎭 Prédire", "🕯️ Visualisations", "📜 À propos"], label_visibility="collapsed")

# ---------- Header ----------
st.markdown("<h1 class='poetic-title'>PoeticMind</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Une intelligence qui lit entre les vers</p>", unsafe_allow_html=True)

# ========== PAGES ==========
if page == "🎭 Prédire":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Entrez votre poème")

        title = st.text_input("**Titre**", placeholder="ex. « Sonnet au crépuscule »", label_visibility="collapsed")
        poem = st.text_area("**Texte complet**", height=280,
                            placeholder="Collez ici vos vers, vos soupirs, vos ombres...", label_visibility="collapsed")

        if st.button("✨ Révéler l’âme du poème", use_container_width=True):
            if title and poem:
                with st.spinner("Je ferme les yeux… je lis entre les lignes…"):
                    clean_poem = clean_and_lemmatize(poem)
                    X = vectorize_new_poem(title, clean_poem, vectorizer)
                    emotion, confidence = predict_emotion(X, centroids, emotions)

                st.markdown(f"<div class='emotion-result'>{emotion.replace('_', ' ').title()}<br><small>Confiance : {confidence:.1%}</small></div>", unsafe_allow_html=True)

                # Poèmes similaires
                st.markdown("### Poèmes qui résonnent avec le vôtre")
                sim = meta_df[meta_df['predicted_emotion'] == emotion] \
                      .sort_values('confidence', ascending=False).head(6)

                for _, row in sim.iterrows():
                    st.markdown(f"• **{row['title']}** — {row['confidence']:.2%} d’affinité")

                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("Un poème sans titre ni vers est une âme sans corps…")

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Conseil du salon")
        st.markdown("""
        - Le **titre** porte 60 % de l’émotion  
        - Écrivez avec votre cœur, pas avec des filtres  
        - Les poèmes courts (< 30 mots) peuvent être plus difficiles à lire  
        - Les émotions dominantes : amour, mélancolie, mort, nature, passion…
        """)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "🕯️ Visualisations":
    st.markdown("### La cartographie secrète des émotions")

    # 1. Bar chart luxueux
    counts = meta_df['predicted_emotion'].value_counts().reindex(emotions, fill_value=0)
    colors = ['#e91e63','#9c27b0','#212121','#607d8b','#4caf50','#ff5722','#795548','#ffeb3b','#03a9f4']

    fig = go.Figure(go.Bar(
        x=[e.replace('_',' ').title() for e in emotions],
        y=counts.values,
        marker_color=colors,
        text=counts.values,
        textposition='outside'
    ))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title="Répartition des âmes poétiques",
        title_font=dict(size=22, family="Playfair Display", color="#e8e6e3"),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # 2. Histogramme confiance
    st.markdown("### Confiance du modèle dans ses lectures")
    fig2, ax = plt.subplots(figsize=(10,5), facecolor='#0a0a0d')
    sns.histplot(meta_df['confidence'], bins=35, kde=True, color="#c19a6b", ax=ax, alpha=0.8)
    ax.set_facecolor('#0a0a0d')
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(colors='#a09a8f')
    ax.set_title("Distribution des certitudes", color="#e8e6e3", fontsize=18, pad=20)
    st.pyplot(fig2, use_container_width=True)

    # 3. t-SNE 3D interactif
    st.markdown("### Nuage d’émotions en trois dimensions")
    st.info("900 poèmes projetés dans l’espace des sentiments – survolez pour découvrir les titres")

    X_poems = joblib.load(os.path.join(MODEL_PATH, 'X_poems_sparse.joblib'))
    sample = min(900, X_poems.shape[0])
    X_sample = X_poems[:sample].toarray()
    tsne = TSNE(n_components=3, perplexity=30, random_state=42, n_iter=1000)
    X_3d = tsne.fit_transform(X_sample)

    fig3 = go.Figure()
    color_map = {
        'love_romantic': '#e91e63', 'melancholy_nostalgia': '#9c27b0', 'death_mortality': '#212121',
        'time_transience': '#607d8b', 'nature_sublime': '#4caf50', 'passion_erotic': '#ff5722',
        'solitude_exile': '#795548', 'joy_ecstasy': '#ffeb3b', 'spirituality_devotion': '#03a9f4'
    }

    for e in emotions:
        mask = meta_df['predicted_emotion'][:sample] == e
        if mask.sum() == 0: continue
        fig3.add_trace(go.Scatter3d(
            x=X_3d[mask,0], y=X_3d[mask,1], z=X_3d[mask,2],
            mode='markers',
            name=e.replace('_',' ').title(),
            marker=dict(size=5, color=color_map[e], opacity=0.8),
            text=meta_df['title'][:sample][mask],
            hovertemplate="<b>%{text}</b><br>%{name}<extra></extra>"
        ))

    fig3.update_layout(
        template='plotly_dark',
        scene=dict(bgcolor='rgba(0,0,0,0)'),
        height=700,
        margin=dict(l=0,r=0,t=40,b=0)
    )
    st.plotly_chart(fig3, use_container_width=True)

elif page == "📜 À propos":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("""
    ### PoeticMind ⋅ V2 — Salon des émotions cachées

    Une intelligence artificielle formée à lire **l’âme** des poèmes anglais classiques  
    grâce à un modèle **lexicon-guided** (centroïdes émotionnels) et un vectoriseur affiné.

    - **9 émotions** principales détectées  
    - Titre pondéré à **60 %** (car il porte souvent la clé)  
    - Esthétique sombre inspirée des salons littéraires du XIXe siècle

    > « La poésie est l’émotion recueillie dans la tranquillité » — Wordsworth  
    > Ici, nous recueillons l’émotion… puis nous la nommons.

    **Dataset** : plusieurs milliers de poèmes classiques (XVIIe-XXe siècles)  
    **Design** : fait main avec amour, bougies et vieux livres.
    """)
    st.markdown(f"**Émotions reconnues** : {', '.join([e.replace('_',' ').title() for e in emotions])}")
    st.markdown(f"**Poèmes dans le corpus** : {len(meta_df):,}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<p style='text-align:center; color:#777; margin-top:40px;'>© 2025 — PoeticMind ⋅ Dans l’ombre des vers, la lumière des sentiments</p>", unsafe_allow_html=True)