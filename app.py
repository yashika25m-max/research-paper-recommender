import streamlit as st
import pickle
import numpy as np
import nltk
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords (for cloud)
nltk.download('stopwords', quiet=True)

# =========================
# LOAD FILES
# =========================
model = pickle.load(open('model.pkl', 'rb'))
df = pickle.load(open('data.pkl', 'rb'))

# =========================
# TEXT CLEANING
# =========================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# =========================
# VECTORIZER (recreated)
# =========================
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(df['cleaned'])

X = vectorizer.transform(df['cleaned'])

# =========================
# STREAMLIT UI
# =========================
st.title("📚 Research Paper Recommender")
st.write("Enter an abstract and get subject + similar papers")

user_input = st.text_area("✍️ Enter research abstract")

if st.button("Predict & Recommend"):

    if user_input.strip() == "":
        st.warning("Please enter some text!")
    
    else:
        # Clean input
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])

        # Predict category
        pred = model.predict(vec)[0]

        # Label mapping (optional)
        label_map = {
            "cs.AI": "Artificial Intelligence",
            "cs.LG": "Machine Learning",
            "cs.CV": "Computer Vision",
            "cs.CL": "Natural Language Processing"
        }

        pred_label = label_map.get(pred, pred)

        st.subheader("📌 Predicted Subject")
        st.success(pred_label)

        # =========================
        # RECOMMENDATION
        # =========================
        sim_scores = cosine_similarity(vec, X)

        top_indices = np.argsort(sim_scores[0])[-5:][::-1]

        st.subheader("📄 Recommended Papers")

        for i in top_indices:
            st.write(f"👉 {df['title'].iloc[i]}")
