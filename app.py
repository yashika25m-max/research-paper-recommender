import streamlit as st
import pickle
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

# =========================
# LOAD FILES
# =========================
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
df = pickle.load(open('data.pkl', 'rb'))

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# =========================
# UI
# =========================
st.title("📚 Research Paper Recommender")
st.markdown("Enter an abstract and get subject + similar papers")

user_input = st.text_area("✍️ Enter research abstract")

if st.button("Predict & Recommend"):
    
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])

        # Prediction
        pred = model.predict(vec)[0]

        label_map = {
            "cs.AI": "Artificial Intelligence",
            "cs.LG": "Machine Learning",
            "cs.CV": "Computer Vision",
            "cs.CL": "Natural Language Processing"
        }

        st.subheader("📌 Predicted Subject")
        st.success(label_map.get(pred, pred))

        # Recommendation
        sim_scores = cosine_similarity(vec, vectorizer.transform(df['cleaned']))
        top_indices = sim_scores[0].argsort()[-6:-1][::-1]

        st.subheader("📚 Recommended Papers")

        for i in top_indices:
            score = sim_scores[0][i]
            st.write(f"👉 {df.iloc[i]['title']} (score: {score:.2f})")