
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("tourism_iternary_dataset.csv")
df = df.dropna(subset=["input__destination", "input__duration", "input__interests__001"])
df["input__duration"] = df["input__duration"].astype(float)
df["BudgetEstimate"] = df["input__duration"] * 2000  # INR/day estimation

# Combine all interests
interest_cols = [col for col in df.columns if "interests" in col]
df["all_interests"] = df[interest_cols].fillna("").agg(" ".join, axis=1)

# Train TF-IDF model
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["all_interests"])

# UI
st.markdown("<h1 style='text-align:center; color:#ff5733;'>🇮🇳 India Travel Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter your interests and budget to get destination suggestions.</p>", unsafe_allow_html=True)

user_input = st.text_input("Your travel interests?", "history architecture")
user_budget = st.slider("Select your budget (in ₹)", 1000, 20000, 8000, step=500)

if user_input:
    user_vec = tfidf.transform([user_input])
    df["similarity"] = cosine_similarity(user_vec, tfidf_matrix).flatten()
    results = df[df["BudgetEstimate"] <= user_budget].sort_values(by="similarity", ascending=False).head(5)

    if not results.empty:
        for _, row in results.iterrows():
            st.subheader(f"📍 {row['input__destination']} — {int(row['input__duration'])} days")
            st.markdown(f"**Morning:** {row['output__optimized_itinerary__days__morning']}")
            st.markdown(f"**Afternoon:** {row['output__optimized_itinerary__days__afternoon']}")
            st.markdown(f"**Evening:** {row['output__optimized_itinerary__days__evening']}")
            st.markdown("---")
    else:
        st.warning("No destinations match your interest and budget.")
