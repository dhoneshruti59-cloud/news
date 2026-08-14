import streamlit as st
import joblib

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article below to predict whether "
    "the news is Fake or Real."
)

news = st.text_area(
    "Enter News Article:",
    height=250,
    placeholder="Paste the news article here..."
)

if st.button("🔍 Check News"):

    if news.strip() == "":
        st.warning("⚠️ Please enter a news article.")

    else:

        news_vector = vectorizer.transform([news])

        prediction = model.predict(news_vector)[0]

        probability = model.predict_proba(news_vector).max() * 100

        if prediction == "FAKE":

            st.error("❌ FAKE NEWS")
            st.write(f"Prediction confidence: {probability:.2f}%")

        else:

            st.success("✅ REAL NEWS")
            st.write(f"Prediction confidence: {probability:.2f}%")