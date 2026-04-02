import streamlit as st
import numpy as np
import pickle
import bs4 

# Page config
st.set_page_config(
    page_title="Auto Tagging System",
    page_icon="🔖",
    layout="centered"
)

# Custom CSS (UI MAGIC ✨)
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1 {
    color: #ffffff;
    text-align: center;
}
.tag-box {
    padding: 8px 15px;
    border-radius: 20px;
    background-color: #1f77b4;
    color: white;
    display: inline-block;
    margin: 5px;
    font-size: 14px;
}
textarea {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# Load models
lr_model = pickle.load(open("lr_model.pkl", "rb"))
word_vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
mlb = pickle.load(open("mlb.pkl", "rb"))

all_tags = mlb.classes_

# Clean text
def clean_text(text):
    bs4.BeautifulSoup(text, "html.parser").get_text()
    return text.lower()

# Tagging function
def tagging(question, top_k=5):
    cleaned_question = clean_text(question)

    vector = word_vectorizer.transform([cleaned_question])
    pred_prob = lr_model.predict_proba(vector)[0]

    top_indices = np.argsort(pred_prob)[-top_k:][::-1]
    tags = [(all_tags[i], pred_prob[i]) for i in top_indices]

    return tags

# Header
st.title("🔖 Auto Tagging System")
st.markdown("### 🚀 Smart NLP-based Tag Generator")

# Input box
user_input = st.text_area(
    "Enter your question:",
    placeholder="e.g. Explain Bayesian vs Frequentist reasoning..."
)

# Button
if st.button("✨ Generate Tags"):
    if user_input.strip() != "":
        tags = tagging(user_input)

        st.markdown("### 🎯 Predicted Tags")

        # Display tags as stylish pills
        for tag, score in tags:
            st.markdown(
                f'<span class="tag-box">{tag} ({round(score,2)})</span>',
                unsafe_allow_html=True
            )

        # Confidence bar chart
        st.markdown("### 📊 Confidence Scores")
        tag_names = [t[0] for t in tags]
        tag_scores = [t[1] for t in tags]

        st.bar_chart(tag_scores)

    else:
        st.warning("⚠️ Please enter some text")
