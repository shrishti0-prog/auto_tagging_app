import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(
    page_title="Auto Tagging System",
    page_icon="🔖",
    layout="centered"
)

# 🌙 Custom Styling (IMPORTANT)
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.main {
    background-color: #0e1117;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
}
.subtitle {
    text-align: center;
    color: #9aa0a6;
    margin-bottom: 30px;
}
.tag {
    display: inline-block;
    background-color: #1f77b4;
    color: white;
    padding: 8px 14px;
    border-radius: 20px;
    margin: 5px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# Load model
lr_model = pickle.load(open("lr_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
mlb = pickle.load(open("mlb.pkl", "rb"))
all_tags = mlb.classes_

# Clean text
def basic_clean(text):
    return text.lower()

# Tagging function
def tagging(text, top_k=5):
    vector = vectorizer.transform([clean_text(text)])
    probs = lr_model.predict_proba(vector)[0]

    top_idx = np.argsort(probs)[-top_k:][::-1]
    return [(all_tags[i], probs[i]) for i in top_idx]

# 🎯 UI Header
st.markdown('<div class="title">🔖 Auto Tagging System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Generate smart tags using NLP</div>', unsafe_allow_html=True)

# ✍️ Input box
user_input = st.text_area(
    "Enter your question:",
    placeholder="e.g. Explain Bayesian vs Frequentist reasoning..."
)

# ⚙️ Sidebar controls
st.sidebar.title("⚙️ Settings")
top_k = st.sidebar.slider("Number of tags", 1, 10, 5)

# 🚀 Button
if st.button("✨ Generate Tags"):
    if user_input.strip():
        
        with st.spinner("Analyzing..."):
            tags = tagging(user_input, top_k)

        st.markdown("### 🎯 Predicted Tags")

        # Tag pills
        for tag, score in tags:
            st.markdown(
                f'<span class="tag">{tag} ({round(score,2)})</span>',
                unsafe_allow_html=True
            )

        # 📊 Chart
        st.markdown("### 📊 Confidence Scores")
        names = [t[0] for t in tags]
        scores = [t[1] for t in tags]
        st.bar_chart(scores)

    else:
        st.warning("⚠️ Please enter a question")
