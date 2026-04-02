🔖 Auto Tagging System using NLP & Machine Learning
🚀 Overview

This project is a Multilabel Auto-Tagging System that automatically predicts relevant tags for user-input questions, similar to platforms like StackOverflow.

It uses Natural Language Processing (NLP) techniques and Machine Learning to analyze text and generate multiple meaningful tags in real-time.

🎯 Features
🔍 Multilabel classification (predicts multiple tags per input)
⚡ Real-time tag generation
📊 Confidence score visualization
🧠 NLP preprocessing (cleaning, HTML removal)
🎨 Interactive web app using Streamlit
📈 Top-K tag selection for better results
🛠️ Tech Stack
Python 🐍
Scikit-learn
TF-IDF Vectorization
Logistic Regression (OneVsRest)
NumPy
BeautifulSoup
Streamlit
🧠 How It Works
Input Processing
User enters a question
HTML tags and noise are removed
Feature Extraction
Text converted into numerical features using TF-IDF
Model Prediction
Logistic Regression predicts probabilities for each tag
Tag Selection
Top-K + threshold-based filtering applied
Output
Relevant tags displayed with confidence scores
📸 Screenshots
🖥️ Web App Interface

🎯 Tag Prediction Output

📊 Confidence Visualization

🌍 Live Demo


👉 Add your deployed link here
Example:

https://your-app-name.streamlit.app
📂 Project Structure
auto-tagging-app/
│── app.py
│── lr_model.pkl
│── vectorizer.pkl
│── mlb.pkl
│── requirements.txt
│── screenshots/
📈 Future Improvements
🔥 Use BERT/transformers for better accuracy
🎯 Per-label threshold tuning
📊 Precision@K evaluation
🌐 Deploy with custom domain
🧠 Improve handling of rare tags
💼 Resume Highlight

Developed a multilabel NLP-based auto-tagging system using TF-IDF and Logistic Regression, deployed as an interactive Streamlit web app with real-time predictions and visualization.

🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

⭐ Show Your Support

If you like this project, please ⭐ the repo!
