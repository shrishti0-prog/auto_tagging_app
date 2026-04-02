# auto_tagging_app
This project is an NLP-based multilabel classification system designed to automatically generate relevant tags for textual questions, similar to platforms like StackOverflow.  The model processes raw text input, performs preprocessing (HTML removal, cleaning), and converts it into numerical features using TF-IDF vectorization.A Logistic Regression model trained using a One-vs-Rest strategy predicts multiple relevant tags for each input.

The application is deployed as an interactive web app using Streamlit, where users can input questions and receive predicted tags along with confidence scores. The system also includes threshold tuning and Top-K tag selection for improved performance.

This project demonstrates practical implementation of NLP, multilabel classification, model optimization, and end-to-end deployment.
Tech Stack:
- Python
- Scikit-learn
- NLP (TF-IDF, text preprocessing)
- Streamlit
- NumPy
- BeautifulSoup
- Key Features:
- Multilabel classification
- Real-time tag prediction
- Confidence score visualization
- Interactive web interface
