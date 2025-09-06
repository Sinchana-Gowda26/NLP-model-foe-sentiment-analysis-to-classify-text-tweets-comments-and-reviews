# Sentiment Analysis Project

Classifies text (tweets, reviews, comments) into:
- Positive
- Neutral
- Negative

## Quick Start (Windows)

1. Run this script:
   python setup_project.py

2. Move into folder:
   cd sentiment_project_full

3. Create virtual env:
   python -m venv .venv
   .\.venv\Scripts\activate

4. Install dependencies:
   pip install --upgrade pip
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"

5. Train BiLSTM model:
   python -m training.train_bilstm --small

6. Run app:
   streamlit run app.py
