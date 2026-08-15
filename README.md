# NLP Sentiment Analysis

An NLP-based sentiment analysis application that classifies text
into Positive, Neutral, and Negative sentiments.

## Features

- Sentiment analysis using VADER and BiLSTM
- Text input analysis
- PDF and CSV support
- URL-based text extraction
- Multilingual input through translation
- Streamlit web interface

## Technologies

- Python
- TensorFlow
- NLTK
- Streamlit
- Pandas
- NumPy
- Scikit-learn

## How It Works

User Input
→ Text Preprocessing
→ Translation (when required)
→ VADER + BiLSTM
→ Sentiment Classification

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
