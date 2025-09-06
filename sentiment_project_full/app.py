import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pypdf import PdfReader
from googletrans import Translator   # Multilanguage support

# --- NLTK setup (fixes punkt_tab error) ---
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')

# --- Load BiLSTM model + tokenizer ---
bilstm_model = None
tokenizer = None
try:
    bilstm_model = tf.keras.models.load_model("artifacts/bilstm_model.h5")
    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except Exception:
    pass  # model not available yet

# --- Sentiment analyzers ---
vader = SentimentIntensityAnalyzer()
translator = Translator()

def translate_text(text, target_lang="en"):
    """Translate input text into English for analysis."""
    try:
        translated = translator.translate(text, dest=target_lang)
        return translated.text
    except:
        return text  # fallback

def analyze_with_vader(text):
    sentences = sent_tokenize(text)
    results = []
    for sent in sentences:
        score = vader.polarity_scores(sent)
        label = "Positive" if score["compound"] > 0.05 else "Negative" if score["compound"] < -0.05 else "Neutral"
        results.append((sent, label, score))
    return results

def analyze_with_bilstm(text):
    if not bilstm_model or not tokenizer:
        return [("Model not trained", "Error", {})]
    sentences = sent_tokenize(text)
    results = []
    for sent in sentences:
        seq = tokenizer.texts_to_sequences([sent])
        padded = pad_sequences(seq, maxlen=100)
        pred = bilstm_model.predict(padded, verbose=0)[0]
        label_idx = np.argmax(pred)
        label = ["Negative", "Neutral", "Positive"][label_idx]
        results.append((sent, label, {"probabilities": pred.tolist()}))
    return results

def process_text_input(user_text, model_choice):
    translated_text = translate_text(user_text, target_lang="en")
    if model_choice == "VADER":
        results = analyze_with_vader(translated_text)
    else:
        results = analyze_with_bilstm(translated_text)
    return results, translated_text

# --- Streamlit UI ---
st.title("🌍 Multilingual Sentiment Analysis (VADER vs BiLSTM)")
st.sidebar.header("Choose Input")

model_choice = st.sidebar.radio("Select Model", ["VADER", "BiLSTM"])
input_type = st.sidebar.radio("Choose Input", ["Text", "URL", "PDF", "CSV"])

text_data = ""

# --- Input options ---
if input_type == "Text":
    text_data = st.text_area("Enter text here (any language):")

elif input_type == "URL":
    url = st.text_input("Enter URL here:")
    if url:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = " ".join([p.get_text() for p in soup.find_all("p")])
            text_data = paragraphs
        except Exception as e:
            st.error(f"Could not fetch URL: {e}")

elif input_type == "PDF":
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file:
        try:
            reader = PdfReader(pdf_file)
            pdf_text = ""
            for page in reader.pages:
                if page.extract_text():
                    pdf_text += page.extract_text()
            text_data = pdf_text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")

elif input_type == "CSV":
    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    if csv_file:
        try:
            df = pd.read_csv(csv_file)
            st.write("CSV Preview:", df.head())
            if "comment" in df.columns:
                text_data = " ".join(df["comment"].astype(str).tolist())
            else:
                column = st.selectbox("Select column for analysis", df.columns)
                text_data = " ".join(df[column].astype(str).tolist())
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# --- Analysis button ---
if st.button("Analyze"):
    if text_data.strip() == "":
        st.warning("No text found to analyze.")
    else:
        results, translated_text = process_text_input(text_data, model_choice)
        st.write("### Results")
        st.info(f"🔤 Translated text used for analysis (English): \n\n{translated_text[:500]}...")
        for sent, label, score in results:
            st.write(f"**{sent}** → {label}")
            st.json(score)
