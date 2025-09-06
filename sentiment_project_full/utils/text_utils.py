import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """
    Cleans input text: lowercasing, removing punctuation/numbers,
    removing stopwords.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)   # remove punctuation/numbers
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)
