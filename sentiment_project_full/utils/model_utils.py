import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

def vader_scores(text: str) -> dict:
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text or "")
