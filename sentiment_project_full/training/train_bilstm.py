import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

# ---- Fix path issue so utils is always found ----
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.text_utils import clean_text

# -----------------------------
# Training Parameters
# -----------------------------
MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
EPOCHS = 3
BATCH_SIZE = 32

# -----------------------------
# Sample Data (replace with real dataset later)
# -----------------------------
texts = [
    "I love this movie!",
    "This film was terrible.",
    "Amazing performance and great direction.",
    "Worst plot ever, I hated it.",
    "It was okay, not the best but fine."
]
labels = [1, 0, 1, 0, 1]  # 1 = positive, 0 = negative

# -----------------------------
# Preprocessing
# -----------------------------
texts = [clean_text(t) for t in texts]

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

labels = np.array(labels)

# -----------------------------
# Model
# -----------------------------
model = Sequential([
    Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("Training BiLSTM model...")
model.fit(padded_sequences, labels, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# -----------------------------
# Save Model + Tokenizer
# -----------------------------
artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

model.save(os.path.join(artifacts_dir, "bilstm_model.h5"))

import pickle
with open(os.path.join(artifacts_dir, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ BiLSTM model and tokenizer saved successfully!")
