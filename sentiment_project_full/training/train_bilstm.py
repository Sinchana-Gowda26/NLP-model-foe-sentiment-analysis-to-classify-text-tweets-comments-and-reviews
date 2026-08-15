import os
import sys
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

# Make sure utils can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.text_utils import clean_text


# -----------------------------
# Training parameters
# -----------------------------

MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
EPOCHS = 10
BATCH_SIZE = 4


# -----------------------------
# Sample training data
# 0 = Negative
# 1 = Neutral
# 2 = Positive
# -----------------------------

texts = [
    "I love this movie",
    "This film was amazing",
    "The acting was excellent",
    "I really enjoyed this product",

    "This movie was okay",
    "The product is average",
    "It was fine, nothing special",
    "The experience was neither good nor bad",

    "I hated this movie",
    "This film was terrible",
    "The product was useless",
    "I am very disappointed"
]

labels = [
    2, 2, 2, 2,
    1, 1, 1, 1,
    0, 0, 0, 0
]


# -----------------------------
# Text preprocessing
# -----------------------------

texts = [clean_text(text) for text in texts]

tokenizer = Tokenizer(
    num_words=MAX_VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

padded_sequences = pad_sequences(
    sequences,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding="post"
)

labels = np.array(labels)


# -----------------------------
# Build BiLSTM model
# -----------------------------

model = Sequential([
    Embedding(
        input_dim=MAX_VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH
    ),

    Bidirectional(
        LSTM(64, return_sequences=True)
    ),

    Dropout(0.3),

    Bidirectional(
        LSTM(32)
    ),

    Dense(32, activation="relu"),

    Dropout(0.3),

    Dense(3, activation="softmax")
])


# -----------------------------
# Compile model
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# Train
# -----------------------------

print("Training BiLSTM model...")

model.fit(
    padded_sequences,
    labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)


# -----------------------------
# Save model and tokenizer
# -----------------------------

artifacts_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "artifacts"
)

os.makedirs(artifacts_dir, exist_ok=True)

model.save(
    os.path.join(
        artifacts_dir,
        "bilstm_model.h5"
    )
)

with open(
    os.path.join(artifacts_dir, "tokenizer.pkl"),
    "wb"
) as f:
    pickle.dump(tokenizer, f)


print("BiLSTM model and tokenizer saved successfully.")import os
import sys
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

# Make sure utils can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.text_utils import clean_text


# -----------------------------
# Training parameters
# -----------------------------

MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
EPOCHS = 10
BATCH_SIZE = 4


# -----------------------------
# Sample training data
# 0 = Negative
# 1 = Neutral
# 2 = Positive
# -----------------------------

texts = [
    "I love this movie",
    "This film was amazing",
    "The acting was excellent",
    "I really enjoyed this product",

    "This movie was okay",
    "The product is average",
    "It was fine, nothing special",
    "The experience was neither good nor bad",

    "I hated this movie",
    "This film was terrible",
    "The product was useless",
    "I am very disappointed"
]

labels = [
    2, 2, 2, 2,
    1, 1, 1, 1,
    0, 0, 0, 0
]


# -----------------------------
# Text preprocessing
# -----------------------------

texts = [clean_text(text) for text in texts]

tokenizer = Tokenizer(
    num_words=MAX_VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

padded_sequences = pad_sequences(
    sequences,
    maxlen=MAX_SEQUENCE_LENGTH,
    padding="post"
)

labels = np.array(labels)


# -----------------------------
# Build BiLSTM model
# -----------------------------

model = Sequential([
    Embedding(
        input_dim=MAX_VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH
    ),

    Bidirectional(
        LSTM(64, return_sequences=True)
    ),

    Dropout(0.3),

    Bidirectional(
        LSTM(32)
    ),

    Dense(32, activation="relu"),

    Dropout(0.3),

    Dense(3, activation="softmax")
])


# -----------------------------
# Compile model
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# Train
# -----------------------------

print("Training BiLSTM model...")

model.fit(
    padded_sequences,
    labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)


# -----------------------------
# Save model and tokenizer
# -----------------------------

artifacts_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "artifacts"
)

os.makedirs(artifacts_dir, exist_ok=True)

model.save(
    os.path.join(
        artifacts_dir,
        "bilstm_model.h5"
    )
)

with open(
    os.path.join(artifacts_dir, "tokenizer.pkl"),
    "wb"
) as f:
    pickle.dump(tokenizer, f)


print("BiLSTM model and tokenizer saved successfully.")
