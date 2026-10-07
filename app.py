import streamlit as st
import pickle
import json
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load trained model
model = load_model("model_bi_lstm_model_small.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load maximum length
with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# Load label map
with open("label_map.json", "r") as f:
    label_map = json.load(f)


# Prediction function
def predict_sentiment(text):

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=max_len
    )

    prediction = model.predict(
        padded,
        verbose=0
    )

    predicted_class = np.argmax(
        prediction,
        axis=1
    )[0]

    predicted_label = label_map[str(predicted_class)]

    confidence = np.max(prediction) * 100

    return predicted_label, confidence


# Streamlit interface
st.title("Financial Sentiment Analysis")

st.write(
    "FastText + BiLSTM based Sentiment Prediction"
)

text = st.text_area(
    "Enter a financial sentence:"
)


if st.button("Predict Sentiment"):

    if text.strip() == "":
        st.warning("Please enter a sentence.")

    else:

        label, confidence = predict_sentiment(text)

        st.success(
            f"Prediction: {label}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )