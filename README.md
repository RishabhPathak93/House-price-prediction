# 🎬 Sentiment Analysis with GRU (IMDB Reviews)

This project is a sentiment analysis tool trained on IMDB movie reviews using a GRU (Gated Recurrent Unit) neural network. It includes a web interface built with **Streamlit** where users can input custom reviews and receive real-time predictions of whether the sentiment is **positive** or **negative**.

---

## 🧠 Model Overview

- **Model Type**: GRU (Gated Recurrent Unit)
- **Dataset**: IMDB (from Keras)
- **Input**: Text reviews (converted to padded sequences)
- **Output**: Binary sentiment (Positive or Negative)

---

## 🚀 Features

- Predicts sentiment from user-written movie reviews.
- GRU-based neural network for handling sequences of text.
- Real-time predictions with confidence score.
- Clean and responsive UI with Streamlit.
- Local model — runs completely offline once trained.

---

## 📁 File Structure

- `train_model.py` – Trains and saves the GRU model + tokenizer.
- `app.py` – Streamlit web app for real-time sentiment prediction.
- `requirements.txt` – Python dependencies.
- `gru_sentiment_model.h5` – Saved Keras model file (after training).
- `preprocessor.joblib` – Serialized dictionary with word index and preprocessing settings.

---

## 📦 Installation

1. **Install dependencies**

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
