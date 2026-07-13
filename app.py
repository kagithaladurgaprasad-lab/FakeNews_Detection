import re
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- Config (must match training) ----------------
MAX_LEN = 300
MODEL_PATH = "fake_news_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


try:
    model, tokenizer = load_artifacts()
    load_error = None
except Exception as e:
    model, tokenizer = None, None
    load_error = str(e)


# ---- Same cleaning function used during training ----
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"<.*?>", " ", text)              # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)            # keep only letters
    text = re.sub(r"\s+", " ", text).strip()         # collapse whitespace
    return text


def predict_news(title, body):
    combined = f"{title} {body}".strip()
    cleaned = clean_text(combined)

    if cleaned == "":
        return None, None, None

    seq = tokenizer.texts_to_sequences([cleaned])
    unknown_ratio = seq[0].count(1) / max(len(seq[0]), 1)  # <OOV> is usually index 1
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = float(model.predict(padded, verbose=0)[0][0])
    label = "Real" if prob > 0.5 else "Fake"
    return label, prob, unknown_ratio


# ---------------- Streamlit UI ----------------
st.title("📰 Fake News Detector")
st.write("GRU neural network trained on the ISOT Fake/Real news dataset (~98% test accuracy)")

if load_error:
    st.error(
        f"Couldn't load model artifacts. Make sure `{MODEL_PATH}` and `{TOKENIZER_PATH}` "
        f"are in the same folder as this app.\n\nDetails: {load_error}"
    )
    st.stop()

title = st.text_input("Article title:", placeholder="Enter the headline...")
body = st.text_area("Article text:", height=220, placeholder="Paste the article body here...")

col1, col2 = st.columns([1, 1])
with col1:
    check_clicked = st.button("Check Article", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

if check_clicked:
    if title.strip() == "" and body.strip() == "":
        st.warning("Please enter a title or article text first.")
    else:
        label, prob, unknown_ratio = predict_news(title, body)

        if label is None:
            st.warning("Couldn't extract any usable text from that input — try adding more content.")
        else:
            confidence = prob if label == "Real" else 1 - prob

            if label == "Real":
                st.success(f"**{label} News** — confidence: {confidence:.1%}")
            else:
                st.error(f"**{label} News** — confidence: {confidence:.1%}")

            st.progress(prob)
            st.caption(f"Raw model output (probability of being real): {prob:.4f}")

            word_count = len(clean_text(f'{title} {body}').split())
            if word_count < 15:
                st.info("⚠️ Short input — predictions on very short text tend to be less reliable.")
            if unknown_ratio and unknown_ratio > 0.3:
                st.info("⚠️ A lot of these words weren't in the training vocabulary "
                         "(unusual topic/language) — treat this prediction with extra caution.")

with st.expander("About this model"):
    st.write(
        "- Trained on the ISOT Fake/Real news dataset (articles from 2016–2017).\n"
        "- The model learns from writing style and word patterns in that dataset, "
        "not a live fact-checking database — it can't verify claims against reality.\n"
        "- It will likely be less reliable on topics, sources, or writing styles that "
        "differ a lot from its training data (e.g. very recent events, satire, non-English text).\n"
        "- Use this as one signal among several, not a final verdict."
    )
