import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------

st.set_page_config(
    page_title="Smart MCQ Solver",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------------------------------
# Model Configuration
# -----------------------------------------------------

MODEL_NAME = "24f2007795/mcq-solver"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ["A", "B", "C", "D", "E"]

# -----------------------------------------------------
# Load Model
# -----------------------------------------------------

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


try:
    tokenizer, model = load_model()
except Exception as e:
    st.error("❌ Unable to load the model from Hugging Face.")
    st.exception(e)
    st.stop()

# -----------------------------------------------------
# Prediction Function
# -----------------------------------------------------

def predict(prompt, a, b, c, d, e):

    text = (
        "Question: " + prompt +
        "\n\nOption A: " + a +
        "\nOption B: " + b +
        "\nOption C: " + c +
        "\nOption D: " + d +
        "\nOption E: " + e
    )

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=384,
        return_tensors="pt"
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=1)[0]

    top3 = torch.topk(probabilities, k=3)

    predictions = []

    for idx, score in zip(top3.indices, top3.values):
        predictions.append(
            (
                LABELS[idx.item()],
                score.item() * 100
            )
        )

    return predictions

# -----------------------------------------------------
# User Interface
# -----------------------------------------------------

st.title("🧠 Smart MCQ Solver")

st.markdown(
    """
This application uses a **fine-tuned RoBERTa model**
to predict the most likely correct answer for a
multiple-choice question.
"""
)

st.divider()

prompt = st.text_area("Question")

a = st.text_input("Option A")
b = st.text_input("Option B")
c = st.text_input("Option C")
d = st.text_input("Option D")
e = st.text_input("Option E")

if st.button("Predict", use_container_width=True):

    if not all([prompt, a, b, c, d, e]):
        st.warning("Please fill in all fields.")
    else:

        with st.spinner("Running inference..."):
            predictions = predict(prompt, a, b, c, d, e)

        st.success("Prediction Complete!")

        st.subheader("Top 3 Predictions")

        medals = ["🥇", "🥈", "🥉"]

        for medal, (label, score) in zip(medals, predictions):

            st.write(f"{medal} **Option {label}**")

            st.progress(score / 100)

            st.write(f"Confidence: **{score:.2f}%**")

            st.write("")

st.divider()

st.caption(
    "Developed using a fine-tuned RoBERTa model trained for Multiple Choice Question Answering."
)
