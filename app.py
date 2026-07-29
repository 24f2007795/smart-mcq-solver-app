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
# Load Model
# -----------------------------------------------------

MODEL_NAME = "24f2007795/mcq-solver"
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Your label order
LABELS = ["A", "B", "C", "D", "E"]

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
# Streamlit UI
# -----------------------------------------------------
st.title("🧠 Smart MCQ Solver")

st.write(
    "Fine-tuned **RoBERTa** model for Multiple Choice Question Answering."
)

prompt = st.text_area("Question")

a = st.text_input("Option A")
b = st.text_input("Option B")
c = st.text_input("Option C")
d = st.text_input("Option D")
e = st.text_input("Option E")

if st.button("Predict"):

    if not all([prompt, a, b, c, d, e]):
        st.warning("Please fill all fields.")
    else:

        predictions = predict(prompt, a, b, c, d, e)

        st.success("Prediction Complete!")

        st.subheader("Top 3 Predictions")

        medals = ["🥇", "🥈", "🥉"]

        for medal, (label, score) in zip(medals, predictions):
            st.write(f"{medal} **{label}** — {score:.2f}% confidence")
