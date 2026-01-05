import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS (HI-FI UI) ----------------
st.markdown("""
<style>
body {
    background-color: #f5f7fb;
}
.main {
    background-color: #f5f7fb;
}
.card {
    background-color: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}
.title {
    text-align: center;
    font-size: 44px;
    font-weight: 700;
    color: #111827;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 35px;
}
.metric {
    text-align: center;
    font-size: 34px;
    font-weight: 700;
}
.metric-label {
    text-align: center;
    color: #6b7280;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

model = load_model()

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🐦 Twitter Sentiment Analysis</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Offline Transformer-based Sentiment Dashboard</div>", unsafe_allow_html=True)

# ---------------- CSV UPLOAD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📁 Upload Twitter CSV File")
uploaded_file = st.file_uploader(
    "Upload a CSV file (must contain a column named 'tweet')",
    type=["csv"]
)
st.markdown("</div>", unsafe_allow_html=True)

# STOP if CSV not uploaded
if uploaded_file is None:
    st.info("⬆️ Upload a CSV file to continue")
    st.stop()

# ---------------- LOAD DATA ----------------
df = pd.read_csv(uploaded_file)

# ---------------- DATA PREVIEW ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🔍 Dataset Preview")
st.write(f"Total Tweets: **{len(df)}**")
st.dataframe(df.head(), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ANALYZE ----------------
if st.button("🚀 Analyze Sentiment", use_container_width=True):
    with st.spinner("Analyzing sentiment..."):
        df["Predicted_Sentiment"] = df["tweet"].apply(
            lambda x: model(str(x))[0]["label"]
        )

    # ---------------- METRICS ----------------
    counts = df["Predicted_Sentiment"].value_counts()

    st.markdown("### 📊 Sentiment Overview")
    col1, col2, col3 = st.columns(3)

    col1.markdown(
        f"<div class='card'><div class='metric'>😊 {counts.get('POSITIVE',0)}</div><div class='metric-label'>Positive</div></div>",
        unsafe_allow_html=True
    )
    col2.markdown(
        f"<div class='card'><div class='metric'>😠 {counts.get('NEGATIVE',0)}</div><div class='metric-label'>Negative</div></div>",
        unsafe_allow_html=True
    )
    col3.markdown(
        f"<div class='card'><div class='metric'>{len(df)}</div><div class='metric-label'>Total</div></div>",
        unsafe_allow_html=True
    )

    # ---------------- CHART ----------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Sentiment Distribution")

    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Tweets")
    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- TABLE ----------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📄 Analyzed Dataset")
    st.dataframe(df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- DOWNLOAD ----------------
    st.download_button(
        "⬇️ Download Results CSV",
        df.to_csv(index=False).encode("utf-8"),
        "sentiment_results.csv",
        "text/csv",
        use_container_width=True
    )
