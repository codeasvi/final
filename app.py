import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide"
)

# ---------------- STYLING ----------------
st.markdown("""
<style>
.main { background-color: #f6f7fb; }
h1 { text-align:center; color:#1f2937; }
.subtitle { text-align:center; color:#6b7280; margin-bottom:30px; }
.card {
    background:white;
    padding:20px;
    border-radius:14px;
    box-shadow:0 4px 14px rgba(0,0,0,0.08);
    margin-bottom:20px;
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
st.markdown("<h1>📊 Twitter Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Offline Transformer-based High-Fidelity Dashboard</div>", unsafe_allow_html=True)

# ---------------- CSV UPLOAD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload Twitter CSV (must contain 'tweet' and 'sentiment' columns)",
    type=["csv"]
)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is None:
    st.info("⬆️ Upload a CSV file to continue")
    st.stop()

df = pd.read_csv(uploaded_file)

# ---------------- DATA PREVIEW ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🔍 Dataset Preview")
st.write(f"Total Tweets: **{len(df)}**")
st.dataframe(df.head(), use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ANALYSIS ----------------
if st.button("🚀 Analyze Sentiment", use_container_width=True):

    with st.spinner("Analyzing sentiment..."):
        results = df["tweet"].apply(lambda x: model(str(x))[0])
        df["Predicted_Sentiment"] = results.apply(lambda x: x["label"])
        df["Confidence"] = results.apply(lambda x: x["score"])

    # Confidence-weighted score
    df["Sentiment_Score"] = np.where(
        df["Predicted_Sentiment"] == "POSITIVE",
        df["Confidence"],
        -df["Confidence"]
    )

    # ---------------- METRICS ----------------
    counts = df["Predicted_Sentiment"].value_counts()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("😊 Positive", counts.get("POSITIVE", 0))
    col2.metric("😠 Negative", counts.get("NEGATIVE", 0))
    col3.metric("📦 Total", len(df))
    col4.metric("📊 Avg Sentiment Score", round(df["Sentiment_Score"].mean(), 2))

    # ---------------- ACCURACY ----------------
    if "sentiment" in df.columns:
        accuracy = (df["sentiment"] == df["Predicted_Sentiment"]).mean()
        st.success(f"🎯 Model Accuracy: {accuracy*100:.2f}%")

    # ---------------- FILTER ----------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    filter_option = st.selectbox(
        "Filter Tweets by Sentiment",
        ["All", "POSITIVE", "NEGATIVE"]
    )
    filtered_df = df if filter_option == "All" else df[df["Predicted_Sentiment"] == filter_option]
    st.dataframe(filtered_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- CHARTS ----------------
    colA, colB = st.columns(2)

    with colA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Sentiment Distribution (Bar)")
        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🥧 Sentiment Distribution (Pie)")
        fig2, ax2 = plt.subplots()
        ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax2.axis("equal")
        st.pyplot(fig2)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- TREND ----------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Sentiment Trend")
    df["Trend_Index"] = range(len(df))
    df["Rolling_Sentiment"] = df["Sentiment_Score"].rolling(20).mean()
    st.line_chart(df.set_index("Trend_Index")["Rolling_Sentiment"])
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- INSIGHTS ----------------
    st.info(f"""
📌 **Auto-Generated Insights**
• Positive tweets: {counts.get('POSITIVE',0)}
• Negative tweets: {counts.get('NEGATIVE',0)}
• Overall public opinion is **{"Positive" if counts.get("POSITIVE",0) > counts.get("NEGATIVE",0) else "Negative"}**
""")

    # ---------------- DOWNLOAD ----------------
    st.download_button(
        "⬇️ Download Final Results CSV",
        df.to_csv(index=False).encode("utf-8"),
        "sentiment_results.csv",
        "text/csv",
        use_container_width=True
    )
