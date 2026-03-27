import streamlit as st
import requests

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# HEADER
# -----------------------------

st.title("🧠 Toxic Comment Detection (XAI)")
st.markdown("Analyze comments using ML + Explainable AI")

# -----------------------------
# INPUT
# -----------------------------

user_input = st.text_area("✍️ Enter your comment:", height=150)

# -----------------------------
# BUTTON
# -----------------------------

if st.button("🔍 Analyze Comment"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        with st.spinner("Analyzing..."):

            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={"text": user_input}
                )

                result = response.json()

                # -----------------------------
                # PREDICTION DISPLAY
                # -----------------------------

                st.subheader("📊 Prediction Result")

                if result["prediction"] == "Toxic":
                    st.error("🚨 Toxic Comment Detected")
                else:
                    st.success("✅ Clean Comment")

                # -----------------------------
                # FEATURE EXPLANATION
                # -----------------------------

                st.subheader("🧾 Why was this predicted?")

                st.info(result["feature_explanation"])

                # -----------------------------
                # LIME EXPLANATION
                # -----------------------------

                if result["lime_explanation"]:
                    st.subheader("🔍 Important Words (Model Insight)")

                    for word, score in result["lime_explanation"]:
                        st.write(f"• **{word}** → {round(score, 3)}")

                # -----------------------------
                # RAW TEXT
                # -----------------------------

                st.subheader("📌 Original Input")
                st.write(user_input)

            except Exception as e:
                st.error("❌ Error connecting to API. Make sure FastAPI is running.")