import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Toxic Comment Benchmarking", layout="wide")

st.title("Toxic Comment Benchmarking Dashboard")

menu = ["Benchmark Runner", "Historical Runs", "Model Rankings", "Error Analysis", "Fairness Dashboard", "Drift Dashboard", "Ensemble Dashboard", "Download Reports"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Benchmark Runner":
    st.header("Run New Benchmark")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if st.button("Start Benchmark"):
        try:
            res = requests.post(f"{API_URL}/benchmark/run", json={"dataset_name": "uploaded", "model_configs": []})
            if res.status_code == 200:
                st.success(f"Benchmark started! {res.json()}")
            else:
                st.error("Failed to start.")
        except Exception as e:
            st.error(f"API Error: Make sure FastAPI is running. Details: {e}")

elif choice == "Historical Runs":
    st.header("Benchmark History")
    st.write("Connects to /benchmark/history API.")

elif choice == "Model Rankings":
    st.header("Model Rankings")
    st.write("Connects to /benchmark/rankings API.")

elif choice == "Error Analysis":
    st.header("Error Analysis & LIME")
    st.write("Interactive error analysis visualizations.")

elif choice == "Fairness Dashboard":
    st.header("Fairness & Bias Analysis")
    st.write("Disparate impact ratios and demographic analyses.")

elif choice == "Drift Dashboard":
    st.header("Data Drift Analysis")
    st.write("Vocabulary, confidence, and toxicity drifts over time.")

elif choice == "Ensemble Dashboard":
    st.header("Ensemble Methods")
    st.write("Compare weighted vs majority voting performance.")

elif choice == "Download Reports":
    st.header("Download Exported Reports")
    st.write("PDF, DOCX, and HTML downloads from /benchmark/report API.")
