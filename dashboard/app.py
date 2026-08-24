import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# =====================================================
# CONFIG
# =====================================================

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Toxic Comment ML Platform",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# STATE
# =====================================================
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# =====================================================
# TITLE
# =====================================================

st.title("🧠 Toxic Comment ML Platform")
st.markdown("Evaluate models via batch benchmarking or single-comment explainable AI (XAI).")

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

menu = [
    "Single Comment (XAI)",
    "Batch Benchmark Runner",
    "Fairness & Drift (Info)"
]

choice = st.sidebar.selectbox("Navigation", menu)

# =====================================================
# 1. SINGLE COMMENT (XAI)
# =====================================================
if choice == "Single Comment (XAI)":
    st.header("🔍 Real-time Prediction & Explainability")
    
    # Fetch models
    available_models = []
    model_statuses = {}
    try:
        resp = requests.get(f"{API_URL}/models")
        if resp.status_code == 200:
            data = resp.json()
            available_models = data.get("models", [])
            model_statuses = data.get("statuses", {})
    except Exception:
        st.sidebar.error("Failed to connect to API to fetch models.")
        
    if not available_models:
        st.warning("No models loaded or API is offline.")
    else:
        st.sidebar.markdown("---")
        st.sidebar.subheader("XAI Configuration")
        selected_models = st.sidebar.multiselect("Available Models", available_models, default=available_models[:1])
        
        unavailable_models = [m for m, is_avail in model_statuses.items() if not is_avail]
        if unavailable_models:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Unavailable Models")
            for um in unavailable_models:
                st.sidebar.markdown(f"🚫 **{um}**")
                st.sidebar.caption("⚠ Missing local model artifact")
            st.sidebar.markdown("---")
            
        normalize_text = st.sidebar.toggle("Normalize Text (Adversarial Detection)", value=False)
        enable_xai = st.sidebar.toggle("Enable XAI (LIME)", value=True)
        
        user_input = st.text_area("✍️ Enter your comment:", height=100)
        
        if st.button("🔍 Analyze Comment"):
            if user_input.strip() == "":
                st.warning("⚠️ Please enter some text")
            elif not selected_models:
                st.warning("⚠️ Please select at least one model.")
            else:
                with st.spinner("Analyzing across selected models..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/predict",
                            json={
                                "text": user_input,
                                "model_ids": selected_models,
                                "normalize": normalize_text,
                                "enable_lime": enable_xai
                            }
                        )
                        if response.status_code == 200:
                            st.session_state.analysis_results = response.json()
                        else:
                            st.error(f"API Error ({response.status_code}): {response.text}")
                    except Exception as e:
                        st.error(f"❌ Error connecting to API: {e}")

        # Display Results
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            st.markdown("---")
            st.subheader("📊 Comparison Results")

            cols = st.columns(len(results))
            for idx, (m_id, res) in enumerate(results.items()):
                with cols[idx]:
                    with st.container():
                        st.markdown(f"### {m_id.upper()}")
                        
                        if "error" in res:
                            st.error(res["error"])
                            continue

                        prediction = res["prediction"]
                        confidence = res["confidence"]
                        latency = res["latency"]

                        if prediction == "Toxic":
                            st.error(f"🚨 **{prediction}** ({confidence:.1%})")
                        else:
                            st.success(f"✅ **{prediction}** ({confidence:.1%})")

                        st.caption(f"⏱ Latency: {latency}")

                        with st.expander("Feature Explanation"):
                            st.write(res["feature_explanation"])

                        if enable_xai and res.get("lime_explanation"):
                            lime_data = res["lime_explanation"]
                            df = pd.DataFrame(lime_data, columns=["Feature", "Weight"])
                            df["Color"] = df["Weight"].apply(lambda x: "red" if x > 0 else "green")
                            df = df.sort_values(by="Weight", ascending=True)

                            fig = px.bar(
                                df, x="Weight", y="Feature", orientation="h",
                                color="Color", color_discrete_map={"red": "#ff4b4b", "green": "#00cc96"},
                                title="LIME Word Importance"
                            )
                            fig.update_layout(
                                showlegend=False,
                                margin=dict(l=0, r=0, t=30, b=0),
                                height=250,
                                xaxis_title="Weight",
                                yaxis_title=None
                            )
                            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 2. BENCHMARK RUNNER
# =====================================================
elif choice == "Batch Benchmark Runner":
    st.header("🚀 Run Batch Benchmark")
    st.write("Upload a CSV dataset with a comment column and an optional ground truth column to run bulk inference and evaluation against all configured models.")
    
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if uploaded_file is not None:
        st.success(f"Loaded file: {uploaded_file.name}")
        
        if st.button("Start Benchmark"):
            try:
                with st.spinner("Running benchmark (this may take a while)..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    response = requests.post(f"{API_URL}/benchmark/run", files=files)

                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Benchmark completed successfully!")
                        st.json(result)
                        st.info("Check the SQLite database (`benchmark_results.db`) for detailed results.")
                    else:
                        st.error(f"❌ API Error {response.status_code}")
                        try:
                            st.json(response.json())
                        except:
                            st.text(response.text)
            except Exception as e:
                st.error(f"❌ Connection Error: {str(e)}")

# =====================================================
# 3. FAIRNESS & DRIFT (INFO)
# =====================================================
elif choice == "Fairness & Drift (Info)":
    st.header("⚖️ Fairness & Drift Dashboards")
    st.info(
        "**Status: Not Available**\n\n"
        "Fairness and Drift analytics require specific metadata that is currently not present in the standard dataset.\n"
        "- **Fairness**: Requires demographic or protected subgroup markers.\n"
        "- **Drift**: Requires historical distributions and temporal tracking markers.\n\n"
        "Fabricated metrics for these features have been disabled to maintain data integrity."
    )