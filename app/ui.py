import streamlit as st
import requests
import plotly.express as px
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# SESSION STATE
# -----------------------------

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("⚙️ Configuration")

mode = st.sidebar.radio(
    "Comparison Mode",
    ["Single Model", "Version vs. Version", "All Models", "Custom Selection"]
)

AVAILABLE_MODELS = ["v1_svm", "v1_lr", "v1_rf", "v2_svm", "v2_lr", "v2_bert"]

selected_models = []
if mode == "Single Model":
    selected_models = [st.sidebar.selectbox("Select Model", AVAILABLE_MODELS)]
elif mode == "Version vs. Version":
    v1_mod = st.sidebar.selectbox("Version 1 Model", ["v1_svm", "v1_lr", "v1_rf"])
    v2_mod = st.sidebar.selectbox("Version 2 Model", ["v2_svm", "v2_lr", "v2_bert"])
    selected_models = [v1_mod, v2_mod]
elif mode == "All Models":
    selected_models = AVAILABLE_MODELS
elif mode == "Custom Selection":
    selected_models = st.sidebar.multiselect("Select Models", AVAILABLE_MODELS, default=["v1_svm", "v2_svm"])

st.sidebar.markdown("---")
st.sidebar.subheader("Preprocessing & XAI")
normalize_text = st.sidebar.toggle("Normalize Text (Adversarial Detection)", value=False)
enable_xai = st.sidebar.toggle("Enable XAI (LIME)", value=True)

# -----------------------------
# HEADER
# -----------------------------

st.title("🧠 Toxic Comment Detection Dashboard")
st.markdown("A hyper-modern comparison dashboard for ML models.")

# -----------------------------
# INPUT
# -----------------------------

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
                    "http://127.0.0.1:8000/predict",
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
                st.error("❌ Error connecting to API. Make sure FastAPI is running.")

# -----------------------------
# RESULTS DISPLAY
# -----------------------------

if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    st.markdown("---")
    st.subheader("📊 Comparison Results")
    
    if not results:
         st.warning("No results returned. The selected models might have failed to load on the backend.")
    else:
         # Create a responsive grid layout
         cols = st.columns(len(results))
         
         for idx, (m_id, res) in enumerate(results.items()):
             with cols[idx]:
                 # Container for a card-like look
                 with st.container(border=True):
                     st.markdown(f"### {m_id.upper()}")
                     
                     prediction = res["prediction"]
                     confidence = res["confidence"]
                     latency = res["latency"]
                     
                     # Highlight with success/error colors
                     if prediction == "Toxic":
                         st.error(f"🚨 **{prediction}** ({confidence:.1%})")
                     else:
                         st.success(f"✅ **{prediction}** ({confidence:.1%})")
                     
                     st.caption(f"⏱ Latency: {latency}")
                     
                     with st.expander("Feature Explanation"):
                         st.write(res["feature_explanation"])
                         
                     # LIME Visualization with Plotly
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