import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import sqlite3
import time
from pathlib import Path

# =====================================================
# CONFIG & PATHS
# =====================================================

API_URL = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Toxic Comment ML Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI polish
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #333;
    }
    .metric-title { font-size: 14px; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 32px; font-weight: bold; color: #fff; margin: 10px 0; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# =====================================================
# STATE & HELPERS
# =====================================================
@st.cache_data(ttl=60)
def get_production_model():
    try:
        resp = requests.get(f"{API_URL}/production-model", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

@st.cache_data(ttl=60)
def get_available_models():
    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("models", [])
    except:
        pass
    return []


prod_model = get_production_model()
is_api_online = prod_model is not None

# =====================================================
# NAVIGATION
# =====================================================

menu = [
    "🏠 Overview",
    "🔎 Toxicity Analyzer",
    "📊 Model Benchmark",
    "🧪 Adversarial Testing",
    "📁 Benchmark History",
    "ℹ️ About & Methodology"
]

st.sidebar.title("🧠 Toxic ML Platform")
st.sidebar.markdown("---")
choice = st.sidebar.radio("Navigation", menu)
st.sidebar.markdown("---")
if is_api_online:
    st.sidebar.success("🟢 API Status: Online")
else:
    st.sidebar.error("🔴 API Status: Offline")

# =====================================================
# 1. OVERVIEW
# =====================================================
if choice == "🏠 Overview":
    st.title("Welcome to the Toxic Comment ML Platform")
    st.markdown("A production-ready platform for detecting toxic comments, benchmarking models, and exploring explainable AI (XAI).")
    
    if not is_api_online:
        st.error("Cannot connect to the backend API. Please ensure it is running.")
    else:
        st.subheader("Production Model Card")
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Model</div><div class="metric-value">{prod_model.get("display_name", "N/A")}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Version</div><div class="metric-value">v{prod_model.get("version", "N/A")}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Selection Metric</div><div class="metric-value">{prod_model.get("selection_metric", "N/A").upper()}</div></div>', unsafe_allow_html=True)
        with c4:
            score = prod_model.get("selection_score", 0)
            st.markdown(f'<div class="metric-card"><div class="metric-title">Score</div><div class="metric-value">{score:.2%}</div></div>', unsafe_allow_html=True)
        
        st.markdown("### Platform Capabilities")
        cols = st.columns(3)
        with cols[0]:
            st.info("✅ **LIME Explainability**\n\nUnderstand *why* a model made its prediction through word importance visualization.")
        with cols[1]:
            st.info("✅ **Batch Benchmarking**\n\nUpload datasets to evaluate model performance (Accuracy, Precision, Recall, F1) across classical and deep learning models.")
        with cols[2]:
            st.info("✅ **Adversarial Defense**\n\nTest robustness against leetspeak and obfuscated text using text normalization.")
            
        st.markdown("---")
        st.info("Please select '🔎 Toxicity Analyzer' from the sidebar to continue.")

# =====================================================
# 2. TOXICITY ANALYZER
# =====================================================
elif choice == "🔎 Toxicity Analyzer":
    st.header("🔎 Toxicity Analyzer")
    st.markdown("Analyze single comments using the active production model.")
    
    if not is_api_online:
        st.error("API is offline.")
    elif "v2_bert" not in get_available_models():
        st.error("Production model artifact (v2_bert) is currently unavailable.")
    else:
        st.info(f"**Active Model:** {prod_model['display_name']} (v{prod_model['version']}) | **Optimized for:** {prod_model['selection_metric'].upper()} ({prod_model['selection_score']:.2%})")
        
        user_input = st.text_area("✍️ Enter comment to analyze:", height=150)
        
        c1, c2 = st.columns(2)
        with c1:
            enable_lime = st.toggle("Enable LIME Explainability", value=True)
        with c2:
            normalize_text = st.toggle("Enable Adversarial Normalization", value=False)
            
        if st.button("Analyze", type="primary", use_container_width=True):
            if not user_input.strip():
                st.warning("Please enter some text.")
            else:
                with st.spinner("Analyzing..."):
                    try:
                        resp = requests.post(f"{API_URL}/predict", json={
                            "text": user_input,
                            "model_ids": ["v2_bert"],
                            "normalize": normalize_text,
                            "enable_lime": enable_lime
                        }).json()
                        
                        res = resp.get("v2_bert", {})
                        if "error" in res:
                            st.error(res["error"])
                        else:
                            st.markdown("---")
                            rc1, rc2 = st.columns([1, 2])
                            
                            with rc1:
                                st.subheader("Result")
                                pred = res["prediction"]
                                conf = res["confidence"]
                                if pred == "Toxic":
                                    st.error(f"🚨 **{pred}**")
                                else:
                                    st.success(f"✅ **{pred}**")
                                    
                                st.metric("Confidence", f"{conf:.2%}")
                                st.metric("Inference Latency", res["latency"])
                                
                                st.markdown("**Explanation:**")
                                st.write(res["feature_explanation"])
                                
                            with rc2:
                                if enable_lime and res.get("lime_explanation"):
                                    st.subheader("LIME Word Importance")
                                    df = pd.DataFrame(res["lime_explanation"], columns=["Feature", "Weight"])
                                    df["Color"] = df["Weight"].apply(lambda x: "red" if x > 0 else "green")
                                    df = df.sort_values(by="Weight", ascending=True)

                                    fig = px.bar(
                                        df, x="Weight", y="Feature", orientation="h",
                                        color="Color", color_discrete_map={"red": "#ff4b4b", "green": "#00cc96"},
                                    )
                                    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=300)
                                    st.plotly_chart(fig, use_container_width=True)
                                elif enable_lime:
                                    st.warning("LIME explanation unavailable (often occurs if confidence is extremely high or text is too short).")
                                
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")

# =====================================================
# 3. MODEL BENCHMARK
# =====================================================
elif choice == "📊 Model Benchmark":
    st.header("📊 Model Benchmark")
    
    tab1, tab2 = st.tabs(["Final Evaluation (Authoritative)", "Run New Batch Benchmark"])
    
    with tab1:
        st.markdown("### Production Model Selection")
        st.info("The production model (`v2_bert`) was selected based on the highest **Toxic F1** score achieved during the final evaluation phase.")
        
        csv_path = PROJECT_ROOT / "version_2" / "outputs" / "final_model_comparison.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            
            # Format dataframe for display
            display_df = df.copy()
            for col in ['accuracy', 'precision_toxic', 'recall_toxic', 'f1_toxic']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
                
            # If latency exists, format it, otherwise drop it if it's all NaNs or not present
            if 'inference_time_seconds' in display_df.columns:
                display_df['inference_time_seconds'] = display_df['inference_time_seconds'].apply(lambda x: f"{x:.4f}s" if pd.notna(x) else "N/A")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Charts
            c1, c2 = st.columns(2)
            with c1:
                fig1 = px.bar(df, x="model", y="f1_toxic", title="Toxic F1 by Model", color="model")
                st.plotly_chart(fig1, use_container_width=True)
            with c2:
                fig2 = px.scatter(df, x="recall_toxic", y="precision_toxic", color="model", title="Toxic Precision vs Recall", size_max=15)
                fig2.update_traces(marker=dict(size=12))
                st.plotly_chart(fig2, use_container_width=True)
                
            if 'inference_time_seconds' in df.columns:
                fig3 = px.bar(df, x="model", y="inference_time_seconds", title="Inference Latency (Seconds)", color="model")
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.error("Final evaluation CSV not found. Ensure `version_2/outputs/final_model_comparison.csv` exists.")

    with tab2:
        st.markdown("### Run Batch Benchmark")
        st.write("Upload a dataset to evaluate model performance across all active models.")
        
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        if uploaded_file:
            df_preview = pd.read_csv(uploaded_file)
            st.write(f"**Preview ({len(df_preview)} samples):**")
            st.dataframe(df_preview.head(3), use_container_width=True)
            
            possible_columns = ["comment", "comments", "text", "sentence", "content", "clean_text"]
            target_columns = ["target", "toxic", "is_toxic", "label", "ground_truth"]
            
            comment_col = next((c for c in possible_columns if c in df_preview.columns), None)
            target_col = next((c for c in target_columns if c in df_preview.columns), None)
            
            if not comment_col:
                st.error(f"Missing comment column. Expected one of: {possible_columns}")
            else:
                st.success(f"Detected comment column: `{comment_col}`")
                
                if not target_col:
                    st.warning("⚠️ No ground-truth column detected. Only predictions will be generated, and metrics will not be calculated.")
                else:
                    st.success(f"Detected ground truth column: `{target_col}`")
                    
                if st.button("Start Benchmark"):
                    if not is_api_online:
                        st.error("API is offline.")
                    else:
                        with st.spinner("Running benchmark..."):
                            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                            resp = requests.post(f"{API_URL}/benchmark/run", files=files)
                            
                            if resp.status_code == 200:
                                st.success("Benchmark Completed!")
                                res = resp.json()
                                with st.expander("Technical Response"):
                                    st.json(res)
                            else:
                                st.error(f"API Error {resp.status_code}")
                                st.write(resp.text)

# =====================================================
# 4. ADVERSARIAL TESTING
# =====================================================
elif choice == "🧪 Adversarial Testing":
    st.header("🧪 Adversarial Testing")
    st.markdown("Test the robustness of the model against text obfuscation and leetspeak.")
    st.warning("🚧 **Experimental Feature:** This demonstrates how normalization impacts predictions on obfuscated text.")
    
    examples = [
        "you are an idiot",
        "y0u @re an id10t",
        "h@te y0u s0 much!!!",
        "norm@l text with symbo1s"
    ]
    
    selected_example = st.selectbox("Select an example or type your own:", ["(Custom)"] + examples)
    
    if selected_example == "(Custom)":
        user_input = st.text_area("✍️ Enter text with obfuscation (e.g. y0u @re 5tupid):")
    else:
        user_input = st.text_area("✍️ Enter text with obfuscation:", value=selected_example)
        
    if st.button("Test Robustness", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        elif not is_api_online:
            st.error("API is offline.")
        else:
            with st.spinner("Testing..."):
                try:
                    # Run without normalization
                    resp_raw = requests.post(f"{API_URL}/predict", json={"text": user_input, "model_ids": ["v2_bert"], "normalize": False, "enable_lime": False}).json()
                    # Run with normalization
                    resp_norm = requests.post(f"{API_URL}/predict", json={"text": user_input, "model_ids": ["v2_bert"], "normalize": True, "enable_lime": False}).json()
                    
                    res_raw = resp_raw.get("v2_bert", {})
                    res_norm = resp_norm.get("v2_bert", {})
                    
                    if "error" in res_raw:
                        st.error(res_raw["error"])
                    else:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.subheader("Raw Prediction")
                            st.code(user_input, language="text")
                            
                            pred = res_raw["prediction"]
                            if pred == "Toxic":
                                st.error(f"🚨 **{pred}** ({res_raw['confidence']:.2%})")
                            else:
                                st.success(f"✅ **{pred}** ({res_raw['confidence']:.2%})")
                                
                        with c2:
                            st.subheader("Normalized Prediction")
                            # We don't get the normalized text back from the API currently, but we can assume it was applied
                            st.code("Normalization Applied", language="text")
                            
                            pred_norm = res_norm["prediction"]
                            if pred_norm == "Toxic":
                                st.error(f"🚨 **{pred_norm}** ({res_norm['confidence']:.2%})")
                            else:
                                st.success(f"✅ **{pred_norm}** ({res_norm['confidence']:.2%})")
                                
                        if res_raw["prediction"] != res_norm["prediction"]:
                            st.info("💡 **Observation:** Normalization changed the prediction outcome!")
                        else:
                            st.info("💡 **Observation:** Prediction outcome remained the same.")
                            
                except Exception as e:
                    st.error(f"Error: {e}")

# =====================================================
# 5. BENCHMARK HISTORY
# =====================================================
elif choice == "📁 Benchmark History":
    st.header("📁 Benchmark History")
    
    db_path = PROJECT_ROOT / "database" / "toxic_comments_benchmark.db"
    
    if not db_path.exists():
        st.info("No benchmark history found. Run a batch benchmark to populate history.")
    else:
        try:
            conn = sqlite3.connect(db_path)
            # Query runs
            runs_df = pd.read_sql_query("SELECT id, run_name, start_time, end_time, status FROM benchmark_runs ORDER BY start_time DESC", conn)
            
            if runs_df.empty:
                st.info("No benchmark runs recorded yet.")
            else:
                st.dataframe(runs_df, use_container_width=True)
                
                st.markdown("### View Run Details")
                selected_run_id = st.selectbox("Select Run ID to inspect:", runs_df['id'].tolist())
                
                if selected_run_id:
                    preds_df = pd.read_sql_query(f"SELECT model_name, text_input, ground_truth, prediction, confidence, latency_ms FROM model_predictions WHERE run_id = {selected_run_id}", conn)
                    if not preds_df.empty:
                        st.write(f"**Predictions for Run {selected_run_id} ({len(preds_df)} rows)**")
                        st.dataframe(preds_df.head(100), use_container_width=True)
                    else:
                        st.warning("No predictions found for this run.")
                        
            conn.close()
        except Exception as e:
            st.error(f"Error accessing database: {e}")

# =====================================================
# 6. ABOUT & METHODOLOGY
# =====================================================
elif choice == "ℹ️ About & Methodology":
    st.header("ℹ️ About & Methodology")
    
    st.markdown("""
    ### Project Purpose
    This platform detects toxic comments using machine learning, providing both batch evaluation tools and a single-comment analysis interface with Explainable AI (XAI).
    
    ### Models & Architecture
    - **Version 1 Baseline:** Basic classical models (Logistic Regression, Random Forest, SVM) trained on raw text.
    - **Version 2 Classical:** Improved classical models leveraging better text cleaning and TF-IDF pipelines.
    - **Production Model (BERT V2):** A deep learning transformer model fine-tuned for sequence classification, chosen as the production standard due to its superior contextual understanding.
    
    ### Selection Criterion
    The **Toxic F1-Score** was used as the primary metric for selecting the production model. Since the dataset is imbalanced and the cost of missing toxic comments (False Negatives) or over-flagging benign comments (False Positives) is high, F1 provides a balanced harmonic mean of Precision and Recall.
    
    ### Capabilities
    - **LIME (Local Interpretable Model-agnostic Explanations):** Used to identify which specific words contributed most to a prediction.
    - **Adversarial Normalization:** An experimental feature that cleans leetspeak and symbol obfuscations before inference to defend against adversarial attacks.
    
    ### Limitations
    - **Fairness Analysis:** Currently disabled. Appropriate demographic and protected subgroup metadata is not available in the dataset.
    - **Drift Analysis:** Currently disabled. Requires temporal reference data and historical distribution tracking which is not yet implemented.
    """)
    st.info("No fabricated metrics or placeholder analytics are presented. Fairness and drift will be enabled only when rigorous data becomes available.")