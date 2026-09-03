# <center>Toxic Comment Detection ML Platform</center>

## Overview
This repository contains a machine learning platform for classifying toxic comments and evaluating multiple models. The system combines classical ML models (Logistic Regression, Random Forest, SVM) with a Streamlit dashboard and a FastAPI backend to provide real-time inference, model explainability (LIME), and bulk benchmarking capabilities.

**Note**: This is an academic/portfolio project. While it uses components like FastAPI, SQLite, and Streamlit, it is designed for evaluation and demonstration rather than enterprise-scale production deployment.

## Features
- **Real-Time Inference**: Submit a comment and receive toxicity predictions from selected models.
- **Explainable AI (XAI)**: Understand why a comment was flagged as toxic using LIME (Local Interpretable Model-Agnostic Explanations) feature importance.
- **Batch Benchmarking**: Upload a CSV of comments and ground truth labels to evaluate models in bulk.
- **Metrics Tracking**: Automatically calculates real Accuracy, Precision, Recall, F1, and F1_Toxic scores based on actual predictions.

### API Documentation
### 1. `GET /health`
Returns the status of the API.

### 2. `GET /models`
Returns a list of all models loaded in the registry.

### 3. `GET /production-model`
Returns the manifest of the active production model (e.g., v2_bert), including its selection metric and score.

### 4. `POST /predict`
Run real-time prediction using one or multiple models.

### Supported Models
- **Classical Models**: Support for Logistic Regression, Random Forest, and Linear SVM models from two different training iterations (Version 1 and Version 2).
- **BERT Model**: The codebase fully supports a HuggingFace BERT model for advanced classification, taking advantage of GPU acceleration if available. Note: The large model weight artifacts for BERT are excluded from the main Git repository via `.gitignore`.
- **Fairness & Drift Analytics**: These features are disabled as appropriate demographic and temporal metadata are absent in standard toxic comment datasets. No fabricated metrics are presented.

## Project Structure

```text
├── .gitignore
├── debug_bert.py
├── eval.ipynb
├── generate_metadata.py
├── init_db.py
├── project_audit_report.md
├── README.md
├── requirements.txt
├── run.py
├── setup_artifacts.py
├── test_bench.csv
├── toxic_comment_prediction_interface.ipynb
├── toxic_comment_prediction_interface_with_bert.ipynb
├── verify_artifacts.py
├── verify_original_models.py
├── verify_v2_lr_1_8.py
├── verify_v2_lr_1_8_extended.py
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── exceptions.py
│   │   ├── main.py
│   │   ├── middleware.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   └── __init__.py
│   ├── core/
│   │   ├── config_loader.py
│   │   ├── constants.py
│   │   ├── exceptions.py
│   │   ├── logger.py
│   │   ├── model_registry.py
│   │   ├── path_manager.py
│   │   ├── validators.py
│   │   └── __init__.py
├── artifacts/
│   ├── v1_lr/
│   │   ├── logistic_regression.pkl
│   │   ├── metadata.json
│   │   └── tfidf.pkl
│   ├── v1_rf/
│   │   ├── metadata.json
│   │   ├── random_forest.pkl
│   │   └── tfidf.pkl
│   ├── v1_svm/
│   │   ├── linear_svm.pkl
│   │   ├── metadata.json
│   │   └── tfidf.pkl
│   ├── v2_bert/
│   │   ├── config.json
│   │   ├── metadata.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── v2_lr/
│   │   ├── logistic_regression.pkl
│   │   ├── metadata.json
│   │   └── tfidf_vectorizer.pkl
│   ├── v2_svm/
│   │   ├── linear_svm.pkl
│   │   ├── metadata.json
│   │   └── tfidf_vectorizer.pkl
├── benchmarking/
│   ├── benchmark_runner.py
│   ├── benchmark_utils.py
│   ├── drift_detection.py
│   ├── ensemble_engine.py
│   ├── error_analysis.py
│   ├── fairness_engine.py
│   ├── inference_engine.py
│   ├── metrics_engine.py
│   ├── ranking_engine.py
│   ├── report_generator.py
│   ├── visualization_engine.py
│   └── __init__.py
├── benchmark_results/
│   ├── .gitkeep
│   ├── charts/
│   ├── csv/
│   ├── json/
│   ├── logs/
│   │   └── system.log
│   ├── raw_predictions/
│   ├── reports/
├── configs/
│   ├── benchmark_config.yaml
│   ├── fairness_config.yaml
│   ├── logging_config.yaml
│   └── model_config.yaml
├── dashboard/
│   └── app.py
├── data/
│   ├── processed/
│   │   ├── cleaned_data.csv
│   │   └── combined_data.csv
│   ├── raw/
│   │   ├── davidson_train.csv
│   │   ├── Jigsaw_train.csv
│   │   └── Unintended_train.csv
├── database/
│   ├── schema.sql
│   ├── storage_manager.py
│   ├── toxic_comments_benchmark.db
│   └── __init__.py
├── docs/
│   └── model_artifacts.md
├── logs/
│   └── system.log
├── temp_uploads/
│   └── comments_only.csv
├── tests/
│   ├── .gitkeep
│   ├── test_api.py
│   ├── test_benchmark.py
│   ├── test_integration.py
│   ├── test_storage.py
│   └── __init__.py
├── version_1/
│   ├── .gitkeep
│   ├── models/
│   │   ├── linear_svm.pkl
│   │   ├── logistic_regression.pkl
│   │   ├── random_forest.pkl
│   │   ├── tfidf.pkl
│   │   ├── X.pkl
│   │   └── y.pkl
│   ├── notebooks/
│   │   └── eda.ipynb
│   ├── outputs/
│   │   ├── eda/
│   │   │   └── class_distribution.png
│   ├── src/
│   │   ├── data_preprocessing.py
│   │   ├── data_preprocessing_2.py
│   │   ├── evaluation.py
│   │   ├── explanation.py
│   │   ├── feature_engineering.py
│   │   └── model_training.py
├── version_2/
│   ├── .gitkeep
│   ├── models/
│   │   ├── checkpoint.pkl
│   │   ├── linear_svm.pkl
│   │   ├── logistic_regression.pkl
│   │   ├── tfidf_vectorizer.pkl
│   │   ├── X_test.pkl
│   │   ├── X_train.pkl
│   │   ├── y_test.pkl
│   │   ├── y_train.pkl
│   │   ├── bert/
│   │   │   ├── config.json
│   │   │   ├── tokenizer.json
│   │   │   ├── tokenizer_config.json
│   │   │   ├── checkpoints/
│   │   │   ├── final/
│   │   │   │   ├── config.json
│   │   │   │   ├── model.safetensors
│   │   │   │   ├── tokenizer.json
│   │   │   │   ├── tokenizer_config.json
│   │   │   │   ├── tokenizer_repair_manifest.json
│   │   │   │   └── training_manifest.json
│   ├── outputs/
│   │   ├── final_classification_report.csv
│   │   ├── final_model_comparison.csv
│   │   └── model_comparison.csv
│   ├── src/
│   │   ├── bert_training.py
│   │   ├── evaluation.py
│   │   ├── feature_engineering.py
│   │   ├── final_evaluation.py
│   │   ├── fix_bert_tokenizer.py
│   │   └── model_training.py
```

### Directory Descriptions
- `app/` — FastAPI backend application.
- `app/api/` — API routes and endpoints.
- `app/core/` — Core configuration, logger, and model registry logic.
- `artifacts/` — Runtime deployment artifacts (models, tokenizers) for inference.
- `benchmarking/` — Engines for executing model benchmarks, computing metrics, and fairness/drift analytics.
- `benchmark_results/` — Output directory for benchmark run predictions, logs, and reports.
- `configs/` — YAML configuration files for benchmarking and models.
- `dashboard/` — Streamlit UI dashboard application.
- `data/` — Project datasets (raw and processed).
- `database/` — SQLite database and storage management for benchmark tracking.
- `docs/` — Project documentation.
- `logs/` — System logs.
- `tests/` — Test suite for API, benchmarking, and storage.
- `version_1/` & `version_2/` — Original training code, notebooks, and models for versions 1 and 2.
- `requirements.txt` — Python dependencies for the platform.
- `run.py` — Application entry point or runner.
- `README.md` — Project documentation.

## Environment & Serialization Compatibility
- **Primary Runtime**: Python 3.14.5 and `scikit-learn` 1.8.0
- **Version 1 Models**: Originally serialized using `scikit-learn` 1.7.2. These models run under 1.8.0 but will legitimately emit `InconsistentVersionWarning` upon loading due to scikit-learn's internal version checks. These warnings are intentionally preserved and not suppressed, as the original training artifacts remain untouched.
- **Version 2 Classical Models**: Fully compatible with `scikit-learn` 1.8.0.

## Installation & Setup

### Prerequisites
- Python 3.9+
- Recommended: Create a virtual environment

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI Backend
Start the backend server on `localhost:8000`:
```bash
uvicorn app.api.main:app --reload
```
You can view the interactive API documentation at `http://127.0.0.1:8000/docs`.

### 3. Run the Streamlit Dashboard
In a separate terminal, launch the UI:
```bash
streamlit run dashboard/app.py
```
The dashboard will open automatically in your browser (usually `http://localhost:8501`).

## Using the Batch Benchmark Runner
1. Navigate to the **Batch Benchmark Runner** tab in the Streamlit UI.
2. Upload a CSV file. The file must contain a comment text column (e.g., `comment`, `text`) and optionally a ground truth column (e.g., `target`, `is_toxic` with values `0` or `1`).
3. Click "Start Benchmark". The FastAPI backend will load the models, execute inference on every row, and calculate real metrics.
4. Results are displayed in the UI and permanently logged to `benchmark_results.db` using SQLite.

## Limitations & Known Issues
- **Memory Usage**: The system loads all requested `scikit-learn` and HuggingFace models into memory. On lower-end machines, this could cause memory pressure if too many large models are enabled simultaneously.
