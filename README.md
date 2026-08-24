# Toxic Comment Detection ML Platform

## Overview
This repository contains a machine learning platform for classifying toxic comments and evaluating multiple models. The system combines classical ML models (Logistic Regression, Random Forest, SVM) with a Streamlit dashboard and a FastAPI backend to provide real-time inference, model explainability (LIME), and bulk benchmarking capabilities.

**Note**: This is an academic/portfolio project. While it uses components like FastAPI, SQLite, and Streamlit, it is designed for evaluation and demonstration rather than enterprise-scale production deployment.

## Features
- **Real-Time Inference**: Submit a comment and receive toxicity predictions from selected models.
- **Explainable AI (XAI)**: Understand why a comment was flagged as toxic using LIME (Local Interpretable Model-Agnostic Explanations) feature importance.
- **Batch Benchmarking**: Upload a CSV of comments and ground truth labels to evaluate models in bulk.
- **Metrics Tracking**: Automatically calculates real Accuracy, Precision, Recall, F1, and F1_Toxic scores based on actual predictions.

### Supported Models
- **Classical Models**: Support for Logistic Regression, Random Forest, and Linear SVM models from two different training iterations (Version 1 and Version 2).
- **BERT Model**: The codebase fully supports a HuggingFace BERT model for advanced classification, taking advantage of GPU acceleration if available. Note: The large model weight artifacts for BERT are excluded from the main Git repository via `.gitignore`.
- **Fairness & Drift Analytics**: The dashboards for fairness and drift detection are placeholders. Genuine fairness analysis requires demographic metadata which is absent in standard toxic comment datasets.

## Project Structure
- `app/api/`: FastAPI backend serving predictions and benchmarking endpoints.
- `app/core/`: Contains the `ModelRegistry` which handles safe, lazy loading of ML models from the `artifacts/` deployment layer.
- `artifacts/`: Runtime deployment artifacts for inference. Contains copies of the required models, tokenizers, vectorizers, and metadata.
- `benchmarking/`: Engines for running batch inference and computing real metrics via `scikit-learn`.
- `dashboard/`: Streamlit UI for interaction and batch execution.
- `database/`: SQLite schema and asynchronous DB integration for storing benchmark results.
- `version_1/` & `version_2/`: (READ-ONLY) Original raw directories containing the full training artifacts, vectorizers, and historical dataset information. Preserved exactly as originally created.

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
