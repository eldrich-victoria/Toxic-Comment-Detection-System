# Model Artifacts

The repository intentionally does not commit trained model artifacts, vectorizers, tokenizers, or BERT weights.

These artifacts are stored locally under:

`artifacts/`

The application loads all runtime models exclusively from this directory.

The following models are supported:
- `v1_svm`
- `v1_lr`
- `v1_rf`
- `v2_svm`
- `v2_lr`
- `v2_bert`

The original training/model directories (`data/`, `version_1/`, `version_2/`) are preserved separately for historical purposes and are **not** used as runtime model sources by the application. 

The BERT model (`v2_bert`) is a fine-tuned project model and its large weight file is intentionally excluded from Git.

## Requirements

To run this application locally, you must provide the deployment artifacts in the `artifacts/` directory at the project root.

The expected file structure is as follows:

```text
artifacts/
├── v1_lr/
│   ├── logistic_regression.pkl
│   ├── metadata.json
│   └── tfidf.pkl
├── v1_rf/
│   ├── metadata.json
│   ├── random_forest.pkl
│   └── tfidf.pkl
├── v1_svm/
│   ├── linear_svm.pkl
│   ├── metadata.json
│   └── tfidf.pkl
├── v2_bert/
│   ├── config.json
│   ├── metadata.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── v2_lr/
│   ├── logistic_regression.pkl
│   ├── metadata.json
│   └── tfidf_vectorizer.pkl
└── v2_svm/
    ├── linear_svm.pkl
    ├── metadata.json
    └── tfidf_vectorizer.pkl
```

### Missing Artifacts
The application is designed to start even if artifacts are missing. Missing models will be marked as "Unavailable" in the Streamlit UI and the API `/models` endpoint. Trying to run predictions or benchmarks on a missing model will return a controlled error message rather than crashing the application.

## Runtime Versions
- **Primary Runtime:** Python 3.14.5, `scikit-learn` 1.8.0
- **Version 1 Models:** Originally serialized using `scikit-learn` 1.7.2. These models run under 1.8.0 but will natively emit an `InconsistentVersionWarning` upon loading due to scikit-learn's internal version checks. These warnings are intentionally preserved and not suppressed.
- **Version 2 Classical Models:** Fully compatible with `scikit-learn` 1.8.0.
