# Toxic Comment Benchmarking & Evaluation Framework

A production-grade, end-to-end Machine Learning Operations (MLOps) framework tailored specifically for evaluating, tracking, and diagnosing Toxic Comment Classification models.

## Phase 5 Final Architecture
- **Phase 1 (Foundation):** Core utilities, strict pathlib integrations, centralized YAML configs, and a robust SQLite schema.
- **Phase 2 (Inference & Storage):** Async benchmarking engine with dynamic model loading and transaction-safe telemetry logging.
- **Phase 3 (Analytics):** Metrics, Fairness, Drift, Error Analysis (LIME hook), and Ensemble engines.
- **Phase 4 (Reporting):** Visual generation (Plotly/Matplotlib) and automated PDF/DOCX/HTML report publishing.
- **Phase 5 (Production & API):** Asynchronous FastAPI backend providing programmatic access, alongside an interactive Streamlit UI for enterprise-grade ML infrastructure management.

## Setup Instructions

1. **Install Python Dependencies:**
   Ensure you are using Python 3.9+.
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize Database:**
   ```bash
   sqlite3 database/toxic_comments_benchmark.db < database/schema.sql
   ```

3. **Verify Environment:**
   Run the test suite to ensure all decoupled components are functional.
   ```bash
   pytest tests/
   ```

## Execution Instructions

**1. Start the API Backend (FastAPI):**
The API serves as the backbone for the frontend and programmatical access.
```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

**2. Start the Frontend Dashboard (Streamlit):**
In a new terminal, launch the dashboard.
```bash
streamlit run dashboard/app.py
```
Access the dashboard at `http://localhost:8501`.

## Testing Instructions

The project uses `pytest` for unit and integration testing.
- `tests/test_api.py`: Validates FastAPI endpoints, status codes, and payload schemas.
- `tests/test_benchmark.py`: Validates mathematical correctness of the metrics and analytics engines.
- `tests/test_storage.py`: Verifies transaction safety and sqlite mock interactions.
- `tests/test_integration.py`: End-to-end API pipeline test.

To execute the test suite:
```bash
pytest -v
```

## Deployment Instructions

### Docker & Cloud Deployment
1. **Containerization:** Write a `Dockerfile` multi-stage build. Stage 1 installs dependencies, Stage 2 copies the `app`, `database`, `benchmarking`, and `configs`.
2. **Orchestration:** Use `docker-compose` to spin up the FastAPI service on port 8000 and Streamlit on port 8501.
3. **Database Migration:** For enterprise deployments, replace the local SQLite implementation inside `database/storage_manager.py` with PostgreSQL using async drivers (`asyncpg`).
4. **CI/CD:** Integrate GitHub Actions to run the `pytest` suite automatically on pull requests.
